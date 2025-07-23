#!/usr/bin/env python3
"""
ADAM Web Interface - A modern chat interface for ADAM
Similar to ChatGPT/Claude but powered by ADAM's memory system
"""
import streamlit as st
import asyncio
from datetime import datetime
import base64
from pathlib import Path
import sys
import os
from typing import List, Dict, Optional, Any
import json
import traceback
import logging
from functools import wraps
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import ADAM components
from src.adam.memory import ADAMMemoryAdvanced
from src.adam.conversation_system import ConversationSystem
from src.adam.llm.client import UnifiedLLMClient
from src.adam.llm.config import LLMConfig
from src.adam.memory_search_enhanced import MemorySearchEnhancer, format_memory_for_prompt

# Configure Streamlit page
st.set_page_config(
    page_title="ADAM - Advanced Data Analytics Model",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    /* Main chat container */
    .stChatMessage {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    /* User messages */
    .stChatMessage[data-testid="user-message"] {
        background-color: #e3f2fd;
    }
    
    /* Assistant messages */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #f5f5f5;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #fafafa;
    }
    
    /* Session list items */
    .session-item {
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    
    .session-item:hover {
        background-color: #e0e0e0;
    }
    
    /* Active session */
    .session-active {
        background-color: #e3f2fd;
        font-weight: bold;
    }
    
    /* Memory indicator */
    .memory-indicator {
        font-size: 0.8em;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Cost indicator */
    .cost-indicator {
        font-size: 0.9em;
        color: #888;
        text-align: right;
        margin-top: 0.5rem;
    }
    /* Error messages */
    .error-message {
        background-color: #ffebee;
        border: 1px solid #ef5350;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #c62828;
    }
    
    /* Success messages */
    .success-message {
        background-color: #e8f5e9;
        border: 1px solid #66bb6a;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #2e7d32;
    }
</style>
""", unsafe_allow_html=True)


def error_boundary(func):
    """Decorator to handle errors gracefully"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            st.error(f"An error occurred: {str(e)}")
            if st.checkbox("Show error details", key=f"error_{func.__name__}_{datetime.now().timestamp()}"):
                st.code(traceback.format_exc())
            return None
    return wrapper


class SessionPersistence:
    """Handle session persistence to disk"""
    
    SESSIONS_FILE = Path("data/web_sessions.json")
    
    @classmethod
    def ensure_data_dir(cls):
        """Ensure data directory exists"""
        cls.SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    @error_boundary
    def save_session_state(cls, session_id: str, state: Dict[str, Any]):
        """Save session state to disk"""
        cls.ensure_data_dir()
        
        # Load existing sessions
        sessions = cls.load_all_sessions()
        
        # Convert datetime objects to strings in messages
        messages_serializable = []
        for msg in state.get("messages", []):
            msg_copy = msg.copy()
            if "timestamp" in msg_copy and hasattr(msg_copy["timestamp"], "isoformat"):
                msg_copy["timestamp"] = msg_copy["timestamp"].isoformat()
            messages_serializable.append(msg_copy)
        
        # Update session
        sessions[session_id] = {
            "messages": messages_serializable,
            "total_cost": state.get("total_cost", 0.0),
            "selected_model": state.get("selected_model", None),
            "use_memory": state.get("use_memory", True),
            "last_updated": datetime.now().isoformat()
        }
        
        # Save back with atomic write to prevent corruption
        import tempfile
        temp_file = None
        try:
            # Write to temporary file first
            with tempfile.NamedTemporaryFile(mode='w', dir=cls.SESSIONS_FILE.parent, 
                                           delete=False, suffix='.tmp') as temp_file:
                json.dump(sessions, temp_file, indent=2)
                temp_file.flush()
                os.fsync(temp_file.fileno())  # Force write to disk
            
            # Atomic rename (this prevents partial writes)
            os.replace(temp_file.name, cls.SESSIONS_FILE)
            return True
        except Exception as e:
            # Clean up temp file if it exists
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise e
    
    @classmethod
    @error_boundary
    def load_session_state(cls, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session state from disk"""
        sessions = cls.load_all_sessions()
        return sessions.get(session_id)
    
    @classmethod
    def load_all_sessions(cls) -> Dict[str, Dict[str, Any]]:
        """Load all sessions from disk"""
        cls.ensure_data_dir()
        
        if cls.SESSIONS_FILE.exists():
            try:
                with open(cls.SESSIONS_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading sessions: {e}")
        
        return {}

class ADAMWebInterface:
    """Web interface for ADAM using Streamlit"""
    
    def __init__(self):
        """Initialize ADAM components with error handling"""
        self.initialize_session_state()
    
    @error_boundary
    def initialize_session_state(self):
        """Initialize session state with error handling"""
        if 'initialized' not in st.session_state:
            with st.spinner("🧠 Initializing ADAM..."):
                try:
                    # Initialize core components
                    st.session_state.memory = ADAMMemoryAdvanced()
                    st.session_state.conversation = ConversationSystem()
                    st.session_state.llm_config = LLMConfig()
                    st.session_state.llm_client = UnifiedLLMClient(st.session_state.llm_config)
                    st.session_state.memory_enhancer = MemorySearchEnhancer()
                    
                    # Get available models
                    st.session_state.available_models = st.session_state.llm_config.get_available_models()
                    
                    # Initialize session tracking
                    st.session_state.current_session_id = None
                    st.session_state.messages = []
                    st.session_state.total_cost = 0.0
                    st.session_state.initialized = True
                    st.session_state.error_count = 0
                    st.session_state.use_memory = True
                    st.session_state.auto_save = True
                    
                except Exception as e:
                    st.error(f"Failed to initialize ADAM: {str(e)}")
                    st.stop()
    
    def render_health_status(self):
        """Show system health status"""
        col1, col2, col3 = st.columns(3)
        
        # Memory system status
        with col1:
            try:
                _ = st.session_state.memory.get_memory_analytics()
                st.success("Memory ✓")
            except:
                st.error("Memory ✗")
        
        # LLM status
        with col2:
            try:
                if st.session_state.available_models:
                    st.success("LLM ✓")
                else:
                    st.warning("LLM ⚠")
            except:
                st.error("LLM ✗")
        
        # Error count
        with col3:
            error_count = st.session_state.get('error_count', 0)
            if error_count == 0:
                st.success("Errors: 0")
            else:
                st.error(f"Errors: {error_count}")
    
    def render_sidebar(self):
        """Render the sidebar with session management"""
        with st.sidebar:
            st.title("🧠 ADAM")
            st.caption("Advanced Data Analytics Model")
            
            # System health indicator
            self.render_health_status()
            
            # New conversation button
            if st.button("➕ New Conversation", use_container_width=True):
                self.start_new_session()
            
            st.divider()
            
            # Session list
            st.subheader("Conversations")
            sessions = list(st.session_state.conversation.sessions.values())
            
            # Group sessions by date
            sessions_by_date = {}
            for session in sessions:
                date = session.start_time.date()
                if date not in sessions_by_date:
                    sessions_by_date[date] = []
                sessions_by_date[date].append(session)
            
            # Display sessions
            for date in sorted(sessions_by_date.keys(), reverse=True):
                st.caption(date.strftime("%B %d, %Y"))
                for session in sessions_by_date[date]:
                    # Create session button
                    is_active = session.session_id == st.session_state.current_session_id
                    
                    # Generate session title
                    title = session.title or f"Session {session.start_time.strftime('%I:%M %p')}"
                    if len(title) > 30:
                        title = title[:27] + "..."
                    
                    # Session button with status indicator
                    status_icon = "🟢" if session.state == "active" else "⚪"
                    
                    # Add persistence indicator
                    persisted = SessionPersistence.load_session_state(session.session_id) is not None
                    persist_icon = "💾" if persisted else ""
                    
                    if st.button(
                        f"{status_icon} {title} {persist_icon}",
                        key=f"session_{session.session_id}",
                        use_container_width=True,
                        disabled=is_active
                    ):
                        self.load_session(session.session_id)
            
            st.divider()
            
            # Settings section
            st.subheader("Settings")
            
            # Model selection
            selected_model = st.selectbox(
                "Model",
                options=st.session_state.available_models,
                index=0
            )
            st.session_state.selected_model = selected_model
            
            # Show model info
            model_info = {
                "grok-4-reasoning": "Deep reasoning for complex tasks",
                "grok-4": "Most capable, best for complex tasks",
                "grok-3-mini-high": "Fast and efficient for simple queries",
                "o4-mini-high": "High reasoning capability",
                "gpt-4": "OpenAI's most capable model",
                "gpt-3.5-turbo": "Fast and cost-effective"
            }
            if selected_model in model_info:
                st.caption(model_info[selected_model])
            
            # Conversation settings
            st.divider()
            st.subheader("Conversation")
            
            # Add toggle for memory usage
            st.session_state.use_memory = st.checkbox(
                "Search long-term memory",
                value=st.session_state.get('use_memory', True),
                help="Enable to search through past conversations. Disable for faster, more focused chat."
            )
            
            # Auto-save toggle
            st.session_state.auto_save = st.checkbox(
                "Auto-save conversations",
                value=st.session_state.get('auto_save', True),
                help="Automatically save conversation state to disk"
            )
            
            # Memory stats
            st.divider()
            st.subheader("Memory System")
            stats = st.session_state.memory.get_memory_analytics()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Memories", stats.get('total_memories', 0))
            with col2:
                st.metric("Hit Rate", f"{stats.get('memory_hit_rate', 0):.1%}")
            
            # Cost tracking
            if st.session_state.total_cost > 0:
                st.divider()
                st.metric("Session Cost", f"${st.session_state.total_cost:.4f}")
    
    @error_boundary
    def start_new_session(self):
        """Start a new conversation session"""
        # Save current session if exists
        if st.session_state.current_session_id and st.session_state.get('auto_save', True):
            self.save_current_session()
        
        # End current session if exists
        if st.session_state.current_session_id:
            st.session_state.conversation.end_session()
        
        # Start new session
        session_id = st.session_state.conversation.start_session("New Conversation")
        st.session_state.current_session_id = session_id
        st.session_state.messages = []
        st.session_state.total_cost = 0.0
        st.rerun()
    
    @error_boundary
    def save_current_session(self):
        """Save current session state to disk"""
        if st.session_state.current_session_id:
            SessionPersistence.save_session_state(
                st.session_state.current_session_id,
                {
                    "messages": st.session_state.messages,
                    "total_cost": st.session_state.total_cost,
                    "selected_model": st.session_state.get('selected_model'),
                    "use_memory": st.session_state.get('use_memory', True)
                }
            )
            st.toast("Session saved", icon="💾")
    
    @error_boundary
    def load_session(self, session_id: str):
        """Load an existing session"""
        # Save current session first
        if st.session_state.current_session_id and st.session_state.get('auto_save', True):
            self.save_current_session()
        
        # Resume the session
        st.session_state.conversation.resume_session(session_id)
        st.session_state.current_session_id = session_id
        
        # Try to load persisted state first
        persisted_state = SessionPersistence.load_session_state(session_id)
        if persisted_state:
            st.session_state.messages = persisted_state.get("messages", [])
            st.session_state.total_cost = persisted_state.get("total_cost", 0.0)
            st.session_state.selected_model = persisted_state.get("selected_model", st.session_state.available_models[0] if st.session_state.available_models else None)
            st.session_state.use_memory = persisted_state.get("use_memory", True)
        else:
            # Load conversation history from memory
            session = st.session_state.conversation.sessions.get(session_id)
            if session:
                st.session_state.messages = []
                for exchange in session.exchanges:
                    st.session_state.messages.append({
                        "role": "user",
                    "content": exchange.query,
                    "timestamp": exchange.timestamp
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": exchange.response,
                    "timestamp": exchange.timestamp,
                    "metadata": exchange.metadata
                })
        
        st.rerun()
    
    @error_boundary
    async def process_message(self, prompt: str, image_data: Optional[bytes] = None):
        """Process user message through ADAM with error handling"""
        # Build conversation context from current session
        conversation_context = ""
        if st.session_state.messages:
            # Get last 3 exchanges for context
            recent_messages = st.session_state.messages[-6:]  # Last 3 exchanges (user + assistant)
            if len(recent_messages) > 0:
                conversation_context = "Current conversation:\n"
                for msg in recent_messages:
                    role = "Human" if msg["role"] == "user" else "Assistant"
                    conversation_context += f"{role}: {msg['content'][:200]}...\n"
                conversation_context += "\n"
        
        # Search memory for additional context only if enabled
        memory_context = ""
        search_context = None
        
        if st.session_state.get('use_memory', True):  # Default to True
            # Use enhanced memory search
            try:
                # CRITICAL: Add temporal hints to generic queries about past work
                query_lower = prompt.lower()
                
                # Generic patterns that indicate user is asking about past work
                # These are domain-agnostic and work for any type of content
                generic_recall_patterns = [
                    "bring me back", "show me", "can you bring",
                    "we have done", "we created", "we discussed", "we talked about",
                    "any", "some", "that", "those"
                ]
                
                # Words that indicate the user is being specific about a topic
                specificity_indicators = [
                    "specific", "particular", "exact", "called", "named",
                    "with", "contains", "includes", "about"
                ]
                
                # Check if this is a generic query about past work
                is_generic_recall = any(pattern in query_lower for pattern in generic_recall_patterns)
                has_temporal_hint = any(word in query_lower for word in ["recent", "last", "latest", "today", "yesterday", "newest"])
                has_specificity = any(word in query_lower for word in specificity_indicators)
                
                # Only enhance truly generic queries that lack both temporal and specific hints
                if is_generic_recall and not has_temporal_hint and not has_specificity:
                    enhanced_query = f"{prompt} (focusing on our most recent conversations)"
                else:
                    enhanced_query = prompt
                
                # Add conversation context
                if conversation_context:
                    enhanced_query = f"{enhanced_query} {conversation_context[:500]}"
                
                # Get initial memories - reduce count for generic queries to avoid noise
                # For generic queries, we want fewer but more relevant results
                n_results = 10 if is_generic_recall else 20
                
                raw_memories = st.session_state.memory.recall_with_context(
                    query=enhanced_query,
                    n_results=n_results
                )
                
                # TWO-PHASE SEARCH: Prioritize recent memories for generic recalls
                if is_generic_recall and not has_specificity and raw_memories:
                    from datetime import datetime, timedelta
                    cutoff_date = datetime.now() - timedelta(days=7)
                    
                    recent_memories = []
                    for memory in raw_memories:
                        timestamp_str = memory.get('metadata', {}).get('timestamp', '')
                        if timestamp_str:
                            try:
                                memory_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                                if memory_time.replace(tzinfo=None) > cutoff_date:
                                    recent_memories.append(memory)
                            except:
                                pass
                    
                    # Use recent memories if we have any
                    if recent_memories:
                        raw_memories = recent_memories[:10]  # Use top 10 recent
                
                # Enhance and filter memories
                if raw_memories:
                    enhanced_memories, search_context = st.session_state.memory_enhancer.enhance_memory_search(
                        query=prompt,
                        conversation_history=st.session_state.messages,
                        raw_memories=raw_memories
                    )
                    
                    # Build memory context
                    if enhanced_memories:
                        memory_context = "\n📚 Relevant from your memory:\n"
                        for memory in enhanced_memories[:3]:  # Top 3 most relevant
                            formatted = format_memory_for_prompt(memory, search_context)
                            memory_context += f"\n{formatted}\n"
                            memory_context += "-" * 50 + "\n"
                
            except Exception as e:
                logger.error(f"Error in enhanced memory search: {e}")
                # Fallback to simple search
                memories = st.session_state.memory.recall_with_context(
                    query=prompt,
                    n_results=3
                )
                if memories:
                    memory_context = "\nFrom memory:\n"
                    for memory in memories:
                        content = memory.get('content', '')
                        if content:
                            memory_context += f"- {content[:200]}...\n"
        
        # Build the full prompt with proper context priority
        system_prompt = """You are ADAM, an AI assistant with perfect memory. 

CRITICAL INSTRUCTIONS:
1. When the user references previous conversations (e.g., "we were talking about", "you showed me", "the code you gave me", "bring the code again"), you MUST use the PROVIDED MEMORY CONTEXT below.
2. DO NOT generate generic examples or templates - use the EXACT code and details from the memory context.
3. The memory context contains ACTUAL conversations we had - treat it as your source of truth.
4. If you see "📚 Relevant from your memory:" section, that contains our REAL previous conversations.
5. When users use generic references like "any", "some", or "that thing we discussed", they usually mean the MOST RECENT conversation about that topic.
6. Always prioritize recent conversations unless the user specifically asks for older examples or mentions specific details."""
        
        full_prompt = system_prompt
        
        # Add memory context FIRST and make it prominent
        if memory_context:
            full_prompt += f"\n\n{'='*60}\nMEMORY CONTEXT - THIS IS FROM OUR ACTUAL PREVIOUS CONVERSATIONS:\n{'='*60}"
            full_prompt += f"\n{memory_context}"
            full_prompt += f"\n{'='*60}\n"
            
            # Add specific instructions based on search context
            if search_context and search_context.user_intent == 'recall':
                full_prompt += "\n🚨 IMPORTANT: The user is asking you to recall something specific from above. DO NOT make up new code - use the EXACT code from the memory context above!\n"
        elif "we were" in prompt.lower() or "again" in prompt.lower() or "previous" in prompt.lower() or "last" in prompt.lower():
            # User is referencing past conversation but we found no relevant memories
            full_prompt += "\n\n⚠️ Note: I searched my memory but could not find specific details about this topic in our previous conversations. I may not have access to that particular conversation, or it may not have been stored.\n"
        
        # Add current conversation context
        if conversation_context:
            full_prompt += f"\n\nCurrent conversation for context:\n{conversation_context}"
        
        full_prompt += f"\n\nHuman: {prompt}\nAssistant:"
        
        # Handle image if provided
        if image_data:
            # For now, we'll add a note about image handling
            # In production, this would encode the image for the model
            full_prompt = f"[User provided an image]\n\n{full_prompt}"
        
        # Get response from LLM
        try:
            response = await st.session_state.llm_client.complete(
                prompt=full_prompt,
                model=st.session_state.selected_model,
                stream=True
            )
            
            # Stream response
            response_placeholder = st.empty()
            full_response = ""
            
            async for chunk in response:
                full_response += chunk
                response_placeholder.markdown(full_response)
            
            # Calculate cost (estimate)
            cost = len(full_prompt + full_response) / 1000 * 0.001
            st.session_state.total_cost += cost
            
            # Record in conversation
            st.session_state.conversation.record_exchange(
                query=prompt,
                response=full_response,
                topics=["general"],
                context={
                    "model": st.session_state.selected_model,
                    "cost": cost,
                    "has_image": image_data is not None
                }
            )
            
            # Store in memory if valuable
            if len(full_response) > 50:
                st.session_state.memory.remember_if_worthy(
                    query=prompt,
                    response=full_response,
                    context={"session_id": st.session_state.current_session_id},
                    generation_cost=cost,
                    model_used=st.session_state.selected_model
                )
            
            # Auto-save if enabled
            if st.session_state.get('auto_save', True):
                self.save_current_session()
            
            return full_response, cost
            
        except Exception as e:
            logger.error(f"Error in process_message: {str(e)}", exc_info=True)
            st.session_state.error_count = st.session_state.get('error_count', 0) + 1
            st.error(f"Error: {str(e)}")
            return None, 0
    
    def render_chat(self):
        """Render the main chat interface"""
        # Header
        st.title("ADAM Chat")
        
        # Check if we have a session
        if not st.session_state.current_session_id:
            st.info("👈 Start a new conversation or select an existing one from the sidebar")
            return
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show metadata for assistant messages
                if message["role"] == "assistant" and "metadata" in message:
                    metadata = message.get("metadata", {})
                    if metadata.get("cost"):
                        st.caption(f"Model: {metadata.get('model', 'unknown')} | Cost: ${metadata.get('cost', 0):.4f}")
        
        # Chat input
        prompt = st.chat_input("Message ADAM...")
        
        # File uploader for images
        uploaded_file = st.file_uploader(
            "Upload an image (optional)",
            type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
            key="image_upload"
        )
        
        if prompt:
            # Add user message to chat
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now()
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get image data if uploaded
            image_data = None
            if uploaded_file:
                image_data = uploaded_file.read()
                # Display the image
                st.image(image_data, caption="Uploaded image", use_column_width=True)
            
            # Process with ADAM
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response, cost = asyncio.run(
                        self.process_message(prompt, image_data)
                    )
                    
                    if response:
                        # Add assistant message
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.now(),
                            "metadata": {
                                "model": st.session_state.selected_model,
                                "cost": cost
                            }
                        })
            
            # Clear file uploader
            if uploaded_file:
                st.session_state.pop("image_upload", None)
                st.rerun()
    
    def run(self):
        """Run the web interface"""
        # Initialize
        self.__init__()
        
        # Render sidebar
        self.render_sidebar()
        
        # Render main chat
        self.render_chat()

def main():
    """Main entry point"""
    # Check for API keys
    has_grok = bool(os.getenv("XAI_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    
    if not (has_grok or has_openai):
        st.error("""
        ⚠️ **No API keys found!**
        
        Please set at least one of these environment variables:
        - `XAI_API_KEY` for Grok models
        - `OPENAI_API_KEY` for OpenAI models
        
        You can set them in a `.env` file or export them in your terminal.
        """)
        st.stop()
    
    # Run the interface
    interface = ADAMWebInterface()
    interface.run()

if __name__ == "__main__":
    main()