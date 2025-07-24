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
            
            # Model selection moved to top of chat interface
            # (keeping this section for other settings)
            
            # Conversation settings
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
            # Include ALL messages from current conversation
            all_messages = st.session_state.messages
            conversation_context = "Current conversation history:\n"
            for i, msg in enumerate(all_messages):
                role = "Human" if msg["role"] == "user" else "Assistant"
                # Include full content for recent messages, truncate older ones
                if i >= len(all_messages) - 4:  # Last 2 exchanges
                    conversation_context += f"{role}: {msg['content']}\n"
                else:
                    conversation_context += f"{role}: {msg['content'][:300]}...\n"
            conversation_context += "\n"
        
        # Search memory for additional context only if enabled
        memory_context = ""
        search_context = None
        
        # Skip memory search for pure image analysis queries
        is_image_analysis = image_data and any(phrase in prompt.lower() for phrase in [
            "explain this image", "what's in this image", "analyze this image",
            "what do you see", "describe this image", "look at this"
        ])
        
        if st.session_state.get('use_memory', True) and not is_image_analysis:  # Default to True
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
                
                # Use Advanced RAG with temporal scoring for better retrieval
                raw_memories = st.session_state.memory.recall_with_advanced_rag(
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
                    
                    # Build memory context - be more selective
                    if enhanced_memories:
                        # Only include memories with high relevance
                        # Increase threshold when image is present to avoid confusion
                        relevance_threshold = 0.6 if image_data else 0.4
                        relevant_memories = [m for m in enhanced_memories if m.get('combined_score', 0) > relevance_threshold]
                        
                        if relevant_memories:
                            # When image is present, clearly separate memory context
                            if image_data:
                                memory_context = "\n📚 [PAST MEMORY - NOT FROM CURRENT IMAGE]:\n"
                                memory_context += "Note: The following is from previous conversations, not from the image you just shared.\n"
                            else:
                                memory_context = "\n📚 Relevant from your memory:\n"
                            
                            # Limit memories more strictly when image is present
                            memory_limit = 1 if image_data else 2
                            for memory in relevant_memories[:memory_limit]:
                                formatted = format_memory_for_prompt(memory, search_context)
                                memory_context += f"\n{formatted}\n"
                                memory_context += "-" * 50 + "\n"
                
            except Exception as e:
                logger.error(f"Error in enhanced memory search: {e}")
                # Fallback to simple search with advanced RAG
                memories = st.session_state.memory.recall_with_advanced_rag(
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
        # Count messages in current conversation
        user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
        message_number = len(user_messages) + 1  # +1 for the current message
        
        system_prompt = f"""You are ADAM, a helpful AI coworker. This is message #{message_number} in our current conversation.

IMPORTANT BEHAVIORAL RULES:
- DO NOT introduce yourself or greet the user (no "Hi!", "I'm ADAM", etc.)
- DO NOT use emojis unless specifically requested
- Be direct and conversational, like a coworker sitting next to them
- Only introduce yourself if explicitly asked "who are you?" or similar
- For message #2 and beyond, continue the conversation naturally without greetings

MEMORY INSTRUCTIONS:
1. When the user references previous conversations, use the PROVIDED MEMORY CONTEXT below.
2. DO NOT generate generic examples - use EXACT code and details from memory.
3. The memory context contains ACTUAL conversations - treat it as truth.
4. "📚 Relevant from your memory:" contains REAL previous conversations.
5. Generic references ("any", "some", "that thing") usually mean MOST RECENT.
6. Prioritize recent conversations unless user asks for older examples."""
        
        full_prompt = system_prompt
        
        # Add current conversation context FIRST (highest priority)
        if conversation_context:
            full_prompt += f"\n\n{'='*60}\nCURRENT CONVERSATION CONTEXT:\n{'='*60}"
            full_prompt += f"\n{conversation_context}"
            full_prompt += f"{'='*60}\n"
        
        # Add memory context SECOND
        if memory_context:
            if image_data:
                # Special handling when image is present
                full_prompt += f"\n\n{'='*60}\n⚠️ MEMORY CONTEXT - FROM PAST CONVERSATIONS (NOT THE CURRENT IMAGE):\n{'='*60}"
                full_prompt += f"\n{memory_context}"
                full_prompt += f"\n{'='*60}\n"
                full_prompt += "\n🚨 IMPORTANT: You have been provided with an image. Focus on what's IN THE IMAGE, not what's in the memory above. The memory is only for additional context if needed.\n"
            else:
                full_prompt += f"\n\n{'='*60}\nMEMORY CONTEXT - THIS IS FROM OUR ACTUAL PREVIOUS CONVERSATIONS:\n{'='*60}"
                full_prompt += f"\n{memory_context}"
                full_prompt += f"\n{'='*60}\n"
            
            # Add specific instructions based on search context
            if search_context and search_context.user_intent == 'recall' and not image_data:
                full_prompt += "\n🚨 IMPORTANT: The user is asking you to recall something specific from above. DO NOT make up new code - use the EXACT code from the memory context above!\n"
        elif "we were" in prompt.lower() or "again" in prompt.lower() or "previous" in prompt.lower() or "last" in prompt.lower():
            # User is referencing past conversation but we found no relevant memories
            full_prompt += "\n\n⚠️ Note: I searched my memory but could not find specific details about this topic in our previous conversations. I may not have access to that particular conversation, or it may not have been stored.\n"
        
        # Don't duplicate conversation context - it's already added above
        
        full_prompt += f"\n\nHuman: {prompt}\nAssistant:"
        
        # Get response from LLM
        try:
            # For automatic routing, we need a non-streaming call first to get routing info
            if st.session_state.selected_model == "automatic":
                # Get initial response without streaming to capture routing decision
                initial_response = await st.session_state.llm_client.complete(
                    prompt=full_prompt,
                    model=st.session_state.selected_model,
                    stream=False,
                    image_data=image_data,
                    max_tokens=1500  # Get full response
                )
                
                # Extract routing information
                actual_model = initial_response.model
                routing_info = initial_response.raw_response.get('routing_decision') if initial_response.raw_response else None
                full_response = initial_response.content
                
                # Display the response
                st.markdown(full_response)
            else:
                # Regular streaming for non-automatic models
                actual_model = st.session_state.selected_model
                routing_info = None
                
                response = await st.session_state.llm_client.complete(
                    prompt=full_prompt,
                    model=st.session_state.selected_model,
                    stream=True,
                    image_data=image_data
                )
                
                # Stream response
                response_placeholder = st.empty()
                full_response = ""
                
                async for chunk in response:
                    full_response += chunk
                    response_placeholder.markdown(full_response)
            
            # Calculate cost (estimate with image handling)
            # Different models have different image pricing
            model_config = st.session_state.llm_config.get_model_config(st.session_state.selected_model)
            
            # Calculate token counts
            input_tokens = len(full_prompt) / 4  # Rough estimate: 1 token ≈ 4 chars
            output_tokens = len(full_response) / 4
            
            # Different pricing for different models
            if st.session_state.selected_model == "grok-2-vision-1212" or ("grok" in st.session_state.selected_model and model_config.supports_vision):
                # Grok vision pricing: $2/million input, $10/million output
                input_cost = (input_tokens / 1_000_000) * 2.00
                output_cost = (output_tokens / 1_000_000) * 10.00
                # Images consume tokens based on tiles (256 tokens per tile + 1 extra tile)
                if image_data:
                    # Maximum 6 tiles, so max 1792 tokens per image
                    # For typical images, estimate 4-5 tiles = ~1280 tokens
                    image_tokens = 1280
                    input_cost += (image_tokens / 1_000_000) * 2.00
                cost = input_cost + output_cost
            elif "gpt-4" in st.session_state.selected_model:
                # GPT-4 pricing
                base_cost_per_1k = model_config.cost_per_1k_tokens if model_config else 0.03
                text_cost = (input_tokens + output_tokens) / 1000 * base_cost_per_1k
                image_cost = 0.01 if image_data else 0  # GPT-4V image pricing
                cost = text_cost + image_cost
            else:
                # Default pricing for other models
                base_cost_per_1k = model_config.cost_per_1k_tokens if model_config else 0.001
                cost = (input_tokens + output_tokens) / 1000 * base_cost_per_1k
            st.session_state.total_cost += cost
            
            # actual_model and routing_info are already set above
            
            context = {
                "model": actual_model,
                "requested_model": st.session_state.selected_model,
                "cost": cost,
                "has_image": image_data is not None
            }
            
            if routing_info:
                context["routing_decision"] = routing_info
            
            st.session_state.conversation.record_exchange(
                query=prompt,
                response=full_response,
                topics=["general"],
                context=context
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
            
            # Return response with metadata
            return full_response, cost, {"model": actual_model, "routing_decision": routing_info}
            
        except Exception as e:
            logger.error(f"Error in process_message: {str(e)}", exc_info=True)
            st.session_state.error_count = st.session_state.get('error_count', 0) + 1
            st.error(f"Error: {str(e)}")
            return None, 0
    
    def render_chat(self):
        """Render the main chat interface"""
        # Header with model selector
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.title("ADAM Chat")
        
        with col2:
            # Model selector at the top
            model_info = {
                "automatic": "🤖 Smart routing (recommended)",
                "grok-4-reasoning": "Deep reasoning 🖼️",
                "grok-4": "Most capable 🖼️",
                "grok-3-mini-fast": "⚡ Super fast",
                "grok-3-mini-high": "Fast & efficient",
                "grok-2-vision-1212": "Vision optimized 🖼️",
                "o4-mini-high": "High reasoning",
                "gpt-4": "OpenAI GPT-4 🖼️",
                "gpt-3.5-turbo": "Fast & cheap"
            }
            
            # Ensure automatic is first in the list
            available_models = st.session_state.available_models.copy()
            if "automatic" in available_models:
                available_models.remove("automatic")
                available_models.insert(0, "automatic")
            
            # Add vision indicator to model names in dropdown
            model_display_names = []
            for model in available_models:
                if model == "automatic":
                    model_display_names.append("🤖 Automatic (Smart Routing)")
                else:
                    config = st.session_state.llm_config.get_model_config(model)
                    if config and config.supports_vision:
                        model_display_names.append(f"{model} 🖼️")
                    else:
                        model_display_names.append(model)
            
            # Handle model selection with automatic as default
            current_model = st.session_state.get('selected_model', 'automatic')
            
            # Find current model in display names
            try:
                if current_model == "automatic":
                    current_index = 0  # automatic is first
                else:
                    # Find the index of current model
                    for i, model in enumerate(available_models):
                        if model == current_model:
                            current_index = i
                            break
                    else:
                        current_index = 0  # Default to automatic
            except:
                current_index = 0
            
            selected_display = st.selectbox(
                "Model",
                options=model_display_names,
                index=current_index,
                key="top_model_selector",
                label_visibility="collapsed"
            )
            
            # Extract actual model name
            if selected_display.startswith("🤖 Automatic"):
                selected_model = "automatic"
            else:
                selected_model = selected_display.replace(" 🖼️", "")
            st.session_state.selected_model = selected_model
            st.caption(model_info.get(selected_model, ""))
        
        with col3:
            # Session cost
            if st.session_state.total_cost > 0:
                st.metric("Cost", f"${st.session_state.total_cost:.4f}")
        
        # Check if we have a session
        if not st.session_state.current_session_id:
            st.info("👈 Start a new conversation or select an existing one from the sidebar")
            return
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show image if user message had one
                if message["role"] == "user" and message.get("has_image"):
                    st.caption("📎 Image was attached to this message")
                
                # Show metadata for assistant messages
                if message["role"] == "assistant" and "metadata" in message:
                    metadata = message.get("metadata", {})
                    model_used = metadata.get('model', 'unknown')
                    cost = metadata.get('cost', 0)
                    
                    # Show routing info if automatic model was used
                    routing_info = metadata.get('routing_decision')
                    if routing_info:
                        st.caption(f"🤖 Auto-selected: {routing_info['selected_model']} (complexity: {routing_info['complexity']}) | Cost: ${cost:.4f}")
                        
                        # Show routing reasoning in expander
                        with st.expander("Why this model?", expanded=False):
                            st.write(f"**Complexity:** {routing_info['complexity']}")
                            st.write(f"**Confidence:** {routing_info['confidence']:.1%}")
                            if routing_info.get('indicators'):
                                st.write(f"**Indicators:** {', '.join(routing_info['indicators'])}")
                            if routing_info.get('reasoning'):
                                st.write(f"**Reasoning:** {', '.join(routing_info['reasoning'])}")
                    elif cost > 0:
                        st.caption(f"Model: {model_used} | Cost: ${cost:.4f}")
        
        # Chat input
        prompt = st.chat_input("Message ADAM...")
        
        # File uploader for images - show if current model supports vision or is automatic
        model_config = st.session_state.llm_config.get_model_config(st.session_state.selected_model)
        supports_vision = (model_config and model_config.supports_vision) or st.session_state.selected_model == "automatic"
        
        if supports_vision:
            uploaded_file = st.file_uploader(
                "Upload an image (optional) 🖼️",
                type=['png', 'jpg', 'jpeg', 'gif', 'webp'],
                key="image_upload",
                help=f"{st.session_state.selected_model} supports image analysis" if st.session_state.selected_model != "automatic" else "Automatic mode will select a vision model for image analysis"
            )
        else:
            uploaded_file = None
            if st.session_state.get('image_upload'):
                st.info(f"ℹ️ {st.session_state.selected_model} doesn't support images. Switch to a vision model (marked with 🖼️) or use Automatic mode.")
        
        if prompt:
            # Get image data if uploaded
            image_data = None
            if uploaded_file:
                image_data = uploaded_file.read()
            
            # Add user message to chat (with image indicator if present)
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now(),
                "has_image": image_data is not None
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
                if image_data:
                    st.image(image_data, caption="Attached image", use_container_width=True)
            
            # Process with ADAM
            with st.chat_message("assistant"):
                # Show different spinner text for automatic mode
                spinner_text = "Thinking..." if st.session_state.selected_model != "automatic" else "Selecting best model and thinking..."
                
                with st.spinner(spinner_text):
                    result = asyncio.run(
                        self.process_message(prompt, image_data)
                    )
                    
                    # Handle both old and new return formats
                    if isinstance(result, tuple) and len(result) == 3:
                        response, cost, metadata = result
                    else:
                        response, cost = result
                        metadata = {"model": st.session_state.selected_model}
                    
                    if response:
                        # Add assistant message with full metadata
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.now(),
                            "metadata": {
                                "model": metadata.get("model", st.session_state.selected_model),
                                "cost": cost,
                                "routing_decision": metadata.get("routing_decision")
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