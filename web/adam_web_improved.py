#!/usr/bin/env python3
"""
ADAM Web Interface - Improved version with error boundaries and better session management
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import ADAM components
from src.adam.memory import ADAMMemoryAdvanced
from src.adam.conversation_system import ConversationSystem
from src.adam.llm.client import UnifiedLLMClient
from src.adam.llm.config import LLMConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="ADAM - Advanced Data Analytics Model",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved styling
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
    
    /* Code blocks */
    .stCodeBlock {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Session persistence indicator */
    .persistence-indicator {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: #4caf50;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.8em;
        z-index: 1000;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .persistence-indicator.show {
        opacity: 1;
    }
    
    /* Loading skeleton */
    .skeleton {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200% 100%;
        animation: loading 1.5s infinite;
    }
    
    @keyframes loading {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
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
            if st.checkbox("Show error details", key=f"error_{func.__name__}"):
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
        
        # Update session
        sessions[session_id] = {
            "messages": state.get("messages", []),
            "total_cost": state.get("total_cost", 0.0),
            "selected_model": state.get("selected_model", None),
            "use_memory": state.get("use_memory", True),
            "last_updated": datetime.now().isoformat()
        }
        
        # Save back
        with open(cls.SESSIONS_FILE, 'w') as f:
            json.dump(sessions, f, indent=2)
        
        return True
    
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
    """Improved web interface for ADAM with error boundaries and session persistence"""
    
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
                    
                    # Get available models
                    st.session_state.available_models = st.session_state.llm_config.get_available_models()
                    
                    # Initialize session tracking
                    st.session_state.current_session_id = None
                    st.session_state.messages = []
                    st.session_state.total_cost = 0.0
                    st.session_state.initialized = True
                    st.session_state.error_count = 0
                    st.session_state.use_memory = True
                    
                    # Success notification
                    st.success("ADAM initialized successfully!")
                    
                except Exception as e:
                    st.error(f"Failed to initialize ADAM: {str(e)}")
                    st.stop()
    
    @error_boundary
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
            
            # Session list with error handling
            try:
                self.render_session_list()
            except Exception as e:
                st.error(f"Error loading sessions: {str(e)}")
            
            st.divider()
            
            # Settings section
            self.render_settings()
            
            # Memory stats
            self.render_memory_stats()
            
            # Cost tracking
            self.render_cost_tracking()
    
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
    
    def render_session_list(self):
        """Render session list with error handling"""
        st.subheader("Conversations")
        
        sessions = list(st.session_state.conversation.sessions.values())
        
        if not sessions:
            st.info("No conversations yet. Start a new one!")
            return
        
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
                self.render_session_button(session)
    
    def render_session_button(self, session):
        """Render a single session button"""
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
    
    def render_settings(self):
        """Render settings section"""
        st.subheader("Settings")
        
        # Model selection with error handling
        try:
            selected_model = st.selectbox(
                "Model",
                options=st.session_state.available_models,
                index=0 if st.session_state.available_models else None
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
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
        
        # Conversation settings
        st.divider()
        st.subheader("Conversation")
        
        # Memory toggle with persistence
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
    
    @error_boundary
    def render_memory_stats(self):
        """Render memory statistics"""
        st.divider()
        st.subheader("Memory System")
        
        try:
            stats = st.session_state.memory.get_memory_analytics()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Memories", stats.get('total_memories', 0))
            with col2:
                st.metric("Hit Rate", f"{stats.get('memory_hit_rate', 0):.1%}")
        except Exception as e:
            st.error(f"Error loading memory stats: {str(e)}")
    
    def render_cost_tracking(self):
        """Render cost tracking"""
        if st.session_state.total_cost > 0:
            st.divider()
            st.metric("Session Cost", f"${st.session_state.total_cost:.4f}")
    
    @error_boundary
    def start_new_session(self):
        """Start a new conversation session"""
        # End current session if exists
        if st.session_state.current_session_id:
            self.save_current_session()
            st.session_state.conversation.end_session()
        
        # Start new session
        session_id = st.session_state.conversation.start_session("New Conversation")
        st.session_state.current_session_id = session_id
        st.session_state.messages = []
        st.session_state.total_cost = 0.0
        st.rerun()
    
    @error_boundary
    def load_session(self, session_id: str):
        """Load an existing session"""
        # Save current session first
        if st.session_state.current_session_id:
            self.save_current_session()
        
        # Resume the session
        st.session_state.conversation.resume_session(session_id)
        st.session_state.current_session_id = session_id
        
        # Try to load persisted state first
        persisted_state = SessionPersistence.load_session_state(session_id)
        if persisted_state:
            st.session_state.messages = persisted_state.get("messages", [])
            st.session_state.total_cost = persisted_state.get("total_cost", 0.0)
            st.session_state.selected_model = persisted_state.get("selected_model", st.session_state.available_models[0])
            st.session_state.use_memory = persisted_state.get("use_memory", True)
        else:
            # Load from conversation system
            session = st.session_state.conversation.sessions.get(session_id)
            if session:
                st.session_state.messages = []
                for exchange in session.exchanges:
                    st.session_state.messages.append({
                        "role": "user",
                        "content": exchange["query"]
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": exchange["response"],
                        "model": exchange.get("model_used", "Unknown"),
                        "cost": exchange.get("generation_cost", 0)
                    })
                
                # Calculate total cost
                st.session_state.total_cost = sum(
                    exchange.get("generation_cost", 0) 
                    for exchange in session.exchanges
                )
        
        st.rerun()
    
    @error_boundary
    def save_current_session(self):
        """Save current session state to disk"""
        if not st.session_state.get('auto_save', True):
            return
        
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
            
            # Show save indicator (would need JavaScript for animation)
            st.toast("Session saved", icon="💾")
    
    @error_boundary
    def render_chat_interface(self):
        """Render the main chat interface"""
        st.title("ADAM Chat")
        
        # Display current session info
        if st.session_state.current_session_id:
            session = st.session_state.conversation.sessions.get(st.session_state.current_session_id)
            if session:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.caption(f"Session: {session.title or 'Untitled'}")
                with col2:
                    st.caption(f"Messages: {len(st.session_state.messages) // 2}")
                with col3:
                    if st.button("💾 Save"):
                        self.save_current_session()
        
        # Chat messages container
        chat_container = st.container()
        
        # Display messages
        with chat_container:
            for message in st.session_state.messages:
                self.render_message(message)
        
        # Chat input
        if prompt := st.chat_input("Ask ADAM anything..."):
            self.handle_user_input(prompt)
    
    def render_message(self, message):
        """Render a single message with error handling"""
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show metadata for assistant messages
            if message["role"] == "assistant":
                col1, col2 = st.columns([6, 1])
                with col1:
                    if "model" in message:
                        st.caption(f"Model: {message['model']}")
                with col2:
                    if "cost" in message:
                        st.caption(f"Cost: ${message.get('cost', 0):.4f}")
    
    @error_boundary
    async def handle_user_input(self, prompt: str):
        """Handle user input with error handling"""
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get memory context if enabled
                    memory_context = ""
                    if st.session_state.use_memory:
                        memories = st.session_state.memory.recall_with_context(
                            query=prompt,
                            n_results=5,
                            conversation_context=st.session_state.messages[-10:]
                        )
                        if memories:
                            memory_context = "\n\nRelevant memories:\n" + "\n".join(
                                f"- {mem['query']}: {mem['response'][:100]}..."
                                for mem in memories
                            )
                    
                    # Build context
                    context = self.build_context() + memory_context
                    
                    # Get response
                    response = await st.session_state.llm_client.complete(
                        prompt=prompt,
                        model=st.session_state.selected_model,
                        context=context,
                        stream=True
                    )
                    
                    # Stream response
                    response_placeholder = st.empty()
                    full_response = ""
                    
                    async for chunk in response:
                        full_response += chunk.content
                        response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)
                    
                    # Calculate cost (simplified)
                    cost = len(prompt) * 0.00001 + len(full_response) * 0.00003
                    st.session_state.total_cost += cost
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "model": st.session_state.selected_model,
                        "cost": cost
                    })
                    
                    # Store in memory if worthy
                    if st.session_state.use_memory:
                        st.session_state.memory.remember_if_worthy(
                            query=prompt,
                            response=full_response,
                            context={"session_id": st.session_state.current_session_id},
                            generation_cost=cost,
                            model_used=st.session_state.selected_model
                        )
                    
                    # Update conversation system
                    if st.session_state.current_session_id:
                        st.session_state.conversation.add_exchange(
                            query=prompt,
                            response=full_response,
                            model_used=st.session_state.selected_model,
                            generation_cost=cost,
                            memory_ids=[]
                        )
                    
                    # Auto-save if enabled
                    if st.session_state.get('auto_save', True):
                        self.save_current_session()
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.session_state.error_count = st.session_state.get('error_count', 0) + 1
                    
                    # Add error message to chat
                    error_response = f"I encountered an error: {str(e)}. Please try again."
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_response,
                        "model": "error",
                        "cost": 0
                    })
    
    def build_context(self) -> str:
        """Build context from recent messages"""
        context_messages = st.session_state.messages[-10:]  # Last 10 messages
        context = ""
        
        for msg in context_messages:
            if msg["role"] == "user":
                context += f"\nUser: {msg['content']}"
            else:
                context += f"\nAssistant: {msg['content'][:200]}..."
        
        return context
    
    def run(self):
        """Run the web interface"""
        # Render sidebar
        self.render_sidebar()
        
        # Main content
        if not st.session_state.current_session_id:
            # Welcome screen
            st.markdown("""
            # Welcome to ADAM! 🧠
            
            ADAM is your Advanced Data Analytics Model with perfect memory.
            
            ### Features:
            - 🧠 **Perfect Memory**: I remember all our conversations
            - 🤖 **Multiple AI Models**: Choose the best model for your needs
            - 💾 **Session Persistence**: Your conversations are automatically saved
            - 📊 **Analytics**: Track usage, costs, and memory performance
            
            ### Getting Started:
            Click **"➕ New Conversation"** in the sidebar to begin!
            """)
        else:
            # Chat interface
            self.render_chat_interface()


def main():
    """Main entry point"""
    interface = ADAMWebInterface()
    interface.run()


if __name__ == "__main__":
    # Run async event loop for the interface
    main()