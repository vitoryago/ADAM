#!/usr/bin/env python3
"""
Test suite for the conversation system

Demonstrates how conversation tracking and continuity work.

TEST PHILOSOPHY:
===============
These tests serve dual purposes:
1. **Verification**: Ensure the system works correctly
2. **Documentation**: Show how to use the system

Each test demonstrates a real-world usage pattern, making
this file valuable for both testing and learning.

TEST COVERAGE:
- Basic session management (create, end, pause, resume)
- Exchange recording and retrieval
- Session continuity and threading
- Topic-based search
- Persistence and loading
- Analytics generation
- Real-world conversation flows

RUNNING TESTS:
```bash
python -m pytest tests/test_conversation_system.py -v
# or
python tests/test_conversation_system.py
```
"""

import unittest
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.adam.conversation_system import ConversationSystem, ConversationSession


class TestConversationSystem(unittest.TestCase):
    """Test the conversation system functionality
    
    This test class covers the core functionality of the conversation system.
    Each test method demonstrates a specific feature and how to use it.
    """
    
    def setUp(self):
        """Create a temporary directory for test storage
        
        WHY TEMP DIRECTORY:
        - Isolates each test run
        - Prevents test pollution
        - Automatically cleaned up
        - Allows parallel test execution
        
        The setUp method runs before each test, ensuring
        a clean slate for every test case.
        """
        self.temp_dir = tempfile.mkdtemp()
        self.conv_system = ConversationSystem(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory
        
        Runs after each test to clean up resources.
        This prevents disk space issues and ensures
        tests don't interfere with each other.
        """
        shutil.rmtree(self.temp_dir)
    
    def test_session_creation(self):
        """Test creating a new conversation session
        
        DEMONSTRATES:
        1. Auto-start behavior (user-friendly)
        2. Explicit session creation with title
        3. Session state management
        
        KEY INSIGHTS:
        - System always has an active session (no null states)
        - Sessions can be titled for easy identification
        - New sessions automatically end previous ones
        """
        # Should auto-start a session on initialization
        # This ensures users can immediately start talking
        self.assertIsNotNone(self.conv_system.current_session)
        
        # Start explicit session with descriptive title
        session_id = self.conv_system.start_session(title="Test Session")
        
        # Verify session was created properly
        self.assertIsNotNone(session_id)
        self.assertEqual(self.conv_system.current_session.title, "Test Session")
        self.assertEqual(self.conv_system.current_session.state, "active")
    
    def test_recording_exchanges(self):
        """Test recording conversation exchanges
        
        DEMONSTRATES:
        1. How to record a user-ADAM interaction
        2. Tracking memory storage decisions
        3. Topic extraction and indexing
        4. Session statistics updates
        
        REAL-WORLD USAGE:
        This is called after every interaction to maintain
        a complete conversation history.
        """
        # Record first exchange - a valuable SQL learning moment
        exchange_id = self.conv_system.record_exchange(
            query="How do I create a CTE in SQL?",
            response="To create a CTE, use WITH clause...",
            topics=["SQL", "CTE"],
            stored_in_memory=True,  # This was deemed memory-worthy
            memory_id="mem_123"     # ID from memory system
        )
        
        # Verify exchange was recorded
        self.assertIsNotNone(exchange_id)
        
        # Check session statistics updated correctly
        self.assertEqual(self.conv_system.current_session.total_exchanges, 1)
        self.assertEqual(self.conv_system.current_session.memories_created, 1)
        
        # Verify topics were indexed for search
        self.assertIn("SQL", self.conv_system.current_session.topics)
        self.assertIn("CTE", self.conv_system.current_session.topics)
    
    def test_session_continuity(self):
        """Test continuing conversations across sessions
        
        DEMONSTRATES:
        1. Creating a conversation thread
        2. Linking related sessions
        3. Topic-based continuation
        
        USE CASE:
        User: "Let's continue our SQL discussion from yesterday"
        System finds the previous SQL session and creates a new
        linked session, maintaining context across time.
        
        WHY NOT JUST RESUME:
        - Each session represents a distinct time period
        - Maintains clear conversation boundaries
        - Enables better analytics and search
        """
        # Day 1: Create first session about SQL
        session1_id = self.conv_system.start_session(title="SQL Discussion")
        
        self.conv_system.record_exchange(
            query="What is a window function?",
            response="Window functions perform calculations across rows...",
            topics=["SQL", "window functions"]
        )
        
        # End of day 1
        self.conv_system.end_session()
        
        # Day 2: User wants to continue SQL discussion
        parent_id, new_id = self.conv_system.continue_conversation("SQL")
        
        # Verify proper linkage
        self.assertEqual(parent_id, session1_id)  # Found previous session
        self.assertIsNotNone(new_id)              # Created new session
        self.assertEqual(self.conv_system.current_session.parent_session_id, session1_id)
    
    def test_topic_search(self):
        """Test finding sessions by topic
        
        DEMONSTRATES:
        1. Topic-based session retrieval
        2. Chronological ordering (newest first)
        3. Multi-session topic tracking
        
        USE CASE:
        "What did we discuss about SQL last week?"
        System can quickly find all SQL-related conversations.
        
        This enables contextual conversation continuation and
        helps users find previous discussions.
        """
        # Create multiple sessions with different topics
        # This simulates a realistic conversation history
        
        # Session 1: SQL
        self.conv_system.start_session(title="SQL Basics")
        self.conv_system.record_exchange(
            query="Explain SELECT",
            response="SELECT retrieves data...",
            topics=["SQL", "SELECT"]
        )
        self.conv_system.end_session()
        
        # Session 2: Python
        self.conv_system.start_session(title="Python Tutorial")
        self.conv_system.record_exchange(
            query="How to use list comprehensions?",
            response="List comprehensions provide concise way...",
            topics=["Python", "comprehensions"]
        )
        self.conv_system.end_session()
        
        # Session 3: More SQL
        self.conv_system.start_session(title="Advanced SQL")
        self.conv_system.record_exchange(
            query="Explain JOINs",
            response="JOINs combine rows from tables...",
            topics=["SQL", "JOIN"]
        )
        self.conv_system.end_session()
        
        # Search for SQL sessions
        sql_sessions = self.conv_system.find_recent_sessions_by_topic("SQL")
        
        # Should find both SQL sessions
        self.assertEqual(len(sql_sessions), 2)
        
        # Verify chronological ordering (newest first)
        # This ordering helps users find recent discussions
        self.assertEqual(sql_sessions[0].title, "Advanced SQL")
        self.assertEqual(sql_sessions[1].title, "SQL Basics")
    
    def test_session_persistence(self):
        """Test saving and loading sessions
        
        DEMONSTRATES:
        1. Automatic session persistence
        2. Loading sessions on system restart
        3. Complete data preservation
        
        CRITICAL FOR:
        - System restarts don't lose conversations
        - Crash recovery
        - Long-term conversation history
        
        This test simulates ADAM restarting and verifying
        all conversation data is preserved.
        """
        # Create session with meaningful exchange
        session_id = self.conv_system.start_session(title="Persistent Session")
        
        self.conv_system.record_exchange(
            query="Test query",
            response="Test response",
            topics=["test"],
            context={"screen": "test.py"}  # Environmental context preserved
        )
        
        # Force save (normally happens automatically)
        self.conv_system._save_current_session()
        
        # Simulate system restart by creating new instance
        new_system = ConversationSystem(storage_path=self.temp_dir)
        
        # Verify session was loaded from disk
        self.assertIn(session_id, new_system.sessions)
        loaded_session = new_system.sessions[session_id]
        
        # Check all data preserved correctly
        self.assertEqual(loaded_session.title, "Persistent Session")
        self.assertEqual(loaded_session.total_exchanges, 1)
        self.assertEqual(len(loaded_session.exchanges), 1)
        
        # Even context data is preserved
        self.assertEqual(loaded_session.exchanges[0].context["screen"], "test.py")
    
    def test_pause_resume(self):
        """Test pausing and resuming sessions
        
        DEMONSTRATES:
        1. Pausing active conversations
        2. Working on something else
        3. Resuming exactly where left off
        
        USE CASE:
        User working on multiple problems needs to context-switch
        without losing their place in any conversation.
        
        Example scenario:
        - Debugging Python code (pause)
        - Handle urgent SQL issue (new session)
        - Return to Python debugging (resume)
        """
        # Start working on first problem
        session_id = self.conv_system.start_session(title="Pausable Session")
        
        self.conv_system.record_exchange(
            query="First exchange",
            response="First response",
            topics=["test"]
        )
        
        # Need to context switch - pause current work
        paused_id = self.conv_system.pause_session()
        self.assertEqual(paused_id, session_id)
        self.assertEqual(self.conv_system.sessions[session_id].state, "paused")
        
        # Work on urgent issue in new session
        self.conv_system.start_session(title="New Session")
        
        # Return to original work - resume paused session
        success = self.conv_system.resume_session(session_id)
        self.assertTrue(success)
        self.assertEqual(self.conv_system.current_session.session_id, session_id)
        self.assertEqual(self.conv_system.current_session.state, "active")
    
    def test_session_summary(self):
        """Test getting session summaries
        
        DEMONSTRATES:
        1. Comprehensive session analysis
        2. Topic frequency tracking
        3. Memory storage metrics
        
        USE CASE:
        Generate conversation reports showing:
        - What was discussed
        - How valuable the conversation was
        - Topic distribution
        
        This helps users understand their interaction
        patterns and ADAM's value delivery.
        """
        session_id = self.conv_system.start_session(title="Summary Test")
        
        # Simulate realistic conversation with mixed topics
        for i in range(5):
            self.conv_system.record_exchange(
                query=f"Question {i}",
                response=f"Answer {i}",
                # Alternate between SQL and Python topics
                topics=["SQL", "testing"] if i % 2 == 0 else ["Python", "testing"],
                # Every other exchange is memory-worthy
                stored_in_memory=(i % 2 == 0)
            )
        
        self.conv_system.end_session()
        
        # Get comprehensive summary
        summary = self.conv_system.get_session_summary(session_id)
        
        # Verify summary completeness
        self.assertEqual(summary['title'], "Summary Test")
        self.assertEqual(summary['total_exchanges'], 5)
        self.assertEqual(summary['memories_created'], 3)  # Exchanges 0, 2, 4
        
        # Check topic analysis
        self.assertIn("testing", summary['topics'])
        self.assertEqual(summary['top_topics'][0][0], "testing")  # Most common
        
        # Memory storage rate indicates conversation value
        self.assertEqual(summary['memory_storage_rate'], "60.0%")
    
    def test_conversation_context(self):
        """Test getting recent conversation context
        
        DEMONSTRATES:
        1. Retrieving recent exchange history
        2. Context window management
        3. Chronological ordering
        
        USE CASE:
        When generating responses, ADAM needs recent
        conversation context to maintain coherence.
        
        This context is included in prompts:
        "Based on our recent discussion about X..."
        """
        # Simulate extended conversation
        for i in range(10):
            self.conv_system.record_exchange(
                query=f"Question {i}",
                response=f"Answer {i}",
                topics=[f"topic{i}"]
            )
        
        # Get recent context for response generation
        context = self.conv_system.get_conversation_context(lookback_exchanges=3)
        
        # Should return exactly 3 most recent exchanges
        self.assertEqual(len(context), 3)
        
        # Verify chronological order (oldest to newest)
        self.assertEqual(context[0]['query'], "Question 7")
        self.assertEqual(context[1]['query'], "Question 8")
        self.assertEqual(context[2]['query'], "Question 9")
    
    def test_analytics(self):
        """Test conversation analytics
        
        DEMONSTRATES:
        1. Gathering usage metrics
        2. Identifying patterns
        3. Measuring engagement
        
        ANALYTICS PROVIDE INSIGHTS INTO:
        - User engagement (session frequency, duration)
        - Content value (memory storage rate)
        - Topic interests (what users discuss)
        - Usage patterns (when users are active)
        
        This data helps improve ADAM's effectiveness
        and understand user needs.
        """
        # Create realistic session history
        base_time = datetime.now()
        
        # Simulate 5 days of conversations
        for day in range(5):
            session = ConversationSession(
                session_id=f"session_day_{day}",
                start_time=base_time - timedelta(days=day),
                end_time=base_time - timedelta(days=day, hours=-1),  # 1 hour sessions
                state="completed"
            )
            
            # Each session has 3 exchanges about SQL
            for i in range(3):
                session.total_exchanges += 1
                session.topics.add("SQL")
                if i == 0:  # First exchange of each session is memorable
                    session.memories_created += 1
            
            self.conv_system.sessions[session.session_id] = session
        
        # Generate analytics for the week
        analytics = self.conv_system.get_analytics(days=7)
        
        # Verify metrics accuracy
        self.assertEqual(analytics['total_sessions'], 5)
        self.assertEqual(analytics['total_exchanges'], 15)  # 5 sessions * 3 exchanges
        self.assertEqual(analytics['total_memories'], 5)   # 5 sessions * 1 memory
        self.assertEqual(analytics['avg_exchanges_per_session'], 3)
        
        # SQL should be the top topic
        self.assertIn('SQL', [topic for topic, count in analytics['top_topics']])


class TestConversationScenarios(unittest.TestCase):
    """Test real-world conversation scenarios
    
    These tests demonstrate complete conversation flows that
    mirror actual user interactions with ADAM.
    
    SCENARIOS COVERED:
    - Multi-day debugging sessions
    - Problem resolution workflows
    - Knowledge building conversations
    
    These tests serve as both validation and documentation
    of complex conversation patterns.
    """
    
    def setUp(self):
        """Create a temporary directory for test storage"""
        self.temp_dir = tempfile.mkdtemp()
        self.conv_system = ConversationSystem(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory
        
        Runs after each test to clean up resources.
        This prevents disk space issues and ensures
        tests don't interfere with each other.
        """
        shutil.rmtree(self.temp_dir)
    
    def test_debugging_session_flow(self):
        """Test a typical debugging conversation flow
        
        REAL-WORLD SCENARIO:
        A user debugging a complex dbt performance issue over
        multiple days. This demonstrates:
        
        1. **Day 1**: Problem identification
           - User reports timeout issue
           - ADAM analyzes and suggests initial fix
           - Solution stored in memory for future reference
           
        2. **Day 2**: Continued troubleshooting
           - User reports partial success
           - ADAM provides additional optimization
           - Conversation linked to maintain context
        
        This pattern is common in software development where
        complex issues require iterative problem-solving.
        """
        # Day 1: User encounters performance issue
        session1 = self.conv_system.start_session(title="DBT Timeout Issue")
        
        # Initial problem report
        self.conv_system.record_exchange(
            query="My dbt model is timing out after 30 minutes",
            response="Let's check the model complexity. Can you show me the SQL?",
            topics=["dbt", "timeout", "performance"]
        )
        
        # User provides details, ADAM gives solution
        self.conv_system.record_exchange(
            query="Here's my model with 5 CTEs and window functions",
            response="I see the issue. Try materializing intermediate CTEs...",
            topics=["dbt", "CTE", "optimization"],
            stored_in_memory=True,  # Valuable solution worth storing
            memory_id="mem_dbt_001"
        )
        
        # End of day 1
        session1_id = self.conv_system.end_session()
        
        # Day 2: User returns with update
        parent_id, session2_id = self.conv_system.continue_conversation("dbt")
        
        # Verify proper conversation threading
        self.assertEqual(parent_id, session1_id)
        
        # Continue troubleshooting
        self.conv_system.record_exchange(
            query="I tried materializing but still getting timeouts",
            response="Let's add indexes on the join columns...",
            topics=["dbt", "indexes", "performance"],
            stored_in_memory=True,  # Another valuable solution
            memory_id="mem_dbt_002"
        )
        
        # Verify conversation continuity
        self.assertEqual(self.conv_system.current_session.parent_session_id, session1_id)
        self.assertIn("dbt", self.conv_system.current_session.topics)
        
        # Verify both sessions are findable by topic
        dbt_sessions = self.conv_system.find_recent_sessions_by_topic("dbt")
        self.assertEqual(len(dbt_sessions), 2)


if __name__ == "__main__":
    unittest.main()