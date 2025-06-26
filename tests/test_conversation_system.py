#!/usr/bin/env python3
"""
Test suite for the conversation system
Demonstrates how conversation tracking and continuity work
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
    """Test the conversation system functionality"""
    
    def setUp(self):
        """Create a temporary directory for test storage"""
        self.temp_dir = tempfile.mkdtemp()
        self.conv_system = ConversationSystem(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_session_creation(self):
        """Test creating a new conversation session"""
        # Should auto-start a session
        self.assertIsNotNone(self.conv_system.current_session)
        
        # Start explicit session
        session_id = self.conv_system.start_session(title="Test Session")
        
        self.assertIsNotNone(session_id)
        self.assertEqual(self.conv_system.current_session.title, "Test Session")
        self.assertEqual(self.conv_system.current_session.state, "active")
    
    def test_recording_exchanges(self):
        """Test recording conversation exchanges"""
        # Record first exchange
        exchange_id = self.conv_system.record_exchange(
            query="How do I create a CTE in SQL?",
            response="To create a CTE, use WITH clause...",
            topics=["SQL", "CTE"],
            stored_in_memory=True,
            memory_id="mem_123"
        )
        
        self.assertIsNotNone(exchange_id)
        self.assertEqual(self.conv_system.current_session.total_exchanges, 1)
        self.assertEqual(self.conv_system.current_session.memories_created, 1)
        self.assertIn("SQL", self.conv_system.current_session.topics)
        self.assertIn("CTE", self.conv_system.current_session.topics)
    
    def test_session_continuity(self):
        """Test continuing conversations across sessions"""
        # Create first session about SQL
        session1_id = self.conv_system.start_session(title="SQL Discussion")
        
        self.conv_system.record_exchange(
            query="What is a window function?",
            response="Window functions perform calculations across rows...",
            topics=["SQL", "window functions"]
        )
        
        # End session
        self.conv_system.end_session()
        
        # Continue conversation about SQL
        parent_id, new_id = self.conv_system.continue_conversation("SQL")
        
        self.assertEqual(parent_id, session1_id)
        self.assertIsNotNone(new_id)
        self.assertEqual(self.conv_system.current_session.parent_session_id, session1_id)
    
    def test_topic_search(self):
        """Test finding sessions by topic"""
        # Create multiple sessions with different topics
        
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
        
        self.assertEqual(len(sql_sessions), 2)
        # Should be ordered newest first
        self.assertEqual(sql_sessions[0].title, "Advanced SQL")
        self.assertEqual(sql_sessions[1].title, "SQL Basics")
    
    def test_session_persistence(self):
        """Test saving and loading sessions"""
        # Create session with exchanges
        session_id = self.conv_system.start_session(title="Persistent Session")
        
        self.conv_system.record_exchange(
            query="Test query",
            response="Test response",
            topics=["test"],
            context={"screen": "test.py"}
        )
        
        # Force save
        self.conv_system._save_current_session()
        
        # Create new system instance to test loading
        new_system = ConversationSystem(storage_path=self.temp_dir)
        
        # Should have loaded the session
        self.assertIn(session_id, new_system.sessions)
        loaded_session = new_system.sessions[session_id]
        
        self.assertEqual(loaded_session.title, "Persistent Session")
        self.assertEqual(loaded_session.total_exchanges, 1)
        self.assertEqual(len(loaded_session.exchanges), 1)
        self.assertEqual(loaded_session.exchanges[0].context["screen"], "test.py")
    
    def test_pause_resume(self):
        """Test pausing and resuming sessions"""
        session_id = self.conv_system.start_session(title="Pausable Session")
        
        self.conv_system.record_exchange(
            query="First exchange",
            response="First response",
            topics=["test"]
        )
        
        # Pause session
        paused_id = self.conv_system.pause_session()
        self.assertEqual(paused_id, session_id)
        self.assertEqual(self.conv_system.sessions[session_id].state, "paused")
        
        # Start new session
        self.conv_system.start_session(title="New Session")
        
        # Resume paused session
        success = self.conv_system.resume_session(session_id)
        self.assertTrue(success)
        self.assertEqual(self.conv_system.current_session.session_id, session_id)
        self.assertEqual(self.conv_system.current_session.state, "active")
    
    def test_session_summary(self):
        """Test getting session summaries"""
        session_id = self.conv_system.start_session(title="Summary Test")
        
        # Add multiple exchanges
        for i in range(5):
            self.conv_system.record_exchange(
                query=f"Question {i}",
                response=f"Answer {i}",
                topics=["SQL", "testing"] if i % 2 == 0 else ["Python", "testing"],
                stored_in_memory=(i % 2 == 0)
            )
        
        self.conv_system.end_session()
        
        # Get summary
        summary = self.conv_system.get_session_summary(session_id)
        
        self.assertEqual(summary['title'], "Summary Test")
        self.assertEqual(summary['total_exchanges'], 5)
        self.assertEqual(summary['memories_created'], 3)  # 0, 2, 4
        self.assertIn("testing", summary['topics'])
        self.assertEqual(summary['top_topics'][0][0], "testing")  # Most common topic
        self.assertEqual(summary['memory_storage_rate'], "60.0%")
    
    def test_conversation_context(self):
        """Test getting recent conversation context"""
        # Add some exchanges
        for i in range(10):
            self.conv_system.record_exchange(
                query=f"Question {i}",
                response=f"Answer {i}",
                topics=[f"topic{i}"]
            )
        
        # Get recent context
        context = self.conv_system.get_conversation_context(lookback_exchanges=3)
        
        self.assertEqual(len(context), 3)
        # Should be most recent exchanges
        self.assertEqual(context[0]['query'], "Question 7")
        self.assertEqual(context[1]['query'], "Question 8")
        self.assertEqual(context[2]['query'], "Question 9")
    
    def test_analytics(self):
        """Test conversation analytics"""
        # Create sessions over multiple days
        base_time = datetime.now()
        
        # Manually create past sessions
        for day in range(5):
            session = ConversationSession(
                session_id=f"session_day_{day}",
                start_time=base_time - timedelta(days=day),
                end_time=base_time - timedelta(days=day, hours=-1),
                state="completed"
            )
            
            # Add exchanges
            for i in range(3):
                session.total_exchanges += 1
                session.topics.add("SQL")
                if i == 0:
                    session.memories_created += 1
            
            self.conv_system.sessions[session.session_id] = session
        
        # Get analytics
        analytics = self.conv_system.get_analytics(days=7)
        
        self.assertEqual(analytics['total_sessions'], 5)
        self.assertEqual(analytics['total_exchanges'], 15)  # 5 sessions * 3 exchanges
        self.assertEqual(analytics['total_memories'], 5)   # 5 sessions * 1 memory
        self.assertEqual(analytics['avg_exchanges_per_session'], 3)
        self.assertIn('SQL', [topic for topic, count in analytics['top_topics']])


class TestConversationScenarios(unittest.TestCase):
    """Test real-world conversation scenarios"""
    
    def setUp(self):
        """Create a temporary directory for test storage"""
        self.temp_dir = tempfile.mkdtemp()
        self.conv_system = ConversationSystem(storage_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)
    
    def test_debugging_session_flow(self):
        """Test a typical debugging conversation flow"""
        # Day 1: Initial problem
        session1 = self.conv_system.start_session(title="DBT Timeout Issue")
        
        self.conv_system.record_exchange(
            query="My dbt model is timing out after 30 minutes",
            response="Let's check the model complexity. Can you show me the SQL?",
            topics=["dbt", "timeout", "performance"]
        )
        
        self.conv_system.record_exchange(
            query="Here's my model with 5 CTEs and window functions",
            response="I see the issue. Try materializing intermediate CTEs...",
            topics=["dbt", "CTE", "optimization"],
            stored_in_memory=True,
            memory_id="mem_dbt_001"
        )
        
        session1_id = self.conv_system.end_session()
        
        # Day 2: Continue debugging
        parent_id, session2_id = self.conv_system.continue_conversation("dbt")
        
        self.assertEqual(parent_id, session1_id)
        
        self.conv_system.record_exchange(
            query="I tried materializing but still getting timeouts",
            response="Let's add indexes on the join columns...",
            topics=["dbt", "indexes", "performance"],
            stored_in_memory=True,
            memory_id="mem_dbt_002"
        )
        
        # Verify conversation chain
        self.assertEqual(self.conv_system.current_session.parent_session_id, session1_id)
        self.assertIn("dbt", self.conv_system.current_session.topics)
        
        # Verify we can find all dbt sessions
        dbt_sessions = self.conv_system.find_recent_sessions_by_topic("dbt")
        self.assertEqual(len(dbt_sessions), 2)


if __name__ == "__main__":
    unittest.main()