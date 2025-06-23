import sys
from pathlib import Path
from datetime import datetime
import types

# Provide a minimal stub for networkx so the module imports without the real dependency
sys.modules.setdefault('networkx', types.SimpleNamespace(DiGraph=lambda: None))

sys.path.append(str(Path(__file__).parent.parent))

from src.adam.memory_network import MemoryNode, MemoryNetworkSystem


def test_question_followed_by_solution_not_unresolved():
    # Build a list of memories where a question is followed by a resolving solution
    memories = [
        MemoryNode(
            memory_id="m0",
            conversation_id="c1",
            timestamp=datetime.now(),
            query="initial statement",
            response="response",
            topics=["topic"],
            memory_type="info",
        ),
        MemoryNode(
            memory_id="m1",
            conversation_id="c1",
            timestamp=datetime.now(),
            query="Why does this fail?",
            response="Not sure",
            topics=["topic"],
            memory_type="question",
        ),
        MemoryNode(
            memory_id="m2",
            conversation_id="c1",
            timestamp=datetime.now(),
            query="I tried a fix",
            response="Problem solved successfully",
            topics=["topic"],
            memory_type="solution",
        ),
    ]

    dummy_system = MemoryNetworkSystem.__new__(MemoryNetworkSystem)
    unresolved = MemoryNetworkSystem._find_unresolved_issues(dummy_system, memories)

    assert unresolved == []
