import sys
from pathlib import Path
import tempfile

sys.path.append(str(Path(__file__).parent.parent))

from src.adam.memory_network import MemoryNetworkSystem

class DummyBaseMemory:
    def __init__(self):
        self.counter = 0
    def remember_if_worthy(self, **kwargs):
        self.counter += 1
        return f"mem_{self.counter}"

class DummyConversationSystem:
    class Session:
        def __init__(self, session_id):
            self.session_id = session_id
    def __init__(self):
        self.current_session = self.Session("conv_1")


def test_evolution_summary_exists():
    tmpdir = tempfile.mkdtemp()
    base = DummyBaseMemory()
    conv = DummyConversationSystem()
    network = MemoryNetworkSystem(base, conv)
    network.network_path = Path(tmpdir)
    network.network_path.mkdir(exist_ok=True)

    network.add_memory_with_references(
        query="initial question",
        response="first response",
        memory_type="explanation",
        topics=["topic1"],
    )
    network.add_memory_with_references(
        query="follow up",
        response="second response",
        memory_type="explanation",
        topics=["topic1"],
    )

    summary = network.get_thread_summary("topic1")
    assert summary is not None
    assert summary["evolution_summary"] is not None
