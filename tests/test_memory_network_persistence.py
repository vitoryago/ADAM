import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.adam.memory_network import MemoryNetworkSystem

class DummyBaseMemory:
    def __init__(self):
        self.counter = 0

    def remember_if_worthy(self, query: str, response: str, generation_cost=0.0, model_used="test"):
        self.counter += 1
        return f"mem_{self.counter}"

class DummyConversationSystem:
    class Session:
        def __init__(self, session_id: str):
            self.session_id = session_id

    def __init__(self):
        self.current_session = self.Session("session_1")

def test_memory_persists_across_reload(tmp_path, monkeypatch):
    """Newly added memories should be available after reloading the network."""
    monkeypatch.chdir(tmp_path)

    base = DummyBaseMemory()
    conv = DummyConversationSystem()

    net = MemoryNetworkSystem(base, conv)
    mem_id = net.add_memory_with_references(
        query="q1",
        response="r1",
        memory_type="test",
        topics=["topic"],
    )

    graph_file = tmp_path / "adam_memory_advanced" / "memory_network" / "memory_graph.gpickle"
    assert graph_file.exists()

    # Reload from disk and verify the memory is present
    net2 = MemoryNetworkSystem(base, conv)
    assert mem_id in net2.memory_graph.nodes
