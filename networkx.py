import pickle

class DiGraph:
    def __init__(self):
        self.nodes = {}
        self._adj = {}

    def add_node(self, node, **attr):
        self.nodes[node] = attr

    def has_node(self, node):
        return node in self.nodes

    def add_edge(self, u, v, **attr):
        self._adj.setdefault(u, {})[v] = attr

    def edges(self):
        return [(u, v) for u, nbrs in self._adj.items() for v in nbrs]

    def subgraph(self, nodes):
        g = DiGraph()
        for n in nodes:
            if n in self.nodes:
                g.add_node(n, **self.nodes[n])
        for u, nbrs in self._adj.items():
            if u in nodes:
                for v, attr in nbrs.items():
                    if v in nodes:
                        g.add_edge(u, v, **attr)
        return g


def write_gpickle(G, path):
    with open(path, 'wb') as f:
        pickle.dump(G, f)


def read_gpickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def spring_layout(graph, k=2, iterations=50):
    # Return dummy positions
    return {n: (0, 0) for n in graph.nodes}


def draw_networkx_nodes(*args, **kwargs):
    pass


def draw_networkx_edges(*args, **kwargs):
    pass


def draw_networkx_labels(*args, **kwargs):
    pass
