import clingo
import networkx as nx
import numpy as np

from sklearn.preprocessing import OneHotEncoder


class DependencyObserver(clingo.Observer):

    def __init__(self):
        super().__init__()
        self.graph = nx.DiGraph()

    def rule(self, choice, head, body):
        # print("rule:", choice, head, body)
        for head_literal in head:
            for body_literal in body:
                self.graph.add_edge(body_literal, head_literal)


class DependencyGraph:

    def __init__(self, ctl):
        self.ctl = ctl
        self.graph = nx.DiGraph()
        self.atom_set = set()
        self.atom_labels = dict()
        self.encoder = OneHotEncoder()

        self.observer = DependencyObserver()
        self.ctl.register_observer(self.observer)

    def parse_dependencies(self):
        self.ctl.ground()
        self.graph = self.observer.graph

        for atom in self.ctl.symbolic_atoms:
            name = atom.symbol.name
            self.atom_set.add(name)
            self.graph.add_node(atom.literal, label=name)

        atom_array = np.array(list(self.atom_set)).reshape(-1, 1)
        self.encoder.fit(atom_array)

    def encode_atoms(self):
        labels = [attributes["label"] for _, attributes in self.graph.nodes.data()]
        label_array = np.array(labels).reshape(-1, 1)
        features = self.encoder.transform(label_array)

        for i, node_id in enumerate(self.graph.nodes()):
            node = self.graph.nodes[node_id]
            node["feature"] = features[i]

    def search_atom(self, literal):
        for atom in self.ctl.symbolic_atoms:
            if atom.literal == literal:
                return atom


dependency_graph = DependencyGraph(clingo.Control())
