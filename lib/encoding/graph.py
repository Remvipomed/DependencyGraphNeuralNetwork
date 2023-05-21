import clingo
import networkx as nx
import numpy as np

from sklearn.preprocessing import OneHotEncoder


class DependencyGraph:

    def __init__(self):
        self.graph = nx.DiGraph()
        self.atom_set = set()
        self.atom_labels = dict()
        self.encoder = OneHotEncoder()

    def parse_dependencies(self, ctl):
        ctl.ground()

        for atom in ctl.symbolic_atoms:
            name = atom.symbol.name
            self.atom_set.add(name)
            self.graph.add_node(atom.literal, label=name)

            if atom.is_fact:
                continue

            # Handle rules
            rule = atom.symbol
            for symbol in rule.arguments:
                if symbol.type == clingo.SymbolType.Number:
                    self.graph.add_edge(atom.literal, symbol.number)

        atom_array = np.array(list(self.atom_set)).reshape(-1, 1)
        self.encoder.fit(atom_array)

    def encode_atoms(self):
        labels = [attributes["label"] for _, attributes in self.graph.nodes.data()]
        label_array = np.array(labels).reshape(-1, 1)
        features = self.encoder.transform(label_array)

        for i, node_id in enumerate(self.graph.nodes()):
            node = self.graph.nodes[node_id]
            node["feature"] = features[i]
