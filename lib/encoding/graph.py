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

    def __init__(self):
        self.graph = nx.DiGraph()
        self.atom_set = {None}
        self.atom_labels = dict()
        self.encoder = OneHotEncoder()

        self.observer = DependencyObserver()

    def parse_dependencies(self, ctl: clingo.Control):
        ctl.register_observer(self.observer)
        ctl.ground()

        self.graph = self.observer.graph
        nx.set_node_attributes(self.graph, None, "type")

        rule_nodes = list()
        fact_nodes = list()

        for atom in ctl.symbolic_atoms:
            name = atom.symbol.name
            self.atom_set.add(name)
            if atom.literal in self.graph.nodes:
                rule_nodes.append(name)
            else:
                fact_nodes.append(name)
            self.graph.add_node(atom.literal)
            self.graph.nodes[atom.literal]["type"] = name

        ctl.solve(on_model=self.determine_labels)

    def determine_labels(self, model):
        for literal in self.graph.nodes:
            self.graph.nodes[literal]["label"] = model.is_true(literal)

    def encode_atoms(self):
        atom_array = np.array(list(self.atom_set)).reshape(-1, 1)
        self.encoder.fit(atom_array)

        labels = [attributes["type"] for _, attributes in self.graph.nodes.data()]
        label_array = np.array(labels).reshape(-1, 1)
        features = self.encoder.transform(label_array)

        for i, node_id in enumerate(self.graph.nodes()):
            node = self.graph.nodes[node_id]
            node["feature"] = features[i]


def create_dependency_graph(ctl: clingo.Control) -> DependencyGraph:
    dependency_graph = DependencyGraph()
    dependency_graph.parse_dependencies(ctl)
    dependency_graph.encode_atoms()
    return dependency_graph
