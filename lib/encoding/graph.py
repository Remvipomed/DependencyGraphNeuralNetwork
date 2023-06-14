import clingo
import networkx as nx
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from spektral import data


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
        nx.set_node_attributes(self.graph, False, "is_fact")

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
            self.graph.nodes[atom.literal]["is_fact"] = atom.is_fact

    def solve_for_labels(self, ctl: clingo.Control):
        ctl.solve(on_model=self.determine_labels)

    def determine_labels(self, model):
        for literal in self.graph.nodes:
            label = 1 if model.is_true(literal) else -1
            self.graph.nodes[literal]["label"] = label

    def encode_atoms(self):
        atom_array = np.array(list(self.atom_set)).reshape(-1, 1)
        self.encoder.fit(atom_array)

        atom_names = [attributes["type"] for _, attributes in self.graph.nodes.data()]
        label_array = np.array(atom_names).reshape(-1, 1)
        name_encoding = self.encoder.transform(label_array).toarray()
        fact_array = np.array([int(attributes["is_fact"]) for _, attributes in self.graph.nodes.data()])
        features = np.column_stack([fact_array, name_encoding])

        for i, node_id in enumerate(self.graph.nodes()):
            node = self.graph.nodes[node_id]
            node["feature"] = features[i]


def create_labeled_graph(ctl: clingo.Control) -> DependencyGraph:
    dependency_graph = DependencyGraph()
    dependency_graph.parse_dependencies(ctl)
    dependency_graph.encode_atoms()
    dependency_graph.solve_for_labels(ctl)
    return dependency_graph


def create_encoded_graph(ctl: clingo.Control) -> DependencyGraph:
    dependency_graph = DependencyGraph()
    dependency_graph.parse_dependencies(ctl)
    dependency_graph.encode_atoms()
    return dependency_graph


def create_spektral_graph(dependency_graph: DependencyGraph):
    nx_graph = dependency_graph.graph

    node_feature_list = np.array([nx_graph.nodes[node]["feature"] for node in nx_graph.nodes])
    node_features = np.array(node_feature_list)
    adjacency_matrix = nx.to_numpy_array(nx_graph)
    node_labels = np.array([nx_graph.nodes[node]["label"] for node in nx_graph.nodes])
    spek_graph = data.Graph(x=node_features, a=adjacency_matrix, y=node_labels)
    return spek_graph
