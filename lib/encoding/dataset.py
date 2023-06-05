
import clingo
import os
import networkx as nx
import numpy as np

from .. import project_dir
from . import graph
from spektral import data
from scipy import sparse


FILENAME_HANOI = "hanoi_sample_graph"


class HanoiDatasetSingle(data.Dataset):

    def __init__(self):
        self.graphs = list()
        super().__init__()

    def download(self):
        encoding = project_dir.joinpath("asp-planning-benchmarks/HanoiTower/encoding_single.asp")
        facts = project_dir.joinpath("asp-planning-benchmarks/HanoiTower/0004-hanoi_tower-60-0.asp")
        asp_files = [encoding, facts]

        ctl = clingo.Control()
        for asp_file in asp_files:
            ctl.load(str(asp_file))

        dependency_graph = graph.create_dependency_graph(ctl)
        nx_graph = dependency_graph.graph

        node_feature_list = np.array([nx_graph.nodes[node]["feature"] for node in nx_graph.nodes])
        node_features = np.array(node_feature_list)
        adjacency_matrix = nx.to_numpy_array(nx_graph)
        node_labels = np.array([nx_graph.nodes[node]["label"] for node in nx_graph.nodes])

        os.makedirs(self.path)
        filename = os.path.join(self.path, FILENAME_HANOI)
        np.savez(filename, x=node_features, a=adjacency_matrix, y=node_labels)

    def read(self):
        spek_data = np.load(os.path.join(self.path, FILENAME_HANOI + ".npz"))
        spek_graph = data.Graph(x=spek_data["x"], y=spek_data["y"], a=spek_data["a"])
        return [spek_graph]
