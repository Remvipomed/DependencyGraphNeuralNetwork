
import clingo
import os
import networkx as nx
import numpy as np

from .. import project_root
from . import graph
from spektral import data
from scipy import sparse


FILENAME_HANOI = "hanoi_sample_graph"
FILENAME_DEMO = "demo_graph"


class DemoSet(data.Dataset):

    def __init__(self):
        self.graphs = list()
        super().__init__()

    def download(self):
        program = project_root.joinpath("demo.asp")
        ctl = clingo.Control()
        ctl.load(str(program))

        dependency_graph = graph.create_labeled_graph(ctl)
        spek_graph = graph.create_spektral_graph(dependency_graph)

        os.makedirs(self.path)
        filename = os.path.join(self.path, FILENAME_DEMO)
        np.savez(filename, x=spek_graph.x, a=spek_graph.a, y=spek_graph.y)

    def read(self):
        spek_data = np.load(os.path.join(self.path, FILENAME_DEMO + ".npz"))
        spek_graph = data.Graph(x=spek_data["x"], y=spek_data["y"], a=spek_data["a"])
        return [spek_graph]


class HanoiDatasetSingle(data.Dataset):

    def __init__(self):
        self.graphs = list()
        super().__init__()

    def download(self):
        encoding = project_root.joinpath("asp-planning-benchmarks/HanoiTower/encoding_single.asp")
        facts = project_root.joinpath("asp-planning-benchmarks/HanoiTower/0004-hanoi_tower-60-0.asp")
        asp_files = [encoding, facts]

        ctl = clingo.Control()
        for asp_file in asp_files:
            ctl.load(str(asp_file))

        dependency_graph = graph.create_labeled_graph(ctl)
        spek_graph = graph.create_spektral_graph(dependency_graph)

        os.makedirs(self.path)
        filename = os.path.join(self.path, FILENAME_HANOI)
        np.savez(filename, x=spek_graph.x, a=spek_graph.a, y=spek_graph.y)

    def read(self):
        spek_data = np.load(os.path.join(self.path, FILENAME_HANOI + ".npz"))
        spek_graph = data.Graph(x=spek_data["x"], y=spek_data["y"], a=spek_data["a"])
        return [spek_graph]
