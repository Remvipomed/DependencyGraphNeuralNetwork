
import clingo

from lib import project_dir
from lib.encoding import graph
# from lib.heuristic import training


def run_dependency():
    encoding = project_dir.joinpath("asp-planning-benchmarks/HanoiTower/encoding_single.asp")
    facts = project_dir.joinpath("asp-planning-benchmarks/HanoiTower/0004-hanoi_tower-60-0.asp")
    asp_files = [encoding, facts]

    ctl = clingo.Control()
    for asp_file in asp_files:
        ctl.load(str(asp_file))

    dependency_graph = graph.create_dependency_graph(ctl)
    print(dependency_graph.graph)


if __name__ == "__main__":
    run_dependency()
