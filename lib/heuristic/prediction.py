
import clingo

from .. import project_root
from .training import GraphNeuralNetworkHeuristic


def evaluate_graph_neural_network():
    program = project_root.joinpath("demo.asp")
    ctl = clingo.Control()
    ctl.load(str(program))

    heuristic = GraphNeuralNetworkHeuristic("demo")
    heuristic.load()
    heuristic.evaluate(ctl)


def run_heuristic():
    model = GraphNeuralNetworkHeuristic("demo")
    model.load()

