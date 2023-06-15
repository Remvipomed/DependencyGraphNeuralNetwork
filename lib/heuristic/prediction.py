
import clingo
import time

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
    print("heuristic loading")
    program = project_root.joinpath("demo.asp")
    ctl = clingo.Control()
    ctl.load(str(program))

    heuristic = GraphNeuralNetworkHeuristic("demo")
    heuristic.load()
    heuristic.predict(ctl)

    print("heuristic solving")
    ctl.solve()


def run_solver():
    print("standard grounding")
    program = project_root.joinpath("demo.asp")
    ctl = clingo.Control()
    ctl.load(str(program))
    ctl.ground()

    print("standard solving")
    ctl.solve()


def measure_solving_times():
    heuristic_start = time.time()
    run_heuristic()
    heuristic_end = time.time()
    print("Heuristic duration:", heuristic_end - heuristic_start)

    standard_start = time.time()
    run_solver()
    standard_end = time.time()
    print("Default duration:", standard_end - standard_start)



