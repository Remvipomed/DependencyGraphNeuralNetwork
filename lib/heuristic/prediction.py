
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


def set_optimal_heuristic(ctl_optimal, model):
    with ctl_optimal.backend() as backend:
        for atom in model.context.symbolic_atoms:
            value = 1 if atom.symbol.positive else -1
            backend.add_heuristic(atom.literal, clingo.HeuristicType.Init, value, 0, [])


def load_program() -> clingo.Control:
    program = project_root.joinpath("demo.asp")
    ctl = clingo.Control()
    ctl.load(str(program))
    ctl.ground()
    return ctl


def run_heuristic():
    print("heuristic loading")
    ctl = load_program()
    heuristic = GraphNeuralNetworkHeuristic("demo")
    heuristic.load()

    print("GNN heuristic solving")
    heuristic_start = time.time()
    heuristic.predict(ctl)
    ctl.solve()
    heuristic_end = time.time()
    print("GNN heuristic solving duration:", heuristic_end - heuristic_start)


def run_solver():
    ctl = load_program()

    print("standard solving")
    standard_start = time.time()
    ctl.solve()
    standard_end = time.time()
    print("solving duration:", standard_end - standard_start)


def run_optimal_solving():
    print("determine optimal heuristic")
    ctl = load_program()
    ctl_optimal = load_program()

    ctl.solve(on_model=lambda x: set_optimal_heuristic(ctl_optimal, x))

    optimal_start = time.time()
    ctl_optimal.solve()
    optimal_end = time.time()
    print("Optimal duration:", optimal_end - optimal_start)


def run_performance_measurement():
    run_heuristic()
    run_solver()
    run_optimal_solving()
