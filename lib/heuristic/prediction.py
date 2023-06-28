
import clingo
import time

from .. import project_root
from .training import GraphNeuralNetworkHeuristic


solution_model = dict()


def evaluate_graph_neural_network():
    program = project_root.joinpath("demo.asp")
    ctl = clingo.Control()
    ctl.load(str(program))

    heuristic = GraphNeuralNetworkHeuristic("demo")
    heuristic.load()
    heuristic.evaluate(ctl)


def set_model(model):
    for atom in model.context.symbolic_atoms:
        solution_model[atom.literal] = atom.symbol.positive


def set_optimal_heuristic(ctl_optimal):
    with ctl_optimal.backend() as backend:
        for literal, positive in solution_model.items():
            value = -1 if positive else 1
            backend.add_heuristic(literal, clingo.HeuristicType.Sign, value, 0, [])


def load_program() -> clingo.Control:
    program = project_root.joinpath("asp-planning-benchmarks/HanoiTower/encoding_single.asp")
    facts = project_root.joinpath("asp-planning-benchmarks/HanoiTower/0004-hanoi_tower-60-0.asp")
    ctl = clingo.Control()
    ctl.load(str(program))
    ctl.load(str(facts))
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
    ctl.solve(on_model=set_model)
    standard_end = time.time()
    print("solving duration:", standard_end - standard_start)


def run_optimal_solving():
    ctl_optimal = load_program()
    set_optimal_heuristic(ctl_optimal)

    print("optimal heuristic solving")
    optimal_start = time.time()
    ctl_optimal.solve()
    optimal_end = time.time()
    print("optimal duration:", optimal_end - optimal_start)


def run_performance_measurement():
    # run_heuristic()
    run_solver()
    run_optimal_solving()
