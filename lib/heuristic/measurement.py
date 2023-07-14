
import clingo
import time

from .. import project_root
from .training import GraphNeuralNetworkHeuristic

MODEL_NAME = "demo"
print("GNN heuristic model loading")
heuristic = GraphNeuralNetworkHeuristic(MODEL_NAME)
heuristic.load()

solution_model = dict()


def set_solution_model(model):
    solution_model.clear()
    for symbol in model.symbols(atoms=True):
        solution_model[str(symbol)] = True
    for symbol in model.symbols(complement=True):
        solution_model[str(symbol)] = False


def set_optimal_heuristic(ctl_optimal):
    with ctl_optimal.backend() as backend:
        for i, atom in enumerate(ctl_optimal.symbolic_atoms):
            if str(atom.symbol) not in solution_model:
                continue

            value = 1 if solution_model[str(atom.symbol)] else -1
            backend.add_heuristic(atom.literal, clingo.HeuristicType.Sign, value, 0, [])


def load_program(program_dict: dict) -> clingo.Control:
    ctl = clingo.Control()
    ctl.load(str(program_dict["facts"]))
    ctl.load(str(program_dict["rules"]))
    ctl.ground()
    return ctl


def run_heuristic_solving(ctl: clingo.Control):
    print("GNN heuristic solving")
    heuristic_start = time.time()
    heuristic.predict(ctl)
    ctl.solve()
    heuristic_end = time.time()
    heuristic_duration = heuristic_end - heuristic_start
    print("GNN heuristic solving duration:", heuristic_duration)
    return {"duration": heuristic_duration, **ctl.statistics}


def run_standard_solving(ctl: clingo.Control):
    print("standard solving")
    standard_start = time.time()
    ctl.solve(on_model=set_solution_model)
    standard_end = time.time()
    standard_duration = standard_end - standard_start
    print("solving duration:", standard_duration)
    return {"duration": standard_duration, **ctl.statistics}


def run_optimal_solving(ctl: clingo.Control):
    ctl.configuration.solver.heuristic = "Domain"
    set_optimal_heuristic(ctl)

    print("optimal heuristic solving")
    optimal_start = time.time()
    ctl.solve()
    optimal_end = time.time()
    optimal_duration = optimal_end - optimal_start
    print("optimal duration:", optimal_duration)
    return {"duration": optimal_duration, **ctl.statistics}


def find_all_programs():
    benchmark_path = project_root.joinpath("asp-planning-benchmarks")
    problem_paths = [path for path in benchmark_path.glob("*") if path.is_dir()]

    problem_collection = list()
    for problem_path in problem_paths:
        problem_dict = dict()
        problem_dict["name"] = problem_path.stem
        problem_dict["rules"] = problem_path.joinpath("encoding_single.asp")
        problem_dict["instances"] = [path for path in problem_path.glob("*") if "encoding" not in path.stem]

        problem_collection.append(problem_dict)

    return problem_collection


def compare_instance_performance(program_dict: dict) -> dict:
    print("instance:", program_dict["name"])
    instance_result = dict()

    # heuristic_ctl = load_program(program_dict)
    # instance_result["heuristic"] = run_heuristic_solving(heuristic_ctl)

    standard_ctl = load_program(program_dict)
    instance_result["standard"] = run_standard_solving(standard_ctl)

    optimal_ctl = load_program(program_dict)
    instance_result["optimal"] = run_optimal_solving(optimal_ctl)

    print()
    return instance_result


def run_performance_measurement():
    problem_collection = find_all_programs()

    problem_results = dict()
    for problem_dict in problem_collection:
        instance_results = list()
        for instance in problem_dict["instances"]:
            program_dict = {"name": instance.stem, "rules": problem_dict["rules"], "facts": instance}
            program_result = compare_instance_performance(program_dict)
            instance_results.append(program_result)

        problem_results[problem_dict["name"]] = instance_results

    return problem_results
