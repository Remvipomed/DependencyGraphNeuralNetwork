

from lib.heuristic import measurement
from lib.heuristic import training


def run():
    # training.train_graph_neural_network()
    # prediction.evaluate_graph_neural_network()
    # result = measurement.run_performance_measurement()
    result = measurement.run_single_measurement()
    print(result)


if __name__ == "__main__":
    run()
