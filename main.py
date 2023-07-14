

from lib.heuristic import measurement


def run():
    # training.train_graph_neural_network()
    # prediction.evaluate_graph_neural_network()
    result = measurement.run_performance_measurement()
    print(result)


if __name__ == "__main__":
    run()
