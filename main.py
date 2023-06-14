
import clingo

from lib.heuristic import training, prediction


def run_heuristic():
    training.train_graph_neural_network()
    prediction.evaluate_graph_neural_network()


if __name__ == "__main__":
    run_heuristic()
