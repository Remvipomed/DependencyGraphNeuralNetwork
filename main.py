
import clingo

from lib.heuristic import training, prediction


def perform_heuristic():
    # training.train_graph_neural_network()
    prediction.run_heuristic()


if __name__ == "__main__":
    perform_heuristic()
