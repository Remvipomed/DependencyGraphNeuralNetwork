
import clingo

from lib import project_dir
from lib.encoding import graph
from lib.encoding import dataset
from lib.heuristic import training


def run_dependency_training():
    data = dataset.HanoiDatasetSingle()
    training.train_graph_neural_network(data)


if __name__ == "__main__":
    run_dependency_training()
