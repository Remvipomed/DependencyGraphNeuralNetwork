
import clingo
import networkx as nx


def generate_dependency_graph(ctl):
    graph = nx.DiGraph()
    ctl.ground()

    for atom in ctl.symbolic_atoms:
        if atom.is_fact:
            # Handle facts
            graph.add_node(atom.literal)
        else:
            # Handle rules
            rule = atom.symbol
            for symbol in rule.arguments:
                if symbol.type.name == "Number":
                    graph.add_edge(atom.literal, symbol.number)
    return graph
