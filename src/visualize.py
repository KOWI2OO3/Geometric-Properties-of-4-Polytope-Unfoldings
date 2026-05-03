import matplotlib.pyplot as plt
import networkx as nx

def visualizeNet(net, graph):
    # Example visualization
    pos = nx.spring_layout(graph)

    plt.figure(figsize=(8, 8))

    # Original graph in light gray
    nx.draw(graph, pos, edge_color="lightgray", with_labels=True)

    # Tree edges highlighted
    nx.draw(
        net,
        pos,
        edge_color="red",
        width=2,
        with_labels=True
    )

    plt.show()