"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

##################
# your code here #
##################

print("\n######## TASK 1 ########\n")
G = nx.read_edgelist("./../code/datasets/CA-HepTh.txt", comments="#", delimiter="\t")

print("The graph has ", G.number_of_nodes(), " nodes.")
print("The graph has", G.number_of_edges(), " edges.")

############## Task 2

##################
# your code here #
##################
print("\n######## TASK 2 ########\n")

# The graph is sparse.
# The largest cc contains majority of the nodes and edges

print("The graph has", nx.number_connected_components(G), "connected components.")

largest_cc = max(nx.connected_components(G), key=len)
print(
    "The largest connected component (giant connected component) in the grpah has",
    len(largest_cc),
    "nodes.",
)
print(
    "The largest connected component contains",
    np.round(100 * len(largest_cc)/G.number_of_nodes(), 2),
    "% of nodes of the graph.",
)

subG = G.subgraph(largest_cc)
print(
    "The largest connected component in the graph has",
    subG.number_of_edges(),
    " edges.",
)

print(
    "The largest connected component contains",
    np.round(100 * subG.number_of_edges()/G.number_of_edges(), 2),
    "% of edges of the graph.",
)


############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

##################
# your code here #
##################
print("\n######## TASK 3 ########\n")

print(f"The minimum degree in the graph is : {np.min(degree_sequence)}")
print(f"The maximum degree in the graph is : {np.max(degree_sequence)}")
print(f"The median degree in the graph is : {np.median(degree_sequence)}")
print(f"The mean degree in the graph is : {np.round(np.mean(degree_sequence), 2)}")


############## Task 4

##################
# your code here #
##################
plt.figure(figsize=(10, 7))
plt.plot(nx.degree_histogram(G))
plt.xlabel("degree")
plt.ylabel("frequency")
plt.title("Node degree distribution")
plt.grid(True)
plt.show()

# we observe that the degree distribution follows a power law
plt.figure(figsize=(10, 7))
plt.loglog(nx.degree_histogram(G))
plt.xlabel("log(degree)")
plt.ylabel("log(frequency)")
plt.title("Node degree distribution in log-log scale")
plt.grid(True)
plt.show()

############## Task 5

##################
# your code here #
##################
print("\n######## TASK 5 ########\n")

## clustering methods

# 1) Calculate Lw = I - D
# 2) Obtain the k egeinvectors corresponding to the k smallest egeinvalues of Lw
# 3) k-means to rows of egeinvectors

print("The global clustering coefficient of the graph is", nx.transitivity(G))
