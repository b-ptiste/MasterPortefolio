"""
Graph Mining - ALTEGRAD - Oct 2023
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans


############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    ##################
    # your code here #
    ##################
    A = nx.adjacency_matrix(G)
    D_inv = diags([1 / G.degree(node) for node in G.nodes()])

    Lrw = eye(G.number_of_nodes()) - D_inv @ A

    evals, evecs = eigs(Lrw, k=k, which="SR")

    evecs = np.real(evecs)
    kmeans = KMeans(n_clusters=k).fit(evecs)

    clustering = {}

    for i, node in enumerate(G.nodes()):
        clustering[node] = kmeans.labels_[i]

    return clustering


############## Task 7

##################
# your code here #
##################

G = nx.read_edgelist(r"./../code/datasets/CA-HepTh.txt", comments="#", delimiter="\t")
largest_cc = max(nx.connected_components(G), key=len)
subG = G.subgraph(largest_cc)

# we extract 50 clusters for the largest connected component
clustering_of_largest_cc = spectral_clustering(subG, 50)


############## Task 8
# Compute modularity value from graph G based on clustering

def modularity(G, clustering):
    ##################
    # your code here #
    ##################
    m = len(G.edges())
    num_clusters = len(set(clustering.values()))
    modularity = 0

    cluster_lists = [[] for _ in range(num_clusters)]

    # Iterate once over the nodes
    for node, cluster in clustering.items():
        cluster_lists[cluster].append(node)

    for cluster in cluster_lists:
        subG = G.subgraph(cluster)
        lc = subG.number_of_edges()
        dc = sum([G.degree(n) for n in cluster])
        modularity += lc / m - (dc / (2 * m)) ** 2

    return modularity


############## Task 9

##################
# your code here #
##################

print("\n######## TASK 9 ########\n")

k = 50
clustering = spectral_clustering(subG, k)
Q_clustering = modularity(G=subG, clustering=clustering)
print(f"The modularity of the spectral clustering is {Q_clustering}")

random_clustering = {node: np.random.randint(0, k) for node in subG.nodes()}
Q_clustering_random = modularity(G=subG, clustering=random_clustering)
print(f"The modularity of the random clustering is {Q_clustering_random}")
