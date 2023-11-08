"""
Graph Mining - ALTEGRAD - Oct 2023
"""

# sapling without replacement sub part and check weather
# nx.is_isomorphoc(G1, G2)
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


############## Task 10
# Generate simple dataset
def create_dataset():
    Gs = list()  # samples
    y = list()  # labels

    ##################
    # your code here #
    ##################

    for i in range(3, 103):
        G = nx.cycle_graph(i)
        Gs.append(G)
        y.append(0)  # means class of cycle

        G = nx.path_graph(i)
        Gs.append(G)
        y.append(1)  # means class of path

    return Gs, y


Gs, y = create_dataset()
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.1)


# Compute the shortest path kernel
def shortest_path_kernel(Gs_train, Gs_test):
    all_paths = dict()
    sp_counts_train = dict()

    for i, G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    sp_counts_test = dict()

    for i, G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(Gs_train), len(all_paths)))
    for i in range(len(Gs_train)):
        for length in sp_counts_train[i]:
            phi_train[i, all_paths[length]] = sp_counts_train[i][length]

    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i, all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


############## Task 11
# Compute the graphlet kernel
def graphlet_kernel(Gs_train, Gs_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]

    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0, 1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0, 1)
    graphlets[2].add_edge(1, 2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0, 1)
    graphlets[3].add_edge(1, 2)
    graphlets[3].add_edge(0, 2)

    phi_train = np.zeros((len(Gs_train), 4))  # we have 4 graph-lets

    ##################
    # your code here #
    ##################

    for i, G in enumerate(Gs_train):
        nodes = list(G.nodes())
        for _ in range(n_samples):
            subset_nodes = np.random.choice(nodes, size=3, replace=False)
            subG = G.subgraph(subset_nodes)
            if nx.is_isomorphic(subG, graphlets[0]):
                phi_train[i, 0] += 1
            if nx.is_isomorphic(subG, graphlets[1]):
                phi_train[i, 1] += 1
            if nx.is_isomorphic(subG, graphlets[2]):
                phi_train[i, 2] += 1
            else:
                phi_train[i, 3] += 1

    phi_test = np.zeros((len(Gs_test), 4))

    ##################
    # your code here #
    ##################

    for i, G in enumerate(Gs_test):
        nodes = list(G.nodes())
        for _ in range(n_samples):
            subset_nodes = np.random.choice(nodes, size=3, replace=False)
            subG = G.subgraph(subset_nodes)
            if nx.is_isomorphic(subG, graphlets[0]):
                phi_test[i, 0] += 1
            if nx.is_isomorphic(subG, graphlets[1]):
                phi_test[i, 1] += 1
            if nx.is_isomorphic(subG, graphlets[2]):
                phi_test[i, 2] += 1
            else:
                phi_test[i, 3] += 1

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)


############## Task 12

##################
# your code here #
##################

K_train_g, K_test_g = graphlet_kernel(G_train, G_test, n_samples=200)


############## Task 13

##################
# your code here #
##################

print("\n######## TASK 13 ########\n")

clf = SVC(kernel="precomputed")
clf.fit(K_train_sp, y_train)
# Predict
y_pred = clf.predict(K_test_sp)
score = accuracy_score(y_pred=y_pred, y_true=y_test)
print(f"The accuracy for shortest path kernel is {score}")


clf = SVC(kernel="precomputed")
clf.fit(K_train_g, y_train)
# Predict
y_pred = clf.predict(K_test_g)
score = accuracy_score(y_pred=y_pred, y_true=y_test)
print(f"The accuracy for graphlet kernel is {score}")

# The accuracy for shortest path kernel is 1.0
# The accuracy for graphlet kernel is 0.45

# The result is not surprising as the graphlet kernel is less discriminative to 
# compare cycles and paths. For instance, both are composed of the same graphlets
# with quite similar distributions. Whereas, the shortest path kernel in a cycle is
# quite different than in a path. In fact the shortest path in a cycle between two
# nodes is bound by the approximatly n/2 (depending if n is odd or even) 
# and in the path by n-1.