import snap
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys

def q2_1():

    # Load graph
    Graph = nx.read_edgelist("../graph/karate.edgelist")

    # Visualize 
    nx.draw(Graph, with_labels=True)
    plt.show()

def load_embeddings(filename):

    with open(filename) as f:  
        # Determine number of nodes and size of embedding
        line = f.readline()
        count = 1
        nodes = int(line.strip().split(' ')[0])
        size = int(line.strip().split(' ')[1])
        # Initialize numpy matrix
        matrix = np.zeros((nodes, size))
        # Add to matrix
        while line:
            line = f.readline()
            try:
                node = int(line.strip().split(' ')[0])
                values = [float(val) for val in line.strip().split(' ')[1:]]
                matrix[node-1, :] = values
            except:
                pass

    return (nodes, matrix)

def q2_3():

    # Load embeddings
    nodes, matrix = load_embeddings("emb/karate-0505.emb")

    # Compute dot products (cosine similarities)
    results = {}
    for i in range(nodes):
        node_num = i + 1
        if not node_num == 33:
            results[node_num] = matrix[i, :].dot(matrix[32, :])

    # Print results
    for key in sorted(results, key=results.get, reverse=True):
        print('Node: %d, Sim: %.4f' % (key, results[key]))

def q2_4():

    # Print degrees
    if False:
        Graph = nx.read_edgelist("graph/karate.edgelist")
        degrees = {}
        for i in range(1,35):
            degrees[i] = Graph.degree(str(i))
        for key in sorted(degrees, key=degrees.get, reverse=True):
            print('Node: %d, Degree: %d' % (key, degrees[key]))

    # Parameters to try
    all_p = [0.05, 0.1]
    all_q = [0.001, 0.01]
    highest_degrees = [34, 1, 33, 3, 2, 4, 32, 14, 24]
    for p in all_p:
        for q in all_q:

            # Run
            os.system("python src/main.py --input graph/karate.edgelist --output emb/karate-out.emb --walk-length 40 --num-walks 100 --p %s --q %s" % (str(p), str(q)))

            # Load embeddings
            nodes, matrix = load_embeddings("emb/karate-out.emb")

            # Compute dot products (cosine similarities)
            results = {}
            for i in range(nodes):
                node_num = i + 1
                if not node_num == 34:
                    results[node_num] = matrix[i, :].dot(matrix[33, :])

            # Print results
            count, found = 0, 0
            for key in sorted(results, key=results.get, reverse=True):
                print('Node: %d, Sim: %.4f' % (key, results[key]))
                count += 1
                if key in highest_degrees: found += 1
                if count == 5: break
            print('p: %.3f, q: %.3f, found: %d' % (p, q, found))

def q2_5():

    if True:
        # Load embeddings
        nodes, matrix = load_embeddings("emb/karate-100-01.emb")

        # Compute dot products (cosine similarities)
        results = {}
        for i in range(nodes):
            node_num = i + 1
            if not node_num == 33:
                row1, row2 = matrix[i, :], matrix[32, :]
                results[node_num] = np.linalg.norm(row1 - row2)

        # Print results
        for key in sorted(results, key=results.get):
            print('Node: %d, Distance: %.4f' % (key, results[key]))

    # Print degrees
    Graph = nx.read_edgelist("graph/karate.edgelist")
    print('Node 33: %d' % Graph.degree('33'))
    print('Node 15: %d' % Graph.degree('10'))
    print('Node 15 neighbors: ' + str([n for n in Graph.neighbors('15')]))

if __name__ == "__main__":
    q2_1()
    q2_3()
    q2_4()
    q2_5()

    print "Done with Question 2!\n"