import snap
import numpy as np
import matplotlib.pyplot as plt


def getDataPointsToPlot(Graph):
    X, Y = [], []
    DegToCntV = snap.TIntPrV()
    snap.GetDegCnt(Graph, DegToCntV)
    for i, item in enumerate(DegToCntV):
        X.append(item.GetVal1())
        Y.append(item.GetVal2()*1.0/Graph.GetNodes())
    return X, Y


def degreeDistribution(Graphs, names, save_as = ""):

    for Graph, name in zip(Graphs,names):
        X, Y =  getDataPointsToPlot(Graph)
        plt.loglog(X, Y, label = name)

    plt.xlabel('Node Degree (log)')
    plt.ylabel('Proportion of Nodes with a Given Degree (log)')
    plt.title('Degree Distribution')
    plt.legend()
    plt.savefig("Plots/" + save_as + "degree_distribution.png")
    # plt.show()


def calculateBowTieStructure(G, name):
    total_nodes = G.GetNodes()
    WCC = snap.GetMxWcc(G).GetNodes()
    SCC_graph = snap.GetMxScc(G)
    SCC = SCC_graph.GetNodes()

    DISCONNECTED = total_nodes - WCC
    cumsum_OUT_vec, cumsum_IN_vec = calCumulativeNodes(G)
    OUT_plus_SCC = max(cumsum_OUT_vec)
    IN_plus_SCC = max(cumsum_IN_vec)
    IN = IN_plus_SCC - SCC
    OUT = OUT_plus_SCC - SCC

    # TENDRILS_TUBES = WCC - SCC - IN - OUT
    print name
    print "Total Nodes", total_nodes
    print "SCC: ", SCC
    print "WCC: ", WCC
    print "Disconnected: ", DISCONNECTED
    print "Out: ", OUT
    print "IN: ", IN
    # print "TENDRILS + TUBES: ", TENDRILS_TUBES
    # print "Total: ", DISCONNECTED + IN + OUT + SCC + TENDRILS_TUBES
    print "Total Estimate: ", DISCONNECTED + IN + OUT + SCC
    return SCC_graph

def calNodeReachability(G, name):
    cumsum_OUT_vec, cumsum_IN_vec = calCumulativeNodes(G)
    plotNodeReachability(cumsum_OUT_vec, cumsum_IN_vec, name = name)

def calCumulativeNodes(G):
    OUT_set = set()
    IN_set = set()
    cumsum_OUT_vec = []
    cumsum_IN_vec = []
    for i in range(100):
        NId = G.GetRndNId()
        bfs_outward_tree = snap.GetBfsTree(G, NId, True, False) #outward
        bfs_inward_tree = snap.GetBfsTree(G, NId, False, True) #inward
        cumsum_IN_vec.append(add_new_nodes(bfs_inward_tree, IN_set))
        cumsum_OUT_vec.append(add_new_nodes(bfs_outward_tree, OUT_set))
    return cumsum_OUT_vec, cumsum_IN_vec

def add_new_nodes(bfs_tree, current_set):
    for node in bfs_tree.Nodes():
        current_set.add(node.GetId())
    return len(current_set)

def plotNodeReachability(cumsum_OUT_vec, cumsum_IN_vec, name = ""):
    frac_starting_nodes = np.arange(1, 101)
    fig = plt.figure()
    plt.plot(frac_starting_nodes, cumsum_OUT_vec, label = "Link reachability")
    plt.title(name + " Reachability (In and Out are the Same)")
    plt.xlabel('Fraction of Starting Nodes')
    plt.ylabel('Number of Nodes Reached')
    plt.legend(loc="best")
    plt.savefig("Plots/" + name + "_reachability.png")


def degreeCount(G, name):
    DegToCntV = snap.TIntPrV()
    snap.GetDegCnt(G, DegToCntV)
    print name
    for item in DegToCntV:
        print "%d nodes with degree %d" % (item.GetVal2(), item.GetVal1())



