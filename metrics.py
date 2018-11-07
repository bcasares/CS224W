from util import *

from collections import defaultdict
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from scipy.stats import wasserstein_distance


# Source files / other
RAW_GEO_PATH = 'Data/Geo/sf_geoboundaries.json'
PROCESSED_GEO_PATH = 'Data/Geo/sf_geoboundaries.shp'
ZONE_INFO_CSV_PATH = 'Data/Geo/sf_zone_info.csv'
TRAVEL_TIMES_PATH = 'Data/Travel_Times/sf_hourly_traveltimes_2018_1.csv'
SF_CENTROID = (-122.445515, 37.751943)

# Graphs
BORDER_GRAPH_PATH = 'Data/Geo/Graphs/sf_geoboundaries_borders.graph'
DISTANCE_GRAPH_PATH = 'Data/Geo/Graphs/sf_geoboundaries_distances.graph'
INTERMEDIATE_UBER_GRAPH_PATH = 'Data/Geo/Graphs/sf_uber_intermediate_graph.graph'
FINAL_UBER_GRAPH_PATH = 'Data/Geo/Graphs/sf_uber_final_graph.graph'

# Output images
MAP_IMAGE_PATH = 'Data/Geo/Images/sf_geoboundaries.png'
UBER_ZONE_BORDER_IMAGE_PATH = 'Data/Geo/Images/sf_uber_zone_borders_image.png'
FINAL_UBER_GRAPH_IMAGE_PATH = 'Data/Geo/Images/sf_uber_final_image.png'

################r###########################################################
###########################################################################
# Build new graph with edges having only a single attribute
###########################################################################
###########################################################################
def build_single_weight_graph(original_graph, attribute):

    # Initialize new graph
    graph = snap.TNEANet.New()

    # Add nodes
    for node in original_graph.Nodes():
        graph.AddNode(node.GetId())
    num_nodes = graph.GetNodes()

    # Add edges
    for edge in original_graph.Edges():
        src, dst, edge_id = edge.GetSrcNId(), edge.GetDstNId(), edge.GetId()
        graph.AddEdge(src, dst, edge_id)
        weight = original_graph.GetFltAttrDatE(edge_id, attribute)
        graph.AddFltAttrDatE(edge_id, weight, 'weight')

    # Print num nodes and edges
    #print('[Original] Num nodes: %d, Num edges: %d' % (original_graph.GetNodes(), original_graph.GetEdges()))
    #print('[New] Num nodes: %d, Num edges: %d' % (graph.GetNodes(), graph.GetEdges()))

    # Return
    return graph

################r###########################################################
###########################################################################
# Build new graph with edges having only a single attribute
###########################################################################
###########################################################################
def compute_node_degree(original_graph, attribute, average=False, only_zone_neighbors=False, zone_neighbor_graph=None):

    # Create new graph using desired attribute
    graph = build_single_weight_graph(original_graph, attribute)

    # Loop through all nodes, add attribute to each that is the sum of all adjacent edge weights
    for node in graph.Nodes():
        node_id, num_out_nodes = node.GetId(), node.GetOutDeg()
        degree = 0
        for i in range(num_out_nodes):
            neighbor_id = node.GetOutNId(i)
            # If we only want to consider neighboring zones
            if only_zone_neighbors and zone_neighbor_graph: include = zone_neighbor_graph.IsEdge(node_id, neighbor_id)
            # Else include everything
            else: include = True
            # Add to total degree
            if include:
                edge_id = graph.GetEI(node_id, neighbor_id).GetId()
                weight = graph.GetFltAttrDatE(edge_id, 'weight')
                if weight > 0: degree += weight # For some reason a few weights are -inf
        # If doing avg degree
        if average: degree /= num_out_nodes
        graph.AddFltAttrDatN(node_id, degree, 'weight')

    # Return
    return graph

###########################################################################
###########################################################################
#Get edge weights for node
###########################################################################
###########################################################################
def get_edge_weights(original_graph, attribute):

    # Create new graph using desired attribute
    graph = build_single_weight_graph(original_graph, attribute)

    edgeWeights = defaultdict(list)
    # Loop through all nodes, add attribute to each that is the sum of all adjacent edge weights
    for node in graph.Nodes():
        node_id, num_out_nodes = node.GetId(), node.GetOutDeg()
        degree = 0
        for i in range(num_out_nodes):
            neighbor_id = node.GetOutNId(i)
            edge_id = graph.GetEI(node_id, neighbor_id).GetId()
            weight = graph.GetFltAttrDatE(edge_id, 'weight')
            if weight > 0: 
                edgeWeights[node_id].append(weight) # For some reason a few weights are -inf

    # Return
    return edgeWeights

def find_clusters(X, n_clusters, rseed=2):
        # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    
    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers, metric=wasserstein_distance)
        
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_clusters)])
        
        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers
    
    return centers, labels

###########################################################################
###########################################################################
# Main function
###########################################################################
###########################################################################
def main():

    # Compute / plot node degrees (sum of all adjacent edge weights)
    if False:
        # Load graph 
        FIn = snap.TFIn(FINAL_UBER_GRAPH_PATH)
        original_graph = snap.TNEANet.Load(FIn)
        # Compute node degree for various attributes
        attributes = ['travel_time_6', 'travel_time_12', 'travel_time_18', \
                        'travel_speed_6', 'travel_speed_12', 'travel_speed_18']
        for attribute in attributes:
            new_graph = compute_node_degree(original_graph, attribute)
            # Plot
            draw_map('Data/Geo/Images/node_degree_'+attribute+'.png', \
                        plot_centroids=True, scale_centroids=True, graph=new_graph)

    # Compute average node degree over time
    if False:
        # Load graph 
        FIn = snap.TFIn(FINAL_UBER_GRAPH_PATH)
        original_graph = snap.TNEANet.Load(FIn)
        # Compute node degree for various attributes
        attributes = [('travel_time', 'minutes'), ('travel_speed', 'mph')]
        for attribute, label in attributes:
            # Compute
            results = []
            for hour in range(24):
                new_graph = compute_node_degree(original_graph, attribute+'_'+str(hour), average=True)
                avg_degree = 0
                for node in new_graph.Nodes():
                    avg_degree += new_graph.GetFltAttrDatN(node.GetId(), 'weight')
                avg_degree /= float(new_graph.GetNodes())
                if attribute == 'travel_time': avg_degree /= 60.0
                print('[%d] %.2f' % (hour, avg_degree))
                results.append(avg_degree)
            # Plot
            plt.figure(figsize=(15,10))
            plt.plot(range(24), results)
            plt.xlabel('Hour of Day')
            plt.xticks(range(24))
            plt.ylabel('Average ' + ' '.join(attribute.split('_')[:2]) + '(' + label + ')')
            plt.title('Average ' + ' '.join(attribute.split('_')[:2]) + ' by Hour of Day')
            plt.savefig('Data/Geo/Images/' + attribute + '_avg_by_hour')

    if True:
        FIn = snap.TFIn(FINAL_UBER_GRAPH_PATH)
        original_graph = snap.TNEANet.Load(FIn)
        edgeWeightDistributions = []
        attributes = ['travel_time_12']
        for attribute in attributes:
            edgeWeights = get_edge_weights(original_graph, attribute)
            #1. Convert each node array of edges to histogram
                #Find global min and max values of weights
            minWeight = min([min(weights) for node, weights in edgeWeights.iteritems()])
            maxWeight = max([max(weights) for node, weights in edgeWeights.iteritems()])
            edgeWeightHistograms = {}
            for node, weights in edgeWeights.iteritems():
                histo = np.histogram(weights, bins=[  21.65,341.75083333,661.85166667,981.9525,1302.05333333, 1622.15416667, 1942.255, 2262.35583333,2582.45666667, 2902.5575,3222.65833333,3542.75916667,3862.86],
                    range=(minWeight, maxWeight))
                edgeWeightHistograms[node] = list(histo[0])

            nodes, histograms = zip(*edgeWeightHistograms.iteritems())
            centers, labels = find_clusters(np.array(histograms), 3)
            node_roles = dict(zip(nodes, labels))
            draw_map(attribute+'_node_roles', plot_centroids=True, centroid_classes=node_roles)




if __name__ == "__main__":
    main()
