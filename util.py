import json
from shapely.geometry import mapping, shape, Polygon, MultiPolygon, Point
import fiona
import csv
import itertools
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch
import snap
from math import radians, cos, sin, asin, sqrt
import networkx as nx
import pandas as pd

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

###########################################################################
###########################################################################
# Math function: calculate miles between two points
###########################################################################
###########################################################################
def calc_mile_distance(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # Apply formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 3956 # Radius of earth in miles.
    return c * r

###########################################################################
###########################################################################
# Load raw data (.json file)
###########################################################################
###########################################################################
def load_raw_data():
    # Open geo boundaries file and read in data
    with open(RAW_GEO_PATH) as f:
        data_raw = json.load(f)
    # Boundaries are stored as a list, accessed with 'features' key
    data = data_raw['features']
    return data

###########################################################################
###########################################################################
# Load processed data (.shp file)
###########################################################################
###########################################################################
def load_processed_data():
    # Open shp file and return data
    data = fiona.open(PROCESSED_GEO_PATH)
    return data

###########################################################################
###########################################################################
# Save list of polygons to .shp file
###########################################################################
###########################################################################
def save_shp(data, radius=None):
    # Define a polygon feature geometry with one attribute
    schema = {
        'geometry': 'MultiPolygon',
        'properties': {'id': 'int', 'name': 'str'},
    }
    # Save
    with fiona.open(PROCESSED_GEO_PATH, 'w', 'ESRI Shapefile', schema) as c:
        for zone in data:
            # Check if zone is within radius
            if radius:
                zone_centroid = shape(zone['geometry']).centroid
                #distance = zone_centroid.distance(Point(SF_CENTROID))
                distance = calc_mile_distance(zone_centroid.x, zone_centroid.y, SF_CENTROID[0], SF_CENTROID[1])
                if distance < radius:
                    c.write({
                        'geometry': zone['geometry'],
                        'properties': {'id': zone['properties']['MOVEMENT_ID'], 'name': zone['properties']['DISPLAY_NAME']},
                    })

###########################################################################
###########################################################################
# Save csv mapping zone ids to zone names
###########################################################################
###########################################################################
def save_zone_info():
    # Extract data
    data = [(zone['properties']['id'], zone['properties']['name'].encode('utf-8')) for zone in fiona.open(PROCESSED_GEO_PATH)]
    polys = MultiPolygon([shape(zone['geometry']) for zone in fiona.open(PROCESSED_GEO_PATH)])
    centroids = [x.centroid for x in polys]
    # Save to csv
    with open(ZONE_INFO_CSV_PATH,'wb') as f:
        csv_out = csv.writer(f)
        csv_out.writerow(['id','address', 'longitude', 'latitude'])
        for i, row in enumerate(data):
            to_write = list(row) + [centroids[i].y, centroids[i].x]
            csv_out.writerow(to_write)

###########################################################################
###########################################################################
# Create spatial SNAP graph using geo boundaries
# Graph is undirected, nodes are zone ids, edges exist if zones share border
###########################################################################
###########################################################################
def create_border_graph(data):

    # Initialize new graph
    graph = snap.TUNGraph.New()

    # Add nodes
    for zone in data:
        graph.AddNode(zone['properties']['id'])
    num_nodes = graph.GetNodes()

    # Add edges
    prev, count = '', 0
    for zone1, zone2 in itertools.combinations(data, 2):
        # Just for tracking progress
        cur = zone1['properties']['id']
        if not cur == prev:
            print('[Border graph] Finished checking %d of %d zones' % (count, num_nodes))
            prev = cur
            count += 1
        # The important stuff
        if shape(zone1['geometry']).touches(shape(zone2['geometry'])):
            zone1_id, zone2_id = zone1['properties']['id'], zone2['properties']['id']
            if not graph.IsEdge(zone1_id, zone2_id): graph.AddEdge(zone1_id, zone2_id)
    num_edges = graph.GetEdges()

    # Print some properties of the graph
    print('Number of nodes (zones): %d' % num_nodes)
    print('Number of edges (zone borders): %d' % num_edges)

    # Save graph
    FOut = snap.TFOut(BORDER_GRAPH_PATH)
    graph.Save(FOut)

###########################################################################
###########################################################################
# Create spatial SNAP graph using geo boundaries
# Graph is undirected but weighted
# Nodes are zone ids, edges exist between each node pair, weighted by distance between zones
###########################################################################
###########################################################################
def create_distance_graph(data):

    # Initialize new graph
    graph = snap.TNEANet.New()

    # Add nodes
    for zone in data:
        graph.AddNode(zone['properties']['id'])
    num_nodes = graph.GetNodes()

    # Add edges
    prev, count = '', 0
    edge_id = 0
    for zone1, zone2 in itertools.combinations(data, 2):
        # Just for tracking progress
        cur = zone1['properties']['id']
        if not cur == prev:
            print('[Distance graph] Finished checking %d of %d zones' % (count, num_nodes))
            prev = cur
            count += 1
        # The important stuff
        zone1_id, zone2_id = zone1['properties']['id'], zone2['properties']['id']
        zone1_centroid, zone2_centroid = shape(zone1['geometry']).centroid, shape(zone2['geometry']).centroid
        distance = calc_mile_distance(zone1_centroid.x, zone1_centroid.y, zone2_centroid.x, zone2_centroid.y)
        if not graph.IsEdge(zone1_id, zone2_id):
            graph.AddEdge(zone1_id, zone2_id, edge_id)
            graph.AddFltAttrDatE(edge_id, distance, 'distance')
            edge_id += 1
        if not graph.IsEdge(zone2_id, zone1_id):
            graph.AddEdge(zone2_id, zone1_id, edge_id)
            graph.AddFltAttrDatE(edge_id, distance, 'distance')
            edge_id += 1
    num_edges = graph.GetEdges()

    # Print some properties of the graph
    print('Number of nodes (zones): %d' % num_nodes)
    print('Number of edges (zone borders): %d' % num_edges)

    # Save graph
    FOut = snap.TFOut(DISTANCE_GRAPH_PATH)
    graph.Save(FOut)

###########################################################################
###########################################################################
# Add time attributes to zones and distances graph from travel time data file
# Remove edges without time attributes
###########################################################################
###########################################################################
def modify_distance_graph(): # sec * (1 min / 60 sec) * (60 min / 1 hr)

    # Load graph
    FIn = snap.TFIn(DISTANCE_GRAPH_PATH)
    graph = snap.TNEANet.Load(FIn)

    # Add time and speed attributes to edges corresponding to each row in Travel Times csv file
    with open(TRAVEL_TIMES_PATH) as f:
        travel_times_reader = csv.reader(f)
        travel_times_reader.next()
        for i, row in enumerate(travel_times_reader):
            if i % 100000 == 0: print('On row %d' % i)
            try:
                source_id, dest_id, hour_of_day, mean_travel_time = row[:4]
                #print source_id, dest_id, hour_of_day, mean_travel_time
                edge_itr = graph.GetEI(int(source_id), int(dest_id))
                # Add travel time attribute
                graph.AddFltAttrDatE(edge_itr, float(mean_travel_time), 'travel_time_'+str(hour_of_day))
                # Add speed attribute
                distance = graph.GetFltAttrDatE(edge_itr.GetId(), 'distance')
                speed = (distance / float(mean_travel_time)) * 3600.0 # in miles per hour
                graph.AddFltAttrDatE(edge_itr, speed, 'travel_speed_'+str(hour_of_day))
            except Exception as e:
                #print(e)
                continue

    # Print number of edges, save intermediate state
    print('[Intermediate] Num nodes: %d' % graph.GetNodes())
    print('[Intermediate] Num edges: %d' % graph.GetEdges())
    FOut = snap.TFOut(INTERMEDIATE_UBER_GRAPH_PATH)
    graph.Save(FOut)

    # Remove edges with no time attributes, then nodes with no connecting edges
    for edge in graph.Edges():
        NameV = snap.TStrV()
        graph.AttrNameEI(edge.GetId(), NameV)
        if len(NameV) < 2:
            graph.DelEdge(edge.GetId())
    for node in graph.Nodes():
        if node.GetDeg() == 0:
            graph.DelNode(node.GetId())

    # Print number of edges, save final state
    print('[Final] Num nodes: %d' % graph.GetNodes())
    print('[Final] Num edges: %d' % graph.GetEdges())
    FOut = snap.TFOut(FINAL_UBER_GRAPH_PATH)
    graph.Save(FOut)

    # NameV = snap.TStrV()
    # print("Edge ID: " + str(edge_itr.GetId()))
    # graph.AttrNameEI(edge_itr.GetId(), NameV)
    # for val in NameV:
    #     print val

    # ValueV = snap.TStrV()
    # graph.AttrValueEI(edge_itr.GetId(), ValueV)
    # for val in ValueV:
    #     print val

###########################################################################
###########################################################################
# Plot and save a map using data from .shp file
###########################################################################
###########################################################################
def draw_map(filename, plot_centroids=False, scale_centroids=False, plot_edges=False, graph=None, centroid_classes=False):
    # Extract polygons
    polys = MultiPolygon([shape(zone['geometry']) for zone in fiona.open(PROCESSED_GEO_PATH)])
    # Setup plot
    fig = plt.figure(figsize=(15, 20))
    ax = fig.add_subplot(111)
    min_x, min_y, max_x, max_y = polys.bounds
    w, h = max_x - min_x, max_y - min_y
    ax.set_xlim(min_x - 0.2 * w, max_x + 0.2 * w)
    ax.set_ylim(min_y - 0.2 * h, max_y + 0.2 * h)
    ax.set_aspect(1)
    # Plot, save, and show
    # Plot zones
    patches = []
    for idx, p in enumerate(polys): patches.append(PolygonPatch(p, fc='#AEEDFF', ec='#555555', alpha=1., zorder=1))
    ax.add_collection(PatchCollection(patches, match_original=True))
    # Plot edges
    if plot_edges and graph:
        zone_info = pd.read_csv(ZONE_INFO_CSV_PATH)
        lat_longs = {}
        for i, row in zone_info.iterrows(): lat_longs[row.id] = (row.latitude, row.longitude)
        for edge in graph.Edges():
            start = (lat_longs[edge.GetSrcNId()][0], lat_longs[edge.GetSrcNId()][1])
            end = (lat_longs[edge.GetDstNId()][0], lat_longs[edge.GetDstNId()][1])
            ax.plot([start[0], end[0]], [start[1], end[1]], color='g', linewidth='1')
    # Plot and scale centroids
    if plot_centroids and scale_centroids and graph:
        zone_info = pd.read_csv(ZONE_INFO_CSV_PATH)
        # Scale weights so that largest is 50
        weights = {}
        for i, row in zone_info.iterrows(): weights[row.id] = graph.GetFltAttrDatN(int(row.id), 'weight')
        largest = max(weights.itervalues())
        for key in weights: weights[key] = (weights[key] / largest) * 50
        # Plot
        lats, longs, scales = [], [], []
        for i, row in zone_info.iterrows(): 
            lats.append(row.latitude)
            longs.append(row.longitude)
            scales.append(weights[row.id])
        ax.scatter(lats, longs, s=scales, c=scales, cmap=plt.cm.get_cmap('plasma'))
    if plot_centroids and centroid_classes:
        print(centroid_classes)
        zone_info = pd.read_csv(ZONE_INFO_CSV_PATH)
        lats = []
        longs = []
        colors = []
        for i, row in zone_info.iterrows(): 
            lats.append(row.latitude)
            longs.append(row.longitude)
            colors.append(centroid_classes[row.id])
        ax.scatter(lats, longs, c=colors, cmap='viridis')
    # Plot centroids
    elif plot_centroids:
        zone_info = pd.read_csv(ZONE_INFO_CSV_PATH)
        for i, row in zone_info.iterrows(): 
            ax.scatter(row.latitude, row.longitude, color='r', s=20)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(filename, alpha=True, dpi=300)
    #plt.show()

################r###########################################################
###########################################################################
# Draw the uber travel data graph
###########################################################################
###########################################################################
def draw_graph(graph, filename, attributes=True):

    # Load zone info
    zone_info = pd.read_csv(ZONE_INFO_CSV_PATH)
    lat_longs = {}
    for i, row in zone_info.iterrows():
        lat_longs[row.id] = (row.latitude, row.longitude)

    # New nx graph
    nx_graph = nx.Graph()
    for node in graph.Nodes():
        nx_graph.add_node(node.GetId(), pos=(lat_longs[node.GetId()][0],lat_longs[node.GetId()][1]))
    for edge in graph.Edges():
        if attributes:
            nameV = snap.TStrV()
            graph.FltAttrNameEI(edge.GetId(), nameV)
            times = [name for name in list(nameV) if 'travel_time_' in name]
            values = [graph.GetFltAttrDatE(edge.GetId(), time) for time in times]
            try:
                nx_graph.add_edge(edge.GetSrcNId(), edge.GetDstNId(), weight=float(sum(values))/len(values))
            except:
                nx_graph.add_edge(edge.GetSrcNId(), edge.GetDstNId())
        else:
            nx_graph.add_edge(edge.GetSrcNId(), edge.GetDstNId())

    pos = nx.get_node_attributes(nx_graph, 'pos')

    plt.figure(figsize=(7,12))
    nx.draw(nx_graph, pos, node_size=1)
    plt.savefig(filename)
    #plt.show()

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
                print(weight)
        # If doing avg degree
        if average: degree /= num_out_nodes
        graph.AddFltAttrDatN(node_id, degree, 'weight')

    # Return
    return graph

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
# Main function
###########################################################################
###########################################################################
def main():

    # Step 1: Load raw data and convert boundaries to .shp file
    if False:
        data = load_raw_data()
        save_shp(data, radius=14) # Radius of 14 miles from center of SF

    # Step 2: Save csv file mapping zone ids to zone names and zone centroids
    if False:
        save_zone_info()

    # Step 3: Create spatial SNAP graph from zones and borders
    if False:
        data = load_processed_data()
        create_border_graph(data)

    # Step 4: Create spatial SNAP graph from zones and distances between them
    if False:
        data = load_processed_data()
        create_distance_graph(data)

    # Step 5: Add time / speed attributes to graph, remove unncesseary edges and nodes
    if True:
        modify_distance_graph()

    # Step 6: Draw map using boundary data in .shp file and graph
    if False:
        # Draw map with zones and centroids
        draw_map(filename=MAP_IMAGE_PATH, plot_centroids=True)
        # Draw map with zones, centroids, and edges based on borders
        FIn = snap.TFIn(BORDER_GRAPH_PATH)
        graph = snap.TUNGraph.Load(FIn)
        draw_map(filename=UBER_ZONE_BORDER_IMAGE_PATH, plot_centroids=True, plot_edges=True, graph=graph)

    # Step 7: Draw final uber graph (edges based on trips)
    if False:
        # Load graph 
        FIn = snap.TFIn(FINAL_UBER_GRAPH_PATH)
        graph = snap.TNEANet.Load(FIn)
        draw_graph(graph, FINAL_UBER_GRAPH_IMAGE_PATH)

    ###########################################################################################
    # Experiments
    ###########################################################################################

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

    # TESTING
    if False:
        # Load graph 
        FIn = snap.TFIn(FINAL_UBER_GRAPH_PATH)
        original_graph = snap.TNEANet.Load(FIn)
        new_graph = compute_node_degree(original_graph, 'travel_speed_12')

if __name__ == "__main__":
    main()
