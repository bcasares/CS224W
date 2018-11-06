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

RAW_GEO_PATH = 'Data/Geo/sf_geoboundaries.json'
PROCESSED_GEO_PATH = 'Data/Geo/sf_geoboundaries.shp'
ZONE_INFO_CSV_PATH = 'Data/Geo/sf_zone_info.csv'
MAP_IMAGE_PATH = 'Data/Geo/sf_geoboundaries.png'
MODIFIED_MAP_IMAGE_PATH = 'Data/Geo/sf_geoboundaries_new.png'
TRAVEL_TIMES_PATH = 'Data/Travel_Times/sf_hourly_traveltimes_2018_1.csv'
BORDER_GRAPH_PATH = 'Data/Geo/sf_geoboundaries_borders.graph'
DISTANCE_GRAPH_PATH = 'Data/Geo/sf_geoboundaries_distances.graph'
INTERMEDIATE_UBER_GRAPH_PATH = 'Data/Geo/sf_uber_intermediate_graph.graph'
FINAL_UBER_GRAPH_PATH = 'Data/Geo/sf_uber_final_graph.graph'
MODIFIED_GRAPH_PATH = 'Data/Geo/TimesDistancesGraph.graph'
SF_CENTROID = (-122.445515, 37.751943)
UBER_ZONE_BORDER_IMAGE_PATH = 'Data/Geo/sf_uber_zone_borders_image.png'
FINAL_UBER_GRAPH_IMAGE_PATH = 'Data/Geo/sf_uber_final_image.png'

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
# Plot and save a map using data from .shp file
###########################################################################
###########################################################################
def draw_map(filename, zone_filter=None):
    # Extract polygons
    if zone_filter:
        polys = MultiPolygon([shape(zone['geometry']) for zone in fiona.open(PROCESSED_GEO_PATH) if zone['properties']['id'] in zone_filter])
    else:
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
    patches = []
    for idx, p in enumerate(polys): patches.append(PolygonPatch(p, fc='#AEEDFF', ec='#555555', alpha=1., zorder=1))
    ax.add_collection(PatchCollection(patches, match_original=True))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(filename, alpha=True, dpi=300)
    plt.show()

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
def modify_distance_graph():

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
                speed = 60 * distance/float(mean_travel_time) #in miles per hour
                graph.AddFltAttrDatE(edge_itr, speed, 'travel_speed_'+str(hour_of_day))
            except:
                #print("Failed")
                continue

    # Print number of edges, save intermediate state
    print('[Intermediate] Num nodes: %d' % graph.GetNodes())
    print('[Intermediate] Num edges: %d' % graph.GetEdges())
    FOut = snap.TFOut(INTERMEDIATE_UBER_GRAPH_PATH)
    graph.Save(FOut)

    # Remove edges with no time attributes, then nodes with no connecting edges
    for edge in graph.Edges():
        NameV = snap.TStrV()
        graph.AttrNameEI(edge_itr.GetId(), NameV)
        if len(NameV) < 2:
            graph.DelEdge(edge)
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

################r###########################################################
###########################################################################
# Plot area covered with modified graph
###########################################################################
###########################################################################
def draw_new_map():

    # Load graph
    FIn = snap.TFIn(MODIFIED_GRAPH_PATH)
    graph = snap.TNEANet.Load(FIn)

    # Get node ids (must have edges connected to them)
    node_ids = [node.GetId() for node in graph.Nodes() if node.GetDeg() > 0]

    print(len(node_ids))

    # Draw map
    draw_map(filename=MODIFIED_MAP_IMAGE_PATH, zone_filter=node_ids)

################r###########################################################
###########################################################################
# Make smaller
###########################################################################
###########################################################################
def modify():

    # Load graph
    FIn = snap.TFIn(MODIFIED_GRAPH_PATH)
    graph = snap.TNEANet.Load(FIn)

    nodes_ids = [zone['properties']['id'] for zone in fiona.open(PROCESSED_GEO_PATH)]

    for node in graph.Nodes():
        if not node.GetId() in nodes_ids:
            graph.DelNode(node.GetId())

    print('Nodes: %d' % graph.GetNodes())
    print('Edges: %d' % graph.GetEdges())

    FOut = snap.TFOut("Data/Geo/smaller.graph")
    graph.Save(FOut)

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
        nx_graph.add_node(node.GetId(), pos=(lat_longs[node.GetId()][1],lat_longs[node.GetId()][0]))
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

    nx.draw(nx_graph, pos, node_size=1)
    #plt.savefig(filename, alpha=True, dpi=300)
    plt.show()

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

    # Step 3: Draw map using boundary data in .shp file
    if False:
        draw_map(filename=MAP_IMAGE_PATH)

    # Step 4: Create spatial SNAP graph from zones and borders
    if False:
        data = load_processed_data()
        create_border_graph(data)

    # Step 5: Create spatial SNAP graph from zones and distances between them
    if False:
        data = load_processed_data()
        create_distance_graph(data)

    # Step 6: Add time / speed attributes to graph, remove unncesseary edges and nodes
    if False:
        modify_distance_graph()

    # Draw new map
    if False:
        draw_new_map()

    if False:
        modify()

    # Draw border uber graph (edges based on zone borders)
    if False:
        # Load graph 
        FIn = snap.TFIn(BORDER_GRAPH_PATH)
        graph = snap.TUNGraph.Load(FIn)
        draw_graph(graph, UBER_ZONE_BORDER_IMAGE_PATH, attributes=False)

    # Draw final uber graph (edges based on trips)
    if True:
        # Load graph 
        FIn = snap.TFIn(FINAL_UBER_GRAPH_PATH)
        graph = snap.TNEANet.Load(FIn)
        draw_graph(graph, FINAL_UBER_GRAPH_IMAGE_PATH)

if __name__ == "__main__":
    main()
