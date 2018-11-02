import json
from shapely.geometry import mapping, shape, Polygon, MultiPolygon
import fiona
import csv
import itertools
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch
import snap

RAW_GEO_PATH = 'Data/Geo/sf_geoboundaries.json'
PROCESSED_GEO_PATH = 'Data/Geo/sf_geoboundaries.shp'
ID_NAME_CSV_PATH = 'Data/Geo/sf_ids_and_names.csv'
MAP_IMAGE_PATH = 'Data/Geo/sf_geoboundaries.png'
BORDER_GRAPH_PATH = 'Data/Geo/sf_geoboundaries_borders.graph'
DISTANCE_GRAPH_PATH = 'Data/Geo/sf_geoboundaries_distances.graph'

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
def save_shp(data):

    # Define a polygon feature geometry with one attribute
    schema = {
        'geometry': 'MultiPolygon',
        'properties': {'id': 'int', 'name': 'str'},
    }

    # Save
    with fiona.open(PROCESSED_GEO_PATH, 'w', 'ESRI Shapefile', schema) as c:
        for zone in data:
            c.write({
                'geometry': zone['geometry'],
                'properties': {'id': zone['properties']['MOVEMENT_ID'], 'name': zone['properties']['DISPLAY_NAME']},
            })

###########################################################################
###########################################################################
# Save csv mapping zone ids to zone names
###########################################################################
###########################################################################
def save_ids_and_names():

    # Extract data
    data = [(zone['properties']['id'], zone['properties']['name'].encode('utf-8')) for zone in fiona.open(PROCESSED_GEO_PATH)]

    # Save to csv
    with open(ID_NAME_CSV_PATH,'wb') as f:
        csv_out = csv.writer(f)
        csv_out.writerow(['id','name'])
        for row in data:
            csv_out.writerow(row)

###########################################################################
###########################################################################
# Plot and save a map using data from .shp file
###########################################################################
###########################################################################
def draw_map():

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
    patches = []
    for idx, p in enumerate(polys): patches.append(PolygonPatch(p, fc='#AEEDFF', ec='#555555', alpha=1., zorder=1))
    ax.add_collection(PatchCollection(patches, match_original=True))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(MAP_IMAGE_PATH, alpha=True, dpi=300)
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
            print('Finished checking %d of %d zones' % (count, num_nodes))
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
            print('Finished checking %d of %d zones' % (count, num_nodes))
            prev = cur
            count += 1
        # The important stuff
        zone1_id, zone2_id = zone1['properties']['id'], zone2['properties']['id']
        zone1_centroid, zone2_centroid = shape(zone1['geometry']).centroid, shape(zone2['geometry']).centroid
        distance = zone1_centroid.distance(zone2_centroid)
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
# Main function
###########################################################################
###########################################################################
def main():

    # Step 1: Load raw data and convert boundaries to .shp file
    if False:
        data = load_raw_data()
        save_shp(data)

    # Step 2: Save csv file mapping zone ids to zone names 
    if False:
        save_ids_and_names()

    # Step 3: Draw map using boundary data in .shp file
    if False:
        draw_map()

    # Step 4: Create spatial SNAP graph from zones and borders
    if False:
        data = load_processed_data()
        create_border_graph(data)

    # Step 5: Create spatial SNAP graph from zones and distances between them
    if True:
        data = load_processed_data()
        create_distance_graph(data)

if __name__ == "__main__":
    main()