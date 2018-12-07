from collections import defaultdict
import copy
from datetime import datetime
import googlemaps
import os
import pandas as pd
import snap
from tqdm import tqdm
import metrics
import geohash
import hashlib
from metrics import compute_centrality

import fiona
from shapely.geometry import mapping, shape, Polygon, MultiPolygon, Point
PROCESSED_GEO_PATH = 'Data/Geo/sf_geoboundaries.shp'
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch


# GOOGLE_API_KEY = "AIzaSyAi5ERQfBOcdO54IT3MOmiCnNubFB1RWWY" # First Key, no-credit
# GOOGLE_API_KEY = "AIzaSyCnzf2UnjffaNCRX_rGvlypYBZb5rt6zyM" #abecs224w1 no-credit
# GOOGLE_API_KEY = "AIzaSyCUHPr2e0NmO5lwrB5cHuVUCDVeSDYWDQ0" #abecs224w2 no-credit
# GOOGLE_API_KEY = "AIzaSyD7_s0eA-M967PpPLQfu9sKwnkiNVSHoE8" #abecs224w3 still works
# GOOGLE_API_KEY = "AIzaSyAqAw4fAIRPP2eavNrNhBSn-f0r1PuDNCc" #abecs224w4


SF_ZONE_INFO = os.path.join("Data", "Geo", "sf_zone_info.csv")
SF_REDUCED_GRAPH = os.path.join("Data", "Geo", "Graphs", "sf_uber_final_graph.graph")
PUBLIC_TRANSIT_GRAPH_PATH_SAVE = os.path.join("Data", "Geo", "Graphs", "public_transit.graph")
PUBLIC_TRANSIT_PLUS_INTER_GRAPH_PATH_SAVE = os.path.join("Data", "Geo", "Graphs", "public_transit_plus_inter.graph")
PUBLIC_TRANSIT_PLUS_INTER_GRAPH_WEIGHTED_PATH_SAVE = os.path.join("Data", "Geo", "Graphs", "public_transit_plus_inter_weighted.graph")
# PUBLIC_TRANSIT_GRAPH_PATH_LOAD = os.path.join("Data", "Geo","Graphs",  "public_transit_complete.graph")
# PUBLIC_TRANSIT_GRAPH_PATH_LOAD = os.path.join("Data", "Geo", "Graphs", "public_transit_reduced5pm.graph")
PUBLIC_TRANSIT_GRAPH_PATH_LOAD = os.path.join("Data", "Geo", "Graphs", "public_transit_plus_inter.graph")
PUBLIC_TRANSIT_GRAPH_WEIGHTED_PATH_LOAD = os.path.join("Data", "Geo", "Graphs", "public_transit_plus_inter_weighted.graph")
PUBLIC_TRANSIT_GRAPH_PLOT = os.path.join("Data", "Geo", "Images", "public_transit_graph_5pm.png")
# PUBLIC_TRANSIT_GRAPH_PLOT = os.path.join("Data", "Geo", "Images", "public_transit_graph_reduced.png")

class PublicTransport(object):

    """Public Transport Graph for CS224W Project. """

    def __init__(self, create_new=False, read_google_maps=False, plot_graph=False, check_attributes=False, reduce_graph=False, graph_path=None):
        self.data = self.loadData()
        self.hash_to_node = {}
        self.node_to_hash = {}
        self.createMapping()
        if graph_path: self.loadGraph(create_new=create_new, file_path=graph_path)
        else: self.loadGraph(create_new=create_new)
        if reduce_graph:
            self.reduceGraph()
        if read_google_maps:
            pass
            # self.readGoogleMapsAndCreateGraph()
            # self.readGoogleMapsAndCreateGraphFromExsitingGraph()
            # self.readGoogleMapsAndSaveResponse()
        if plot_graph:
            self.drawGraph()
        if check_attributes:
            self.checkAttributes()

    def createMapping(self):
        self.node_to_hash = {}
        self.hash_to_node = {}
        for i, row in self.data.iterrows():
            hash_name = geohash.encode(round(row.longitude,3), round(row.latitude,3))
            self.hash_to_node[hash_name] = row.id
            self.node_to_hash[str(row.id)] = hash_name

        self.node_count_when_builidng = max(self.data.id) + 1



    def loadData(self, file_path=SF_ZONE_INFO):
        """
        Load Uber zone data from sf_zone_info.csv
        """
        data = pd.read_csv(file_path)
        def latLong(row):
            # return str(round(row.longitude,3)) + ", " + str(round(row.latitude,3))
            return str(row.longitude) + ", " + str(row.latitude)
            # return str(row.latitude) + ", " + str(row.longitude)
        data["lat_long"] = data.apply(latLong, axis = 1)
        return data

    def loadGraph(self, file_path=PUBLIC_TRANSIT_GRAPH_PATH_LOAD, create_new=False, to_print=True):
        """
        Loads the graph if the graph already exists
        """
        # Load graph
        print file_path
        if create_new:
            self.graph = snap.TNEANet.New()
        else:
            # try :
                FIn = snap.TFIn(file_path)
                self.graph = snap.TNEANet.Load(FIn)
                self.getNodeToHashHashToNode()
                if to_print:
                    print "Number of nodes", self.graph.GetNodes()
                    print "Number of edges", self.graph.GetEdges()
            # except :
            #     if to_print:
            #         print("creating new graph")
            #     self.graph = snap.TNEANet.New()

    def getNodeToHashHashToNode(self):
        for i, row in pd.read_csv("Data/ExtraPublicTransit/hash_node.csv").iterrows():
            # print(type(row.hash_name), type(row.node_id))
            self.hash_to_node[row.hash_name] = row.node_id
            self.node_to_hash[str(row.node_id)] = row.hash_name


    def saveGraph(self, file_path=PUBLIC_TRANSIT_GRAPH_PATH_SAVE, i = 0, j = 0):
        """
        Save graph and outputs the iteration number to a temp file.
        """
        print("saving graph", i, j)
        FOut = snap.TFOut(file_path)
        self.graph.Save(FOut)
        with open('temp.txt', 'w') as f:
            print >> f, 'i=', i, 'j=', j

    def drawGraph(self, file_path=PUBLIC_TRANSIT_GRAPH_PLOT):
        metrics.draw_graph(self.graph, file_path)


    def checkAttributes(self, to_print=True):
        distances_m = []
        times_s = []
        for edge in self.graph.Edges():
            distance_m = self.graph.GetFltAttrDatE(edge.GetId(), "distance_meters")
            time_s = self.graph.GetFltAttrDatE(edge.GetId(), "duration_seconds")
            distances_m.append(distance_m)
            times_s.append(time_s)

        avg_distance_m = sum(distances_m)/len(distances_m)
        avg_distance_miles = avg_distance_m/1609.34

        avg_time_s = sum(times_s)/len(times_s)
        avg_time_min = avg_time_s/60
        avg_time_h = avg_time_min/60

        avg_speed_miles_hour = avg_distance_miles/avg_time_h

        if to_print:
            print "The average travel distance is", avg_distance_m, "meters"
            print "or", avg_distance_miles, "miles"
            print "The average travel time is", avg_time_s, "seconnds"
            print "or", avg_time_min, "minutes"
            print "or", avg_time_h, "hours"
            print "The average speed is", avg_speed_miles_hour, "miles per hour"

    def saveGraphToCSVPredictionAnalysis(self, file_path="Data/ExtraPublicTransit/public_trans_graph_df.cvs"):

        distances_m = []
        times_s = []
        origin_nodes = []
        destination_nodes = []
        for edge in self.graph.Edges():
            distance_m = self.graph.GetFltAttrDatE(edge.GetId(), "distance_meters")
            time_s = self.graph.GetFltAttrDatE(edge.GetId(), "duration_seconds")
            distances_m.append(distance_m)
            times_s.append(time_s)
            origin_nodes.append(edge.GetSrcNId())
            destination_nodes.append(edge.GetDstNId())
        d = {"origin": origin_nodes, "destination" : destination_nodes,
                "distance meters" : distances_m, "time sec" : times_s}
        df = pd.DataFrame(data=d)
        df.to_csv(file_path)

    def reduceGraph(self, load_reduced_graph_path=SF_REDUCED_GRAPH, to_print = True):
        FIn = snap.TFIn(load_reduced_graph_path)
        temp_graph = snap.TNEANet.Load(FIn)
        for edge in self.graph.Edges():
            if temp_graph.IsNode(edge.GetSrcNId()) and temp_graph.IsNode(edge.GetDstNId()):
                if not temp_graph.IsEdge(edge.GetSrcNId(), edge.GetDstNId()):
                    self.graph.DelEdge(edge.GetId())

        # Delete a node that had no neighbors, this is done manually once
        self.graph.DelNode(244)
        if to_print:
            print "Reducing graph..."
            print "Number of Nodes", self.graph.GetNodes()
            print "Number of Edges", self.graph.GetEdges()

    def readGoogleMapsAndSaveResponse(self, date_time='Dec 12 2018  5:00PM'):
        """
        Use Google API to compute travel time between the points for each time of day
        (using public transportation)
        """
        gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

        # now = datetime.now()
        now = datetime.strptime(date_time, '%b %d %Y %I:%M%p')

        edge_id=0
        temp_graph = self.graph
        rows1 = []
        rows2 = []
        directions = []
        for count, edge in tqdm(enumerate(temp_graph.Edges())):
            if count % 2 == 0 :
                try :
                    row_1 = self.data[self.data.id == edge.GetSrcNId()].iloc[0]
                    row_2 = self.data[self.data.id == edge.GetDstNId()].iloc[0]
                    directions_result = gmaps.directions(row_1.lat_long,
                                                         row_2.lat_long,
                                                         mode="transit",
                                                         departure_time=now)
                    rows1.append(row_1.lat_long)
                    rows2.append(row_2.lat_long)
                    directions.append(directions_result)
                    if (count % 1000) == 0 :
                        d = {'node1': rows1, 'node2':rows2, "response": directions}
                        df = pd.DataFrame(data = d)
                        df.to_pickle("./dummy.pkl")
                        print count

                except Exception as ex:
                    print ex
                    continue
        d = {'node1': rows1, 'node2':rows2, "response": directions}
        df = pd.DataFrame(data = d)
        df.to_pickle("./dummy.pkl")
        print "Done"

    def readGoogleMapsAndCreateGraphFromExsitingGraph(self, date_time='Dec 12 2018  5:00PM'):
        """
        Use Google API to compute travel time between the points for each time of day
        (using public transportation)
        """
        gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

        # now = datetime.now()
        now = datetime.strptime(date_time, '%b %d %Y %I:%M%p')

        edge_id=0
        temp_graph = self.graph
        self.graph = snap.TNEANet.New()
        for count, edge in tqdm(enumerate(temp_graph.Edges())):
            if count % 2 == 0 :
                try :
                    row_1 = self.data[self.data.id == edge.GetSrcNId()].iloc[0]
                    row_2 = self.data[self.data.id == edge.GetDstNId()].iloc[0]
                    directions_result = gmaps.directions(row_1.lat_long,
                                                         row_2.lat_long,
                                                         mode="transit",
                                                         departure_time=now)

                    distance_meters = directions_result[0]['legs'][0]['distance']['value']
                    duration_seconds = directions_result[0]['legs'][0]['duration']['value']

                    self.addNodesAndEdgesPublicTransitGraph(row_1.id, row_2.id, edge_id, distance_meters, duration_seconds)
                    edge_id+=2

                    if (count % 1000) == 0 :
                        self.saveGraph(file_path=PUBLIC_TRANSIT_GRAPH_PATH_SAVE, i=count, j=count)
                    # print "_______________________________________________________"
                    # print row_1
                    # print row_2
                    # print(directions_result[0]['legs'][0]['distance']['text'])
                    # print distance_meters, "meters"
                    # print(directions_result[0]['legs'][0]['duration']['text'])
                    # print duration_seconds, "seconds"

                    # raw_input()

                except Exception as ex:
                    print ex
                    continue

        self.saveGraph(file_path=PUBLIC_TRANSIT_GRAPH_PATH_SAVE, i=count, j=count)


    def readGoogleMapsAndCreateGraph(self, date_time = 'Dec 12 2018  5:00PM'):
        """
        Use Google API to compute travel time between the points for each time of day
        (using public transportation)
        """
        gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

        now = datetime.now()
        # now = datetime.strptime(date_time, '%b %d %Y %I:%M%p')

        edge_id=0
        count = 0
        # N = self.data.shape[0]
        for i, row_1 in self.data.iterrows():
            for j, row_2 in self.data.iterrows():
                if i > j: #  and (i*N + j > 303*N + 246):
                    # print i, j
                    count+=1
                    try :
                        directions_result = gmaps.directions(row_1.lat_long,
                                                             row_2.lat_long,
                                                             mode="transit",
                                                             departure_time=now)
                        distance_meters = directions_result[0]['legs'][0]['distance']['value']
                        duration_seconds = directions_result[0]['legs'][0]['duration']['value']

                        self.addNodesAndEdgesPublicTransitGraph(row_1.id, row_2.id, edge_id, distance_meters, duration_seconds)
                        edge_id+=2

                        if (count % 1000) == 0:
                            self.saveGraph(file_path=PUBLIC_TRANSIT_GRAPH_PATH_SAVE, i=count, j=count)

                        # print "_______________________________________________________"
                        # print(directions_result[0]['legs'][0]['distance']['text'])
                        # print distance_meters, "meters"
                        # print(directions_result[0]['legs'][0]['duration']['text'])
                        # print duration_seconds, "seconds"

                        # raw_input()

                    except Exception as ex:
                        print ex
                        continue

        self.saveGraph(file_path=PUBLIC_TRANSIT_GRAPH_PATH_SAVE, i=count, j=count)

    def CreateGraphFromSavedData(self, unpickled_df, to_print=True, graph_file=PUBLIC_TRANSIT_PLUS_INTER_GRAPH_PATH_SAVE):
        """
        Compute Graph based on intermediate stesps from the Google Response.
        """
        edge_id=0
        self.count_dup = 0
        for i, row in unpickled_df.iterrows():
            try :
                directions = row.response[0]["legs"][0]
                edge_id = self.intermediateLocations(directions, edge_id)
            except Exception as ex:
                #print('Fail')
                pass
                # print row.response
                # print ex
            # if (i == 10):
            # break
        if to_print:
            print "Number of duplicate edges", self.count_dup
            print "The total number of Nodes is",  self.graph.GetNodes()
            print "The total number of Edges is",  self.graph.GetEdges()

        self.saveGraph(file_path=graph_file)
        d = {'node_id': self.hash_to_node.values(), 'hash_name': self.hash_to_node.keys()}
        df = pd.DataFrame(data = d)
        df.to_csv("hash_node.csv")

    def intermediateLocations(self, directions, edge_id):
        """
        Gives the recursive directions based on some google response
        """
        # print("________________________")
        # print("NEW ADDRESS")
        # print("________________________")
        locations = [direction['start_location'] for direction in directions["steps"]]
        locations.append(directions["steps"][-1]["end_location"])
        for i in range(len(locations) - 1):
            direction = directions["steps"][i]
            distance_meters = direction['distance']['value']
            duration_seconds = direction['duration']['value']
            travel_mode = direction['travel_mode']
            # print "distance", distance_meters
            # print "durantion", duration_seconds
            # print "travel mode", travel_mode
            # print "start location", start_location
            # print "end location", end_location
            # print("________________________")
            self.addNodesAndEdgesPublicTransitGraph(locations[i], locations[i+1], edge_id, distance_meters,
                    duration_seconds, travel_mode=travel_mode, building_lat_long=True)
            edge_id+=2
        return edge_id

    def addNodesAndEdgesPublicTransitGraph(self, zone1_id, zone2_id, edge_id, distance_meters, duration_seconds, travel_mode=None, building_lat_long=False, to_print=False):
        # Add nodes
        if building_lat_long:
            zone1_id_key = geohash.encode(round(zone1_id[u'lat'], 3), round(zone1_id[u'lng'], 3))
            zone2_id_key = geohash.encode(round(zone2_id[u'lat'], 3), round(zone2_id[u'lng'], 3))
            if zone1_id_key not in self.hash_to_node.keys():
                self.hash_to_node[zone1_id_key] = self.node_count_when_builidng
                self.node_to_hash[str(self.node_count_when_builidng)] = zone1_id_key
                self.node_count_when_builidng += 1
            if zone2_id_key not in self.hash_to_node.keys():
                self.hash_to_node[zone2_id_key] = self.node_count_when_builidng
                self.node_to_hash[str(self.node_count_when_builidng)] = zone2_id_key
                self.node_count_when_builidng += 1
            zone1_id = self.hash_to_node[zone1_id_key]
            zone2_id = self.hash_to_node[zone2_id_key]

        if not self.graph.IsNode(zone1_id) : self.graph.AddNode(zone1_id)
        if not self.graph.IsNode(zone2_id) : self.graph.AddNode(zone2_id)
        if not self.graph.IsEdge(zone1_id, zone2_id) and not self.graph.IsEdge(zone2_id, zone1_id):
            self.graph.AddEdge(zone1_id, zone2_id, edge_id)
            self.graph.AddEdge(zone2_id, zone1_id, edge_id+1)
            self.graph.AddFltAttrDatE(edge_id, distance_meters, 'distance_meters')
            self.graph.AddFltAttrDatE(edge_id, duration_seconds, 'duration_seconds')
            self.graph.AddFltAttrDatE(edge_id+1, distance_meters, 'distance_meters')
            self.graph.AddFltAttrDatE(edge_id+1, duration_seconds, 'duration_seconds')
            # Initialize 'weight' to be 1
            self.graph.AddIntAttrDatE(edge_id, 1, 'weight')
            self.graph.AddIntAttrDatE(edge_id+1, 1, 'weight')
            if travel_mode is not None:
                self.graph.AddStrAttrDatE(edge_id, travel_mode , 'travel_mode')
                self.graph.AddStrAttrDatE(edge_id+1, travel_mode, 'travel_mode')
        else:
            # Increment duplicate count
            self.count_dup += 1
            # Add to weight of edge(s)
            if self.graph.IsEdge(zone1_id, zone2_id):
                edge_id = self.graph.GetEI(zone1_id, zone2_id)
                prev_weight = self.graph.GetIntAttrDatE(edge_id, 'weight')
                self.graph.AddIntAttrDatE(edge_id, prev_weight+1, 'weight')
            if self.graph.IsEdge(zone2_id, zone1_id):
                edge_id = self.graph.GetEI(zone2_id, zone1_id)
                prev_weight = self.graph.GetIntAttrDatE(edge_id, 'weight')
                self.graph.AddIntAttrDatE(edge_id, prev_weight+1, 'weight')

        if to_print:
            # print "here"
            num_edges = self.graph.GetEdges()
            num_nodes = self.graph.GetNodes()
            # print num_edges
            # print num_nodes
            # Print some properties of the graph
            print('Number of nodes (zones): {}'.format(num_nodes))
            print('Number of edges (zone borders): {}'.format(num_edges))


    def draw_map(self, filename, plot_edges=False, plot_nodes=False, node_scaling='degree', centrality=None):

        ###################################################
        # Always the same
        ###################################################
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
        # Plot zones
        patches = []
        for idx, p in enumerate(polys): patches.append(PolygonPatch(p, fc='#AEEDFF', ec='#555555', alpha=1., zorder=1))
        ax.add_collection(PatchCollection(patches, match_original=True))

        ###################################################
        # Plot edges
        ###################################################
        if plot_edges:
            for i, edge in tqdm(enumerate(self.graph.Edges())):
                start = geohash.decode(self.node_to_hash[str(edge.GetSrcNId())])
                end = geohash.decode(self.node_to_hash[str(edge.GetDstNId())])
                ax.plot([start[1], end[1]], [start[0], end[0]], color='g', linewidth='1')

        ###################################################
        # Plot nodes
        ###################################################
        if plot_nodes:
            lats, longs, degrees = [], [], []
            for node in self.node_to_hash:
                # Get lat and long of node
                long, lat = geohash.decode(self.node_to_hash[node])
                lats.append(lat)
                longs.append(long)
                # Use node degree
                if node_scaling == 'degree':
                    try: degree = self.graph.GetNI(int(node)).GetDeg()
                    except: degree = 0
                    degrees.append(degree)
                # Get weighted node degree
                elif node_scaling in ['degree_weighted_out', 'degree_weighted_in', 'degree_weighted_both']:
                    try: 
                        node, degree = self.graph.GetNI(int(node)), 0
                        # Based on out edges
                        if node_scaling in ['degree_weighted_out', 'degree_weighted_both']:
                            for i in range(node.GetOutDeg()):
                                neighbor_id = node.GetOutNId(i)
                                edge_id = self.graph.GetEI(node.GetId(), neighbor_id).GetId()
                                weight = self.graph.GetFltAttrDatE(edge_id, 'weight')
                                if weight > 0: degree += weight
                        # Based on in edges
                        elif node_scaling in ['degree_weighted_in', 'degree_weighted_both']:
                            for i in range(node.GetInDeg()):
                                neighbor_id = node.GetInNId(i)
                                edge_id = self.graph.GetEI(node.GetId(), neighbor_id).GetId()
                                weight = self.graph.GetFltAttrDatE(edge_id, 'weight')
                                if weight > 0: degree += weight   
                    except: 
                        degree = 0
                    degrees.append(degree)
            # If doing node centrality
            if node_scaling == 'centrality': degrees = [float(x) for x in centrality]
            # Scale degrees so that dots are properly sized
            max_degree = float(max(degrees))
            scales = [float(x)/max_degree*75 for x in degrees]
            vis = ax.scatter(lats, longs, s=scales, c=degrees, cmap=plt.cm.get_cmap('plasma'))
            fig.colorbar(vis)

        ###################################################
        # Always the same
        ###################################################
        ax.set_yticks([])
        plt.savefig(filename, alpha=True, dpi=300)


if __name__ == "__main__":
    #public_transport = PublicTransport(create_new=False, read_google_maps=True, plot_graph=False, check_attributes=False, reduce_graph=False)

    # metrics.plotDegreeDistribution(original_graph=public_transport.graph, attribute="duration_seconds", type_graph="public_transit")
    # metrics.plotDegreeDistribution(original_graph=public_transport.graph, attribute="distance_meters", type_graph="public_transit")
    # metrics.compute_centrality(public_transport.graph, graph_type="public_transit")
    # metrics.find_node_roles(public_transport.graph, attributes=["duration_seconds", "distance_meters"], graph_type="public_transit")

    # uber_graph = metrics.load_graph()
    # metrics.plotDegreeDistribution(original_graph=uber_graph, attribute="travel_time_17", type_graph="uber")
    # metrics.plotDegreeDistribution(original_graph=uber_graph, attribute="travel_speed_17", type_graph="uber")

    # unpickled_df = pd.read_pickle("./dummy.pkl")

    #unpickled_df = pd.read_pickle("Data/ExtraPublicTransit/google_response_data.pkl")
    # unpickled_df.to_csv("Data/ExtraPublicTransit/google_respose.csv")
    #public_transport = PublicTransport(create_new=True, read_google_maps=False, plot_graph=False, check_attributes=False, reduce_graph=False)
    #public_transport.CreateGraphFromSavedData(unpickled_df=unpickled_df)
    #public_transport.draw_map("public_transport_plus_intermediate.png")

    # Load graph
    graph_file = PUBLIC_TRANSIT_GRAPH_WEIGHTED_PATH_LOAD
    public_transport = PublicTransport(create_new=False, read_google_maps=False, plot_graph=False, \
                                        check_attributes=False, reduce_graph=False, graph_path=graph_file)
    
    # Plot all edges
    if False:
        public_transport.draw_map("Plots/public_transport_plus_intermediate_all_edges.png", plot_edges=True)

    # Make new graph that has weights on edges
    if False:
        unpickled_df = pd.read_pickle("Data/ExtraPublicTransit/google_response_data.pkl")
        public_transport = PublicTransport(create_new=True, read_google_maps=False, plot_graph=False, check_attributes=False, reduce_graph=False)
        public_transport.CreateGraphFromSavedData(unpickled_df=unpickled_df, graph_file=PUBLIC_TRANSIT_PLUS_INTER_GRAPH_WEIGHTED_PATH_SAVE)

    # Plot all nodes
    if False:
        public_transport.draw_map("Plots/public_transport_plus_intermediate_all_nodes_degree.png", plot_nodes=True, node_scaling='degree')
        #public_transport.draw_map("Plots/public_transport_plus_intermediate_all_nodes_degree_weighted_in.png", plot_nodes=True, node_scaling='degree_weighted_in')
        #public_transport.draw_map("Plots/public_transport_plus_intermediate_all_nodes_degree_weighted_out.png", plot_nodes=True, node_scaling='degree_weighted_out')
        public_transport.draw_map("Plots/public_transport_plus_intermediate_all_nodes_degree_weighted_both.png", plot_nodes=True, node_scaling='degree_weighted_both')

    # Compute and plot node centrality
    if True:
        centrality = compute_centrality(public_transport.graph, graph_type='weighted_inter', node_to_hash=public_transport.node_to_hash)
        public_transport.draw_map("Plots/public_transport_plus_intermediate_all_nodes_centrality.png", plot_nodes=True, node_scaling='centrality', centrality=centrality)




