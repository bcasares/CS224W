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
import more_metrics
from util import get_edge_weight_distribution
import math

import fiona
from shapely.geometry import mapping, shape, Polygon, MultiPolygon, Point
PROCESSED_GEO_PATH = 'Data/Geo/sf_geoboundaries.shp'
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch

import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from scipy.stats import wasserstein_distance

# GOOGLE_API_KEY = "AIzaSyAi5ERQfBOcdO54IT3MOmiCnNubFB1RWWY" # First Key, no-credit
# GOOGLE_API_KEY = "AIzaSyCnzf2UnjffaNCRX_rGvlypYBZb5rt6zyM" #abecs224w1 no-credit
# GOOGLE_API_KEY = "AIzaSyCUHPr2e0NmO5lwrB5cHuVUCDVeSDYWDQ0" #abecs224w2 no-credit
# GOOGLE_API_KEY = "AIzaSyD7_s0eA-M967PpPLQfu9sKwnkiNVSHoE8" #abecs224w3 still works
# GOOGLE_API_KEY = "AIzaSyAqAw4fAIRPP2eavNrNhBSn-f0r1PuDNCc" #abecs224w4


SF_ZONE_INFO = os.path.join("Data", "Geo", "sf_zone_info.csv")
SF_REDUCED_GRAPH = os.path.join("Data", "Geo", "Graphs", "sf_uber_final_graph.graph")
PUBLIC_TRANSIT_GRAPH_PATH_SAVE = os.path.join("Data", "Geo", "Graphs", "public_transit.graph")
PUBLIC_TRANSIT_PLUS_INTER_GRAPH_PATH_SAVE = os.path.join("Data", "Geo", "Graphs", "public_transit_plus_inter.graph")

# PUBLIC_TRANSIT_GRAPH_PATH_LOAD = os.path.join("Data", "Geo","Graphs",  "public_transit_complete.graph")
# PUBLIC_TRANSIT_GRAPH_PATH_LOAD = os.path.join("Data", "Geo", "Graphs", "public_transit_reduced5pm.graph")
PUBLIC_TRANSIT_GRAPH_PATH_LOAD_INTERMEDIATE = os.path.join("Data", "Geo", "Graphs", "public_transit_plus_inter.graph")
PUBLIC_TRANSIT_GRAPH_PATH_LOAD_TRANSIT = os.path.join("Data", "Geo", "Graphs", "transit_system.graph")
PUBLIC_TRANSIT_GRAPH_PATH_LOAD_WALKING = os.path.join("Data", "Geo", "Graphs", "walking.graph")
PUBLIC_TRANSIT_GRAPH_PATH_LOAD_WALKING_SCC = os.path.join("Data", "Geo", "Graphs", "walking_scc.graph")

PUBLIC_TRANSIT_GRAPH_PATH_LOAD_TRANSIT_WEIGHTED = os.path.join("Data", "Geo", "Graphs", "weighted_transit_system.graph")
PUBLIC_TRANSIT_GRAPH_PATH_LOAD_WALKING_WEIGHTED = os.path.join("Data", "Geo", "Graphs", "weighted_walking.graph")

PUBLIC_TRANSIT_PLUS_INTER_GRAPH_WEIGHTED_PATH_SAVE = os.path.join("Data", "Geo", "Graphs", "public_transit_plus_inter_weighted.graph")
PUBLIC_TRANSIT_GRAPH_WEIGHTED_PATH_LOAD = os.path.join("Data", "Geo", "Graphs", "public_transit_plus_inter_weighted.graph")

PUBLIC_TRANSIT_GRAPH_PLOT = os.path.join("Data", "Geo", "Images", "public_transit_graph_5pm.png")
# PUBLIC_TRANSIT_GRAPH_PLOT = os.path.join("Data", "Geo", "Images", "public_transit_graph_reduced.png")

PUBLIC_TRANSIT_GRAPH_PATH_LOAD_ALL_WEIGHTED_10PLUS = os.path.join("Data", "Geo", "Graphs", "weighted_public_transit_all_gt10edgeweight.graph")
PUBLIC_TRANSIT_GRAPH_PATH_LOAD_WALKING_WEIGHTED_10PLUS = os.path.join("Data", "Geo", "Graphs", "weighted_public_transit_onlywalking_gt10edgeweight.graph")
PUBLIC_TRANSIT_GRAPH_PATH_LOAD_TRANSIT_WEIGHTED_10PLUS = os.path.join("Data", "Geo", "Graphs", "weighted_public_transit_onlytransit_gt10edgeweight.graph")

CLUSTER_COLORS = ['r', 'g', 'b', 'c', 'm']

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

    def loadGraph(self, file_path, create_new=False, to_print=True):
        """
        Loads the graph if the graph already exists
        """
        # Load graph
        if create_new:
            self.graph = snap.TNEANet.New()
        else:
            FIn = snap.TFIn(file_path)
            self.graph = snap.TNEANet.Load(FIn)
            self.getNodeToHashHashToNode()
            if to_print:
                print "Number of nodes", self.graph.GetNodes()
                print "Number of edges", self.graph.GetEdges()

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
        if False:
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

                        if (count % 1000) == 0 :
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

       # self.saveGraph(file_path=graph_file)
       #  d = {'node_id': self.hash_to_node.values(), 'hash_name': self.hash_to_node.keys()}
       #  df = pd.DataFrame(data = d)
       #  df.to_csv("hash_node.csv")

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

    def createSubGraphs(self, to_print = True, weighted=False):
        """
        Build two subgraphs based on the graph plus intermediate steps

        There are two graphs, either 'WALKING', 'TRANSIT'.
        """

        walking_graph = snap.TNEANet.New()
        transit_system_graph = snap.TNEANet.New()
        edge_id_walking = 0
        edge_id_transit_system = 0
        for edge in self.graph.Edges():
            node1 = edge.GetSrcNId()
            node2 = edge.GetDstNId()
            distance_meters = self.graph.GetFltAttrDatE(edge.GetId(), "distance_meters")
            duration_seconds = self.graph.GetFltAttrDatE(edge.GetId(), "duration_seconds")
            travel_mode = self.graph.GetStrAttrDatE(edge.GetId(), "travel_mode")
            if weighted:
                edge_weight = self.graph.GetIntAttrDatE(edge.GetId(), 'weight')
            else:
                edge_weight = None
            if travel_mode == "WALKING":
                addNodesAndEdges(walking_graph, node1, node2, edge_id_walking, distance_meters, duration_seconds,
                        travel_mode, edge_weight=edge_weight, to_print=False)
                edge_id_walking += 2
            else: # Travel mode is 'TRANSTI'
                addNodesAndEdges(transit_system_graph, node1, node2, edge_id_transit_system, distance_meters, duration_seconds,
                        travel_mode, edge_weight=edge_weight, to_print=False)
                edge_id_transit_system += 2
        if to_print:
            # Walking
            num_edges = walking_graph.GetEdges()
            num_nodes = walking_graph.GetNodes()
            print('number of nodes for the walking graph is: {}'.format(num_nodes))
            print('number of edges for the walking graph is: {}'.format(num_edges))

            # Transit
            num_edges = transit_system_graph.GetEdges()
            num_nodes = transit_system_graph.GetNodes()
            print('number of nodes for the transit system graph is: {}'.format(num_nodes))
            print('number of edges for the transit system graph is: {}'.format(num_edges))


        walking_name = os.path.join("Data", "Geo", "Graphs", "weighted_walking.graph")
        transit_name = os.path.join("Data", "Geo", "Graphs", "weighted_transit_system.graph")
        saveGraph(walking_graph, walking_name)
        saveGraph(transit_system_graph, transit_name)

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
            num_edges = self.graph.GetEdges()
            num_nodes = self.graph.GetNodes()
            print('Number of nodes (zones): {}'.format(num_nodes))
            print('Number of edges (zone borders): {}'.format(num_edges))


    def draw_map(self, filename, plot_edges=False, edge_weight_threshold=None, edge_scaling=None, plot_nodes=False, node_scaling='degree', centrality=None, classification=None, new_fig=True, last=True):

        ###################################################
        # Always the same
        ###################################################
        # Extract polygons
        if new_fig:
            polys = MultiPolygon([shape(zone['geometry']) for zone in fiona.open(PROCESSED_GEO_PATH)])
            # Setup plot
            self.fig = plt.figure(figsize=(15, 20))
            self.ax = self.fig.add_subplot(111)
            ax = self.ax
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
            # if plot_edges:
            #     for i, edge in tqdm(enumerate(self.graph.Edges())):
            #         start = geohash.decode(self.node_to_hash[str(edge.GetSrcNId())])
            #         end = geohash.decode(self.node_to_hash[str(edge.GetDstNId())])
            #         ax.plot([start[1], end[1]], [start[0], end[0]], color='g', linewidth='1')

        ###################################################
        # Plot edges
        ###################################################
        if plot_edges:
            X, Y, scaling = [], [], []
            for i, edge in tqdm(enumerate(self.graph.Edges())):
                start = geohash.decode(self.node_to_hash[str(edge.GetSrcNId())])
                end = geohash.decode(self.node_to_hash[str(edge.GetDstNId())])
                # Determine if edge should be added
                add = True
                if edge_weight_threshold:
                    if self.graph.GetIntAttrDatE(edge.GetId(), 'weight') <= edge_weight_threshold:
                        add = False
                # If edge should be added
                if add == True:
                    X.append((start[1], end[1]))
                    Y.append((start[0], end[0]))
                    if edge_scaling == 'weight':
                        scaling.append(self.graph.GetIntAttrDatE(edge.GetId(), edge_scaling))
                    else:
                        scaling.append(self.graph.GetFltAttrDatE(edge.GetId(), edge_scaling))
            # Plot
            if not edge_scaling:
                ax.plot(X, Y, color='g', linewidth='1')
            else:
                scaling = [math.log(x) if not int(x) == 0 else 0 for x in scaling] # Need to use log scaling when doing weights
                cmap = plt.get_cmap('YlOrRd')
                cNorm  = colors.Normalize(vmin=min(scaling), vmax=max(scaling))
                print('min: %f, max: %f' % (min(scaling), max(scaling)))
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
                print scalarMap.get_clim()
                c = [scalarMap.to_rgba(x) for x in scaling]
                #print(c)
                for i, x in enumerate(X):
                    ax.plot(X[i], Y[i], color=c[i])

        ###################################################
        # Plot nodes
        ###################################################
        ax = self.ax
        if plot_nodes:
            lats, longs, degrees = [], [], []
            for node in self.graph.Nodes():
                node_id = str(node.GetId())
                # Get lat and long of node
                long, lat = geohash.decode(self.node_to_hash[node_id])
                lats.append(lat)
                longs.append(long)
                # Use node degree
                if node_scaling == 'degree':
                    try: degree = self.graph.GetNI(int(node_id)).GetDeg()
                    except: degree = 0
                # Use weighted node degree
                elif node_scaling in ['degree_weighted_out', 'degree_weighted_in', 'degree_weighted_both']:
                    try:
                        node, degree = self.graph.GetNI(int(node)), 0
                        # Based on out edges
                        if node_scaling in ['degree_weighted_out', 'degree_weighted_both']:
                            for i in range(node.GetOutDeg()):
                                neighbor_id = node.GetOutNId(i)
                                edge_id = self.graph.GetEI(node.GetId(), neighbor_id).GetId()
                                weight = self.graph.GetIntAttrDatE(edge_id, 'weight')
                                if weight > 0: degree += weight
                        # Based on in edges
                        elif node_scaling in ['degree_weighted_in', 'degree_weighted_both']:
                            for i in range(node.GetInDeg()):
                                neighbor_id = node.GetInNId(i)
                                edge_id = self.graph.GetEI(node.GetId(), neighbor_id).GetId()
                                weight = self.graph.GetIntAttrDatE(edge_id, 'weight')
                                if weight > 0: degree += weight
                    except:
                        degree = 0
                # Use node centrality
                elif node_scaling == 'centrality':
                    try: degree = centrality[int(node_id)]
                    except: degree = 0
                # Use node classification
                elif node_scaling == 'classification':
                    try: degree = classification[int(node_id)]
                    except: degree = 0
                # Append to degrees list
                degrees.append(degree)

            if not node_scaling == 'classification':
                # Scale degrees so that dots are properly sized
                max_degree = float(max(degrees))
                scales = [float(x)/max_degree*75 for x in degrees]
                vis = ax.scatter(lats, longs, s=scales, c=degrees, cmap=plt.cm.get_cmap('plasma'))
                fig.colorbar(vis)
            else:
                vis = ax.scatter(lats, longs)
                degrees_set = set(degrees)
                for i, lat in enumerate(lats):
                    ax.scatter(lats[i], longs[i], c=CLUSTER_COLORS[degrees[i]], s=10)

        ###################################################
        # Always the same
        ###################################################
        ax.set_yticks([])
        if last:
            plt.savefig(filename, alpha=True, dpi=300)
            # plt.show()

def addNodesAndEdges(graph, node1, node2, edge_id, distance_meters, duration_seconds, travel_mode, edge_weight=None, to_print=False):
    if not graph.IsNode(node1) : graph.AddNode(node1)
    if not graph.IsNode(node2) : graph.AddNode(node2)
    if not graph.IsEdge(node1, node2) and not graph.IsEdge(node2, node1):
        graph.AddEdge(node1, node2, edge_id)
        graph.AddEdge(node2, node1, edge_id+1)
        graph.AddFltAttrDatE(edge_id, distance_meters, 'distance_meters')
        graph.AddFltAttrDatE(edge_id, duration_seconds, 'duration_seconds')
        graph.AddFltAttrDatE(edge_id+1, distance_meters, 'distance_meters')
        graph.AddFltAttrDatE(edge_id+1, duration_seconds, 'duration_seconds')
        if edge_weight is not None:
            graph.AddIntAttrDatE(edge_id, edge_weight, 'weight')
            graph.AddIntAttrDatE(edge_id+1, edge_weight, 'weight')
        if travel_mode is not None:
            graph.AddStrAttrDatE(edge_id, travel_mode , 'travel_mode')
            graph.AddStrAttrDatE(edge_id+1, travel_mode, 'travel_mode')

def saveGraph(graph, file_path):
    """
    Save graph
    """
    FOut = snap.TFOut(file_path)
    graph.Save(FOut)

def find_clusters(X, n_clusters, rseed=2):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]
    #print('max: %f, min: %f' % (np.min(X), np.max(X)))
    #print np.any(np.isnan(X))
    #print np.any(np.isinf(X))
    while True:
        #print(centers)
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers, metric=wasserstein_distance)
        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
        # 2c. Check for convergence
        if np.all(centers == new_centers): break
        centers = new_centers
        # Break if values are invalid
        if np.any(np.isnan(centers)) or np.any(np.isinf(centers)): break
    return centers, labels

def find_node_roles(graph, attribute='weight'):
    # Build dictionary mapping node_ids to adjacent edge weights
    edgeWeights = defaultdict(list)
    # Loop through all nodes, add attribute to each that is the sum of all adjacent edge weights
    for node in graph.Nodes():
        node_id, num_out_nodes = node.GetId(), node.GetOutDeg()
        for i in range(num_out_nodes):
            neighbor_id = node.GetOutNId(i)
            edge_id = graph.GetEI(node_id, neighbor_id).GetId()
            if attribute == 'weight':
                weight = graph.GetIntAttrDatE(edge_id, attribute)
                if weight > 0: edgeWeights[node_id].append(math.log(weight))
            else:
                weight = graph.GetFltAttrDatE(edge_id, attribute)
                if weight > 0: edgeWeights[node_id].append(weight)
    # Convert each node array of edges to histogram; Find global min and max values of weights
    minWeight = min([min(weights) for node, weights in edgeWeights.iteritems()])
    maxWeight = max([max(weights) for node, weights in edgeWeights.iteritems()])
    # Build histograms
    edgeWeightHistograms = {}
    for node, weights in edgeWeights.iteritems():
        histo = np.histogram(weights, bins=13, range=(minWeight, maxWeight))
        edgeWeightHistograms[node] = list(histo[0])
    # Finish
    nodes, histograms = zip(*edgeWeightHistograms.iteritems())
    centers, labels = find_clusters(np.array(histograms), 5)
    node_roles = dict(zip(nodes, labels))
    # Return
    return node_roles

if __name__ == "__main__":
    #public_transport = PublicTransport(create_new=False, read_google_maps=True, plot_graph=False, check_attributes=False, reduce_graph=False)

    # metrics.plotDegreeDistribution(original_graph=public_transport.graph, attribute="duration_seconds", type_graph="public_transit")
    # metrics.plotDegreeDistribution(original_graph=public_transport.graph, attribute="distance_meters", type_graph="public_transit")
    # metrics.compute_centrality(public_transport.graph, graph_type="public_transit")
    # metrics.find_node_roles(public_transport.graph, attributes=["duration_seconds", "distance_meters"], graph_type="public_transit")

    # uber_graph = metrics.load_graph()
    # metrics.plotDegreeDistribution(original_graph=uber_graph, attribute="travel_time_17", type_graph="uber")
    # metrics.plotDegreeDistribution(original_graph=uber_graph, attribute="travel_speed_17", type_graph="uber")

    #unpickled_df = pd.read_pickle("Data/ExtraPublicTransit/google_response_data.pkl")
    # unpickled_df.to_csv("Data/ExtraPublicTransit/google_respose.csv")
    # public_transport = PublicTransport(create_new=True, read_google_maps=False, plot_graph=False, check_attributes=False, reduce_graph=False)
    # public_transport.CreateGraphFromSavedData(unpickled_df=unpickled_df)
    # public_transport.draw_map("public_transport_plus_intermediate.png")

    if False:
        public_transport = PublicTransport(create_new=False, read_google_maps=False, plot_graph=False, check_attributes=False, reduce_graph=False,
                graph_path=PUBLIC_TRANSIT_GRAPH_PATH_LOAD_INTERMEDIATE)
        public_transport.saveGraphToCSVPredictionAnalysis()



    # Create Subraphs based on intermediate
    #if False:
        #public_transport = PublicTransport(create_new=False, read_google_maps=False, plot_graph=False, check_attributes=False, reduce_graph=False,
        #        graph_path=PUBLIC_TRANSIT_GRAPH_PATH_LOAD_INTERMEDIATE)
        #public_transport.createSubGraphs(to_print = True)
        # public_transport.draw_map("public_transport_plus_intermediate.png")

    #####################################
    # LOAD GRAPH
    #####################################

    # CHOOSE GRAPH TO LOAD
    # graph_file = PUBLIC_TRANSIT_GRAPH_WEIGHTED_PATH_LOAD # Weighted, both transit and walking
    # graph_file = PUBLIC_TRANSIT_GRAPH_PATH_LOAD_WALKING # Unweighted, only walking
    # graph_file = PUBLIC_TRANSIT_GRAPH_PATH_LOAD_WALKING_SCC # Unweighted, scc of only walking
    # graph_file = PUBLIC_TRANSIT_GRAPH_PATH_LOAD_TRANSIT # Unweighted, only transit
    # graph_file = PUBLIC_TRANSIT_GRAPH_PATH_LOAD_WALKING_WEIGHTED # Weighted, only walking
    # graph_file = PUBLIC_TRANSIT_GRAPH_PATH_LOAD_TRANSIT_WEIGHTED # Weighted, only transit

    # LOAD
    all_intermediate = PublicTransport(graph_path=PUBLIC_TRANSIT_GRAPH_PATH_LOAD_INTERMEDIATE)
    walking = PublicTransport(graph_path=PUBLIC_TRANSIT_GRAPH_PATH_LOAD_WALKING)
    transit_systems = PublicTransport(graph_path=PUBLIC_TRANSIT_GRAPH_PATH_LOAD_TRANSIT)
    # walking_scc = PublicTransport(graph_path=PUBLIC_TRANSIT_GRAPH_PATH_LOAD_WALKING_SCC)

    #####################################
    # COMPUTING METRICS
    #####################################
    # Graphs = [all_intermediate.graph, walking.graph, transit_systems.graph]
    # names=["all_intermediata", "walking", "transit_systems"]

    if False:
        Graph = walking.graph
        Components = snap.TCnComV()
        snap.GetSccs(Graph, Components)
        count = 0
        for i, CnCom in tqdm(enumerate(Components)):
            NIdV = snap.TIntV()
            for e in CnCom:
                NIdV.Add(e)
            all_intermediate.graph = snap.GetSubGraph(walking.graph, NIdV)
            num_nodes = all_intermediate.graph.GetNodes()
            if num_nodes > 25 :
                if count == 0:
                    all_intermediate.draw_map("", plot_edges=False, plot_nodes=True,
                        node_scaling=None, last=False)
                if count == 4:
                    all_intermediate.draw_map("Plots/sscc_walking_nodes_top_5.png", plot_edges=False, plot_nodes=True,
                        node_scaling=None, new_fig=False, last=True)
                else:
                    all_intermediate.draw_map("", plot_edges=False, plot_nodes=True,
                        node_scaling=None, new_fig=False, last=False)
                count+=1

        # Draw Nodes based on Node2Vec on the graph
    if False:
        random = [3180,3033,3967,2730,963] # P = 1, q = 1
        structural_similarity = [2721,3033,3071,5137,4881] # p 0.001 q = 1000
        community_detection = [3180,3826,3967,4855,3619] # p 1000 q = 0.1
        all_node_to_vec = [random, structural_similarity, community_detection]

        for i, elem in enumerate(all_node_to_vec):
            # graph = snap.TNEANet.New()
            NIdV = snap.TIntV()
            for e in elem:
                NIdV.Add(e)
            walking.graph = snap.GetSubGraph(all_intermediate.graph, NIdV)
            num_nodes = walking.graph.GetNodes()
            print num_nodes
            if i == 0:
                walking.draw_map("", plot_edges=False, plot_nodes=True,
                    node_scaling=None, last=False)
            if i == 1:
                walking.draw_map("", plot_edges=False, plot_nodes=True,
                    node_scaling=None, new_fig=False, last=False)
            if i == 2:
                walking.draw_map("Plots/node_to_vec.png", plot_edges=False, plot_nodes=True,
                    node_scaling=None, new_fig=False, last=True)

    # Get edge list to run node2vec
    if False:
        with open('node2vec/graph/intermediate.edgelist', 'w') as f:
            for edge in all_intermediate.graph.Edges():
                print >> f, edge.GetSrcNId(),edge.GetDstNId()

    # Get Node with highest degree.
    if False:
        NId1 = snap.GetMxDegNId(all_intermediate.graph)
        print NId1


    # Calculate degree distribution
    if False:
        # more_metrics.degreeDistribution(Graphs=Graphs, names=names)
        pass

    # Calculate  SCC, WCC, etc
    if False:
        for G, name in zip(Graphs, names):
            scc = more_metrics.calculateBowTieStructure(G, name = name)
            # if name =="walking":
            #     print "saving"
            #     FOut = snap.TFOut(PUBLIC_TRANSIT_GRAPH_PATH_LOAD_WALKING_SCC)
            #     scc.Save(FOut)

    # Calulate degree count
    if False:
        for G, name in zip(Graphs, names):
            more_metrics.degreeCount(G, name = name)

    # Plot node reachability
    if False:
        more_metrics.calNodeReachability(walking.graph, name="walking")

    # Plot scc for walking
    if False:
        walking_scc.draw_map("Plots/scc_walking_nodes.png", plot_edges=False, plot_nodes=True,
                node_scaling=None)


    # Walking Graph Weighted
    if False:
        public_transport = PublicTransport(create_new=False, read_google_maps=False, plot_graph=False, \
                check_attributes=False, reduce_graph=False, graph_path=PUBLIC_TRANSIT_GRAPH_PATH_LOAD_WALKING_WEIGHTED)
        public_transport.draw_map("Plots/public_transport_walking_weighted_nodes.png", plot_edges=False, plot_nodes=True,
                node_scaling='degree_weighted_both')
        # WEIGHTED GRAPH WITH ALL TRANSPORT
        if False:
            unpickled_df = pd.read_pickle("Data/ExtraPublicTransit/google_response_data.pkl")
            public_transport = PublicTransport(create_new=True, read_google_maps=False, plot_graph=False, check_attributes=False, reduce_graph=False)
            public_transport.CreateGraphFromSavedData(unpickled_df=unpickled_df, graph_file=PUBLIC_TRANSIT_PLUS_INTER_GRAPH_WEIGHTED_PATH_SAVE)

        # WEIGHTED TRANSIT AND WALKING GRAPHS
        if False:
            public_transport.createSubGraphs(to_print=True, weighted=True)

    # Transit system Graph Weighted
    if False:
        public_transport = PublicTransport(create_new=False, read_google_maps=False, plot_graph=False, \
                check_attributes=False, reduce_graph=False, graph_path=PUBLIC_TRANSIT_GRAPH_PATH_LOAD_TRANSIT_WEIGHTED)
        public_transport.draw_map("Plots/public_transport_transit_systems_weighted_nodes.png", plot_edges=False, plot_nodes=True,
                node_scaling='degree_weighted_both')

    # Plot all edges
    if False:
        public_transport.draw_map("Plots/public_transport_plus_intermediate_all_edges.png", plot_edges=True)
        # WEIGHTED, ONLY EDGES WITH WEIGHT > 10
        if True:
            graph = public_transport.graph
            print('Original edges: %d' % graph.GetEdges())
            for edge in graph.Edges():
                edge_id = edge.GetId()
                weight = graph.GetIntAttrDatE(edge_id, 'weight')
                if weight <= 10:
                    start = edge.GetSrcNId()
                    end = edge.GetDstNId()
                    graph.DelEdge(start, end)
            print('New edges: %d' % graph.GetEdges())
            filename = os.path.join("Data", "Geo", "Graphs", "weighted_public_transit_onlytransit_gt10edgeweight.graph")
            saveGraph(graph, filename)

    #####################################
    # PLOTTING
    #####################################
    if False:
        # CHOOSE IMAGE FILE TO SAVE
        #plot_file = "Plots/public_transport_all_weighted_nodes.png" # Weighted, both transit and walking
        #plot_file = "Plots/public_transport_walking.png" # Unweighted, only walking
        #plot_file = "Plots/public_transport_transit_systems.png" # Unweighted, only transit
        #plot_file = "Plots/public_transport_plus_intermediate_all_edges.png" # Unweighted, both transit and walking
        #plot_file = "Plots/public_transport_walking_weighted.png" # Weighted, walking only
        #plot_file = "Plots/public_transport_transit_systems_weighted.png" # Weighted, transit only

        #plot_file = "Plots/public_transport_all_weighted_10plus_node_degree.png"
        #plot_file = "Plots/public_transport_walking_weighted_10plus_node_degree.png"
        #plot_file = "Plots/public_transport_transit_systems_weighted_10plus_node_degree.png"

        #plot_file = "Plots/public_transport_all_weighted_10plus_edges.png"
        #plot_file = "Plots/public_transport_walking_weighted_10plus_edges.png"
        #plot_file = "Plots/public_transport_transit_systems_weighted_10plus_edges.png"

        #plot_file = "Plots/public_transport_walking_weighted_10plus_edges_scaled_weight.png"
        plot_file = "Plots/public_transport_walking_weighted_10plus_edges_scaled_distance.png"

        # CHOOSE PLOTTING OPTIONS
        plot_edges = True
        edge_weight_threshold = 10
        edge_scaling = 'distance_meters'
        plot_nodes = False
        node_scaling = ''

        # PLOT
        public_transport.draw_map(plot_file, plot_edges=plot_edges, edge_weight_threshold=edge_weight_threshold, edge_scaling=edge_scaling,
                                             plot_nodes=plot_nodes, node_scaling=node_scaling)

    #####################################
    # METRICS
    #####################################
    if False:

        # Compute and plot node centrality
        if False:
            centrality = compute_centrality(public_transport.graph, graph_type='weighted_inter', node_to_hash=public_transport.node_to_hash)
            public_transport.draw_map("Plots/public_transport_transit_node_centrality.png", plot_nodes=True, node_scaling='centrality', centrality=centrality)

        # Compute and plot edge weight distribution
        if False:
            # Load 3 graphs
            graph_file_all = PUBLIC_TRANSIT_GRAPH_WEIGHTED_PATH_LOAD # Weighted, both transit and walking
            graph_file_walking = PUBLIC_TRANSIT_GRAPH_PATH_LOAD_WALKING_WEIGHTED # Weighted, only walking
            graph_file_transit = PUBLIC_TRANSIT_GRAPH_PATH_LOAD_TRANSIT_WEIGHTED # Weighted, only transit
            graph_all = PublicTransport(create_new=False, read_google_maps=False, plot_graph=False, \
                                        check_attributes=False, reduce_graph=False, graph_path=graph_file_all).graph
            graph_walking = PublicTransport(create_new=False, read_google_maps=False, plot_graph=False, \
                                        check_attributes=False, reduce_graph=False, graph_path=graph_file_walking).graph
            graph_transit = PublicTransport(create_new=False, read_google_maps=False, plot_graph=False, \
                                        check_attributes=False, reduce_graph=False, graph_path=graph_file_transit).graph
            # Compute
            X_all, Y_all = get_edge_weight_distribution(graph_all, 'weight')
            X_walking, Y_walking = get_edge_weight_distribution(graph_walking, 'weight')
            X_transit, Y_transit = get_edge_weight_distribution(graph_transit, 'weight')
            # Plot
            plt.figure(figsize=(15, 10))
            plt.loglog(X_all, Y_all, color='b', label='Walking and Transit')
            plt.loglog(X_walking, Y_walking, color='g', label='Only Walking')
            plt.loglog(X_transit, Y_transit, color='r', label='Only Transit')
            plt.xlabel('Edge Weight (log)')
            plt.ylabel('Proportion of Edges with a Given Weight (log)')
            plt.title('Edge Weight Distributions')
            plt.legend()
            plt.savefig("Plots/public_transit_edge_weight_distributions.png")

        # Compute and plot node classification
        if True:
            node_roles = find_node_roles(public_transport.graph, attribute='weight')
            public_transport.draw_map("Plots/public_transport_transit_10plus_node_classification_weight.png", plot_nodes=True, node_scaling='classification', classification=node_roles)
            node_roles = find_node_roles(public_transport.graph, attribute='distance_meters')
            public_transport.draw_map("Plots/public_transport_transit_10plus_node_classification_distance.png", plot_nodes=True, node_scaling='classification', classification=node_roles)
            node_roles = find_node_roles(public_transport.graph, attribute='duration_seconds')
            public_transport.draw_map("Plots/public_transport_transit_10plus_node_classification_duration.png", plot_nodes=True, node_scaling='classification', classification=node_roles)

        # Compute clustering coefficients
        if False:
            # Load 3 graphs
            graph_file_all = PUBLIC_TRANSIT_GRAPH_WEIGHTED_PATH_LOAD # Weighted, both transit and walking
            graph_file_walking = PUBLIC_TRANSIT_GRAPH_PATH_LOAD_WALKING_WEIGHTED # Weighted, only walking
            graph_file_transit = PUBLIC_TRANSIT_GRAPH_PATH_LOAD_TRANSIT_WEIGHTED # Weighted, only transit
            graph_all = PublicTransport(create_new=False, read_google_maps=False, plot_graph=False, \
                                        check_attributes=False, reduce_graph=False, graph_path=graph_file_all).graph
            graph_walking = PublicTransport(create_new=False, read_google_maps=False, plot_graph=False, \
                                        check_attributes=False, reduce_graph=False, graph_path=graph_file_walking).graph
            graph_transit = PublicTransport(create_new=False, read_google_maps=False, plot_graph=False, \
                                        check_attributes=False, reduce_graph=False, graph_path=graph_file_transit).graph
            # Print
            print('All transport: %.4f' % snap.GetClustCf(graph_all, -1))
            print('Only walking: %.4f' % snap.GetClustCf(graph_walking, -1))
            print('Only transit: %.4f' % snap.GetClustCf(graph_transit, -1))

            # Load 3 graphs
            graph_file_all = PUBLIC_TRANSIT_GRAPH_PATH_LOAD_ALL_WEIGHTED_10PLUS
            graph_file_walking = PUBLIC_TRANSIT_GRAPH_PATH_LOAD_WALKING_WEIGHTED_10PLUS
            graph_file_transit = PUBLIC_TRANSIT_GRAPH_PATH_LOAD_TRANSIT_WEIGHTED_10PLUS
            graph_all = PublicTransport(create_new=False, read_google_maps=False, plot_graph=False, \
                                        check_attributes=False, reduce_graph=False, graph_path=graph_file_all).graph
            graph_walking = PublicTransport(create_new=False, read_google_maps=False, plot_graph=False, \
                                        check_attributes=False, reduce_graph=False, graph_path=graph_file_walking).graph
            graph_transit = PublicTransport(create_new=False, read_google_maps=False, plot_graph=False, \
                                        check_attributes=False, reduce_graph=False, graph_path=graph_file_transit).graph
            # Print
            print('All transport (10+): %.4f' % snap.GetClustCf(graph_all, -1))
            print('Only walking (10+): %.4f' % snap.GetClustCf(graph_walking, -1))
            print('Only transit (10+): %.4f' % snap.GetClustCf(graph_transit, -1))

        # Compute and plot edge distance distribution for walking graph
        if False:
            # Load 3 graphs
            graph_file_walking = PUBLIC_TRANSIT_GRAPH_PATH_LOAD_WALKING_WEIGHTED_10PLUS
            graph_walking = PublicTransport(create_new=False, read_google_maps=False, plot_graph=False, \
                                        check_attributes=False, reduce_graph=False, graph_path=graph_file_walking).graph

            # Compute
            X_walking, Y_walking = get_edge_weight_distribution(graph_walking, 'distance_meters')
            # Plot
            plt.figure(figsize=(15, 10))
            plt.loglog(X_walking, Y_walking, color='g')
            plt.xlabel('Edge Distance (meters)')
            plt.ylabel('Proportion of Edges with a Given Distance')
            plt.title('Walking Trips: Edge Distance Distributions')
            plt.legend()
            plt.savefig("Plots/public_transit_walking_edge_distance_distributions.png")
