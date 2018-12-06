from collections import defaultdict
import copy
from datetime import datetime
import googlemaps
import os
import pandas as pd
import snap
from tqdm import tqdm
import metrics


# GOOGLE_API_KEY = "AIzaSyAi5ERQfBOcdO54IT3MOmiCnNubFB1RWWY" # First Key, no-credit
# GOOGLE_API_KEY = "AIzaSyCnzf2UnjffaNCRX_rGvlypYBZb5rt6zyM" #abecs224w1 no-credit
# GOOGLE_API_KEY = "AIzaSyCUHPr2e0NmO5lwrB5cHuVUCDVeSDYWDQ0" #abecs224w2 no-credit
GOOGLE_API_KEY = "AIzaSyD7_s0eA-M967PpPLQfu9sKwnkiNVSHoE8" #abecs224w3

SF_ZONE_INFO = os.path.join("Data", "Geo", "sf_zone_info.csv")
SF_REDUCED_GRAPH = os.path.join("Data", "Geo", "Graphs", "sf_uber_final_graph.graph")
PUBLIC_TRANSIT_GRAPH_PATH_SAVE = os.path.join("Data", "Geo", "Graphs", "public_transit.graph")
# PUBLIC_TRANSIT_GRAPH_PATH_LOAD = os.path.join("Data", "Geo","Graphs",  "public_transit_complete.graph")
PUBLIC_TRANSIT_GRAPH_PATH_LOAD = os.path.join("Data", "Geo", "Graphs", "public_transit_reduced5pm.graph")
PUBLIC_TRANSIT_GRAPH_PLOT = os.path.join("Data", "Geo", "Images", "public_transit_graph_5pm.png")
# PUBLIC_TRANSIT_GRAPH_PLOT = os.path.join("Data", "Geo", "Images", "public_transit_graph_reduced.png")

class PublicTransport(object):

    """Public Transport Graph for CS224W Project. """

    def __init__(self, create_new=False, read_google_maps=False, plot_graph=False, check_attributes=False, reduce_graph=False):
        self.data = self.loadData()
        self.loadGraph(create_new=create_new)
        if reduce_graph:
            self.reduceGraph()
        if read_google_maps:
            # self.readGoogleMapsAndCreateGraph()
            self.readGoogleMapsAndCreateGraphFromExsitingGraph()
        if plot_graph:
            self.drawGraph()
        if check_attributes:
            self.checkAttributes()

    def loadData(self, file_path=SF_ZONE_INFO):
        """
        Load Uber zone data from sf_zone_info.csv
        """
        data = pd.read_csv(file_path)
        def latLong(row):
            return str(row.longitude) + ", " + str(row.latitude)
            # return str(row.latitude) + ", " + str(row.longitude)
        data["lat_long"] = data.apply(latLong, axis = 1)
        return data

    def loadGraph(self, file_path=PUBLIC_TRANSIT_GRAPH_PATH_LOAD, create_new=False, to_print=False):
        """
        Loads the graph if the graph already exists
        """
        # Load graph
        if create_new:
            self.graph = snap.TNEANet.New()
        else:
            try :
                FIn = snap.TFIn(file_path)
                self.graph = snap.TNEANet.Load(FIn)
                if to_print:
                    print "Number of nodes", self.graph.GetNodes()
                    print "Number of edges", self.graph.GetEdges()
            except :
                if to_print:
                    print("creating new graph")
                self.graph = snap.TNEANet.New()

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

    def readGoogleMapsAndCreateGraphFromExsitingGraph(self, date_time='Nov 7 2018  5:00PM'):
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


    def readGoogleMapsAndCreateGraph(self, date_time = 'Nov 7 2018  5:00PM'):
        """
        Use Google API to compute travel time between the points for each time of day
        (using public transportation)
        """
        gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

        # now = datetime.now()
        now = datetime.strptime(date_time, '%b %d %Y %I:%M%p')

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


    def addNodesAndEdgesPublicTransitGraph(self, zone1_id, zone2_id, edge_id, distance_meters, duration_seconds, to_print = False):
        # Add nodes
        if not self.graph.IsNode(zone1_id) : self.graph.AddNode(zone1_id)
        if not self.graph.IsNode(zone2_id) : self.graph.AddNode(zone2_id)
        if not self.graph.IsEdge(zone1_id, zone2_id) and not self.graph.IsEdge(zone2_id, zone1_id):
            self.graph.AddEdge(zone1_id, zone2_id, edge_id)
            self.graph.AddEdge(zone2_id, zone1_id, edge_id+1)
            self.graph.AddFltAttrDatE(edge_id, distance_meters, 'distance_meters')
            self.graph.AddFltAttrDatE(edge_id, duration_seconds, 'duration_seconds')
            self.graph.AddFltAttrDatE(edge_id+1, distance_meters, 'distance_meters')
            self.graph.AddFltAttrDatE(edge_id+1, duration_seconds, 'duration_seconds')

        if to_print:
            num_edges = self.graph.GetEdges()
            num_nodes = self.graph.GetNodes()

            # Print some properties of the graph
            print('Number of nodes (zones): %d' % num_nodes)
            print('Number of edges (zone borders): %d' % num_edges)

if __name__ == "__main__":
    public_transport = PublicTransport(create_new=False, read_google_maps=False, plot_graph=False, check_attributes=False, reduce_graph=False)
    metrics.plotDegreeDistribution(original_graph=public_transport.graph, attribute="duration_seconds", type_graph="public_transit")
    metrics.plotDegreeDistribution(original_graph=public_transport.graph, attribute="distance_meters", type_graph="public_transit")
    metrics.compute_centrality(public_transport.graph, graph_type="public_transit")
    metrics.find_node_roles(public_transport.graph, attributes=["duration_seconds", "distance_meters"], graph_type="public_transit")

    # uber_graph = metrics.load_graph()
    # metrics.plotDegreeDistribution(original_graph=uber_graph, attribute="travel_time_17", type_graph="uber")
    # metrics.plotDegreeDistribution(original_graph=uber_graph, attribute="travel_speed_17", type_graph="uber")


