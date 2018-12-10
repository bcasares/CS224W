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
