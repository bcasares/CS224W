import snap
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from collections import deque, defaultdict, Counter

def readEmb(file_path):
    emb = {}
    with open(file_path) as f:
        for i, line in enumerate(f):
            L = deque(line.split())
            if i == 0 :
                continue
            nid = int(L.popleft())
            L = np.asarray(L)
            emb[nid] = np.asarray(map(float, L))
    return emb

def dotProductNodeEmbeding(node_id, emb):
    node_similarity = {}
    for nid, vec2 in emb.iteritems():
        node_similarity[(node_id, nid)] = np.dot(emb[node_id], vec2)
    top_5 = dict(Counter(node_similarity).most_common(6))
    print top_5

def calculate_top5():
    # emb = readEmb("node2vec/emb/intermediate.emd")
    # emb = readEmb("node2vec/emb/intermediate_p_0.001_q_1000.emd")
    emb = readEmb("node2vec/emb/intermediate_p_1000_q_0.001.emd")
    dotProductNodeEmbeding(2828,emb)

if __name__ == "__main__":
    calculate_top5()
