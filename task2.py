import os
from pyspark import SparkContext
import sys
import json
import time
import copy
import itertools
from collections import Counter

# spark-submit task2.py <filter threshold> <input_file_path> <betweenness_output_file_path>  <community_output_file_path>
# spark-submit task2.py 7 ub_sample_data.csv betweenness.txt community.txt

start_time = time.time()

filter_threshold = int(sys.argv[1])
input_file_path = sys.argv[2]
betweenness_output_file_path = sys.argv[3]
community_output_file_path = sys.argv[4]

sc = SparkContext('local[*]', 'task2')
sc.setLogLevel('ERROR')

rdd0 = sc.textFile(input_file_path)
header = rdd0.first()  # extract header
rdd1 = rdd0.filter(lambda row: row != header)  # filter out header
rdd2 = rdd1.map(lambda line: (line.split(',')[1], [line.split(',')[0]])) \
    .reduceByKey(lambda a, b: a + b) \
    .mapValues(lambda v: set(v)) \
    .flatMapValues(lambda x: itertools.combinations(x, 2)) \
    .mapValues(lambda x: tuple(sorted(x))) \
    .map(lambda x: (x[1], [x[0]])) \
    .reduceByKey(lambda a, b: a + b) \
    .mapValues(lambda x: set(x)) \
    .filter(lambda x: len(x[1]) >= filter_threshold) \
    .map(lambda x: x[0]) \
    .persist()

rdd3 = rdd2.flatMap(lambda x: x) \
    .distinct()

u_ids = rdd3.collect()
u_ids.sort()
u_index_map = {}
index_u_map = {}

for index, user in enumerate(u_ids):
    u_index_map[user] = index
    index_u_map[index] = user

rddedges = rdd2.map(lambda x: (u_index_map[x[0]], u_index_map[x[1]]))\
    .persist()  # (a, b) only
m = rddedges.count()
rddv2v = rddedges.flatMap(lambda x: [(x[0], [x[1]]), (x[1], [x[0]])]) \
    .reduceByKey(lambda x, y: x + y) \
    .persist()  # (user, [connected users])

vertices = rddv2v.map(lambda x: x[0]).persist()

node_list = vertices.collect()

v2v = rddv2v.collect()

graph = {item[0]: item[1] for item in v2v}



# 498 edges
# 222 vertices

# graph = {
#         'e': ['d', 'f'],
#         'd': ['e', 'f', 'g'],
#         'f': ['d', 'e', 'g'],
#         'b': ['a', 'c'],
#         'g': ['d', 'f'],
#         'a': ['b', 'c'],
#         'c': ['a', 'b']
#         }


def bfs(g, root):
    visited = set()  # update
    cpc_dict = {}  # update
    v_level_list = []  # update
    newroot_set = set([root])  # update
    v_level_list.append(newroot_set)
    cpc_dict[root] = {root: 1}
    visited.add(root)
    while len(newroot_set) != 0:
        oldroot_set = copy.deepcopy(newroot_set)
        newroot_set = set()
        for oldroot in oldroot_set:
            connectedv_list = g[oldroot]
            for connectedv in connectedv_list:
                if connectedv not in visited:
                    newroot_set.add(connectedv)
                    pathcount = sum(cpc_dict[oldroot].values())
                    if connectedv not in cpc_dict:
                        cpc_dict[connectedv] = {}
                    cpc_dict[connectedv].update({oldroot: pathcount})
        visited.update(newroot_set)
        v_level_list.append(newroot_set)
    return cpc_dict, v_level_list

def edgeweight(cpc_dict, v_level_list):
    verlist = list(cpc_dict.keys())
    vcredit_dict = {key: 1.0 for key in verlist}
    ecredit_list = []
    pcredit_dict = {}  # {vertex: [credits]}
    for levelv_set in reversed(v_level_list[1:]):
        for vertex in levelv_set:
            tmpdict = cpc_dict[vertex]
            totalpath = sum(tmpdict.values())
            if vertex in pcredit_dict:
                vcredit = sum(pcredit_dict[vertex])
                vcredit_dict[vertex] = vcredit_dict[vertex] + vcredit
            for parent, count in tmpdict.items():
                ecredit = float(vcredit_dict[vertex]) * count / totalpath
                if parent in pcredit_dict:
                    pcredit_dict[parent].append(ecredit)
                else:
                    pcredit_dict[parent] = []
                    pcredit_dict[parent].append(ecredit)
                key = tuple(sorted([parent, vertex]))
                ecredit_list.append((key, ecredit))
    return ecredit_list  # [((smallerv,  largerv), credit)]


def betweenness(g, root):
    cpc_dict, v_level_list = bfs(g, root)
    ecredit_list = edgeweight(cpc_dict, v_level_list)
    return ecredit_list


# print(betweenness('e'))

rddbetweenness = vertices.flatMap(lambda x: betweenness(graph, x))\
    .reduceByKey(lambda a, b: a + b)\
    .mapValues(lambda x: x/2)\
    .sortBy(lambda x: x[1], ascending=False)\
    .collect()  # (v1, v2), betweenness

betweennessresult = [((index_u_map[x[0][0]], index_u_map[x[0][1]]), x[1]) for x in rddbetweenness]

betweennessresult = sorted(betweennessresult, key=lambda x: x[0][0])
betweennessresult = sorted(betweennessresult, key=lambda x: x[1], reverse=True)

boutput = open(betweenness_output_file_path, 'w')
for item in betweennessresult:
    boutput.write("('" + str(item[0][0]) + "', '" + str(item[0][1]) + "'), " + str(item[1]) + "\n")
boutput.close()
rdd2.unpersist()
vertices.unpersist()
rddedges.unpersist()
rddv2v.unpersist()

# communities
# m = 498
A = copy.deepcopy(graph)

# betweennessresult = [(('b','d'),12),(('a','b'),5),(('b','c'),5),(('d','e'),4.5)
#     ,(('d','g'),4.5),(('d','f'),4),(('e','f'),1.5),(('f','g'),1.5),(('a','c'),1)]
#
# A = {
#         'e': ['d', 'f'],
#         'd': ['e', 'f', 'g','b'],
#         'f': ['d', 'e', 'g'],
#         'b': ['a', 'c','d'],
#         'g': ['d', 'f'],
#         'a': ['b', 'c'],
#         'c': ['a', 'b']
#         }
#
# graph = {
#         'e': ['d', 'f'],
#         'd': ['e', 'f', 'g'],
#         'f': ['d', 'e', 'g'],
#         'b': ['a', 'c'],
#         'g': ['d', 'f'],
#         'a': ['b', 'c'],
#         'c': ['a', 'b']
#         }
#
# m = 9


def findroot(aNode, aRoot):
    while aNode != aRoot[aNode][0]:
        aNode = aRoot[aNode][0]
    return aNode, aRoot[aNode][1]


def connected_components(newgraph):
    rc_dict = {}
    for vertex in newgraph.keys():
        rc_dict[vertex] = (vertex,0)
    for myI in newgraph:
        for myJ in newgraph[myI]:
            rc_dict_myI, myDepthMyI = findroot(myI, rc_dict)
            rc_dict_myJ, myDepthMyJ = findroot(myJ, rc_dict)
            if rc_dict_myI != rc_dict_myJ:
                myMin = rc_dict_myI
                myMax = rc_dict_myJ
                if  myDepthMyI > myDepthMyJ:
                    myMin = rc_dict_myJ
                    myMax = rc_dict_myI
                rc_dict[myMax] = (myMax, max(rc_dict[myMin][1]+1, rc_dict[myMax][1]))
                rc_dict[myMin] = (rc_dict[myMax][0], -1)
    myToRet = {}
    for myI in newgraph:
        if rc_dict[myI][0] == myI:
            myToRet[myI] = []
    for myI in newgraph:
        myToRet[findroot(myI, rc_dict)[0]].append(myI)
    return list(myToRet.values())

global_modularity = -99
global_communities = []
cut = rddbetweenness[0][0]
cutnum = 0

while cutnum != m:
    cutnum += 1
    # print(cut)
    tmplist = graph[cut[0]]
    tmplist.remove(cut[1])
    graph.update({cut[0]: tmplist})
    tmplist = graph[cut[1]]
    tmplist.remove(cut[0])
    graph.update({cut[1]: tmplist})

    communities = connected_components(graph)
    modularity = 0.0
    for community in communities:
        for nodeI in community:
            for nodeJ in community:
                tmpq = int(nodeJ in A[nodeI])
                tmpq -= float(len(A[nodeI])*len(A[nodeJ]))/(2*m)
                modularity += tmpq
    modularity = modularity/(2*m)
    if modularity > global_modularity:
        global_modularity = modularity
        global_communities = communities

    betweennessdict = {}
    for node in node_list:
        graph = graph
        tmp = betweenness(graph, node)
        # print(tmp)
        for item in tmp:
            edge = item[0]
            val = item[1]
            if edge in betweennessdict:
                betweennessdict[edge] = betweennessdict.get(edge, 0) + val
            else:
                betweennessdict[edge] = val
    if len(betweennessdict) == 0:
        break
    else:
        cut = max(betweennessdict, key=betweennessdict.get)

communityresult = [sorted([index_u_map[x] for x in community]) for community in global_communities]
communityresult = sorted(communityresult, key=lambda x: x[0])
communityresult = sorted(communityresult, key=len)

coutput = open(community_output_file_path, 'w')
for community in communityresult:
    for i in range(len(community)):
        if i == len(community) - 1:
            coutput.write("'" + community[i] + "'" + "\n")
        else:
            coutput.write("'" + community[i] + "', ")

print('Duration: ' + str(time.time() - start_time))
