from functools import reduce
from graphframes import *
import os
from pyspark import SparkContext
import sys
import json
import time
import copy
import itertools
from pyspark.sql import SQLContext, SparkSession
import pyspark

# spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 task1.py <filter_threshold> <input_file_path> <community_output_file_path>
# spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 task1.py 7 ub_sample_data.csv task1.txt

# pip install graphframes
os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")

start_time = time.time()

filter_threshold = int(sys.argv[1])
input_file_path = sys.argv[2]
community_output_file_path = sys.argv[3]



scConf = pyspark.SparkConf() \
    .setAppName('task1') \
    .setMaster('local[3]')

sc = pyspark.SparkContext(conf=scConf)
# sc = SparkContext('local[1]', 'task1')
sc.setLogLevel('ERROR')
sqlContext = SQLContext(sc)

# spark = SparkSession \
#     .builder \
#     .appName("task1") \
#     .getOrCreate()

rdd0 = sc.textFile(input_file_path)
header = rdd0.first()  # extract header
rdd1 = rdd0.filter(lambda row: row != header)   # filter out header
rdd2 = rdd1.map(lambda line: (line.split(',')[1], [line.split(',')[0]]))\
    .reduceByKey(lambda a, b: a + b)\
    .mapValues(lambda v: set(v))\
    .flatMapValues(lambda x: itertools.combinations(x, 2))\
    .mapValues(lambda x: tuple(sorted(x))) \
    .map(lambda x: (x[1], [x[0]])) \
    .reduceByKey(lambda a, b: a + b) \
    .mapValues(lambda x: set(x)) \
    .filter(lambda x: len(x[1]) >= filter_threshold) \
    .map(lambda x: x[0])\
    .persist()

rdd3 = rdd2.flatMap(lambda x: x)\
    .distinct()\
    .map(lambda x: (x,))\
    .persist()

# u_ids = rdd3.collect()
# #u_ids.sort()
# u_index_map = {}
#
# for index, user in enumerate(u_ids):
#     u_index_map[user] = index

rdd4 = rdd2.map(lambda x: [(x[0], x[1]), (x[1], x[0])])\
    .flatMap(lambda x: x)
# .map(lambda x: (u_index_map[x[0]], u_index_map[x[1]]))\
# rdd5 = rdd3.map(lambda x: (u_index_map[x], x))

edges = rdd4.toDF(["src", "dst"])

vertices = rdd3.toDF(["id"])

graph = GraphFrame(vertices, edges)

communities = graph.labelPropagation(maxIter=5)

rdd6 = communities.rdd.map(lambda x: (x[1], [x[0]]))\
    .reduceByKey(lambda a, b: a + b)\
    .map(lambda x: sorted(x[1]))\
    .collect()

sorted_list = sorted(rdd6, key=lambda x: x[0])
sorted_list = sorted(sorted_list, key=len)

output = open(community_output_file_path, 'w')
for com in sorted_list:
    for i in range(len(com)):
        if i != len(com) - 1:
            output.write("'"+com[i]+"', ")
        else:
            output.write("'"+com[i]+"'"+"\n")

print('Duration: ' + str(time.time() - start_time))
