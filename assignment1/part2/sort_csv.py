# make sure to set spark context etc up here (and imports)
from pyspark import SparkContext, SparkConf

import sys

conf = SparkConf().setAppName("TestApp")
sc = SparkContext(conf=conf)

# Read filepath arguments
file_path = sys.argv[1]
out_file_path = sys.argv[2]

# read the file from HDFS
rdd_org = sc.textFile(file_path)

# extract and filter out header
header = rdd_org.first()
rdd_header = sc.parallelize([header])
rdd = rdd_org.filter(lambda line: line != header)

# Append key using union of the country code and the timestamp
rdd = rdd.map(lambda line: line.split(","))
rdd = rdd.map(lambda line: (line[2] + "--" + line[-1], line))

# Sort by key
rdd_sorted = rdd.sortByKey()

# Generate and write out the sorted rdd to file
rdd_sorted_file = rdd_sorted.map(lambda line: ",".join(line[1]))

# put the header back
rdd_sorted_file = rdd_header.union(rdd_sorted_file)
rdd_sorted_file.saveAsTextFile(out_file_path)
