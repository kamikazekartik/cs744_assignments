# make sure to set spark context etc up here (and imports)
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("TestApp").setMaster("spark://c220g5-111230vm-1.wisc.cloudlab.us:7077")
sc = SparkContext(conf=conf)

file_path = "hdfs://10.10.1.1:9000/assignment1/export.csv"

# read the file from HDFS
rdd_org = sc.textFile(file_path)
# extract and filter out header
header = rdd_org.first()
rdd_header = sc.parallelize([header])
rdd = rdd_org.filter(lambda line: line != header)

rdd = rdd.map(lambda line: line.split(","))
rdd = rdd.map(lambda line: (line[2] + "--" + line[-1], line))

def print_nice(x):
	for i in x:
		print(i)

rdd_sorted = rdd.sortByKey()

rdd_sorted_file = rdd_sorted.map(lambda line: ",".join(line[1]))
# put the header back
rdd_sorted_file = rdd_header.union(rdd_sorted_file)

# write rdd_sorted_file to file
rdd_sorted_file.saveAsTextFile("hdfs://10.10.1.1:9000/assignment1/sorted_output.csv")
