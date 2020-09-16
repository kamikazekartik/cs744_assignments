# make sure to set spark context etc up here (and imports)
from pyspark import SparkContext, SparkConf
from operator import add

conf = SparkConf().setAppName("TestApp")
sc = SparkContext(conf=conf)

file_path = "hdfs://10.10.1.1:9000/part3/wikidata/wikidata.csv"
# file_path = "hdfs://10.10.1.1:9000/part3/filtered_web-BerkStan.txt"

lines = sc.textFile(file_path)
# ^ .persist() ?
links = lines.map(lambda line: tuple(line.split("\t")))
links_1 = links.map(lambda line: line[0])
links_2 = links.map(lambda line: line[1])
links_all = sc.union([links_1,links_2]).distinct()
ranks = links_all.map(lambda line: (line, 1) )

# COMMENT TO US: DELETE
# GROUP BY WAS KEY!!!
links = links.groupByKey()
ranks = links.map(lambda url_neighbors: (url_neighbors[0], 1.0))


for iteration in range(10):
    # Calculates URL contributions to the rank of other URLs.
    contribs = links.join(ranks).flatMap(
    	lambda (url, (links, rank)): [(x, rank/len(links)) for x in links])

    # Re-calculates URL ranks based on neighbor contributions.
    ranks = contribs.reduceByKey(lambda x, y: x+y).mapValues(lambda rank: rank * 0.85 + 0.15)

# write ranks to file
# ranks.saveAsTextFile("hdfs://10.10.1.1:9000/part3/berkstan_ranks.csv")
ranks.saveAsTextFile("hdfs://10.10.1.1:9000/part3/wikidata_ranks.csv")
