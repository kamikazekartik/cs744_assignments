# make sure to set spark context etc up here (and imports)
from pyspark import SparkContext, SparkConf
from operator import add

import sys

NUM_ITERATIONS = 2

conf = SparkConf().setAppName("TestApp")
sc = SparkContext(conf=conf)

def solve_record(record):
    if record[1][1] is not None:
        return (record[0], record[1][0] + record[1][1])
    return (record[0], record[1][0])


file_path = sys.argv[1]
file_path_out = sys.argv[2]

lines = sc.textFile(file_path)

# ^ .persist() ?

links = lines.filter(lambda line: ((not ':' in line) or (line.startswith("Category:"))) and (not line.startswith("#")))
links = links.map(lambda line: tuple(line.lower().split("\t")))
links = links.filter(lambda line_lst: len(line_lst) == 2)

src_urls = links.map(lambda line: line[0]).distinct()

ranks = src_urls.map(lambda line: (line, 1.0))
src_base_ranks = src_urls.map(lambda line: (line, 0.15))

links = links.distinct().groupByKey().cache()


for iteration in range(NUM_ITERATIONS):
    # Calculates URL contributions to the rank of other URLs
    contribs = links.join(ranks).flatMap(
    	lambda (src_url, (links, src_rank)): [(dst_link, src_rank/len(links)) for dst_link in links])

    # Re-calculates URL ranks based on neighbor contributions.
    ranks = contribs.reduceByKey(lambda x, y: x+y).mapValues(lambda rank: rank * 0.85)
    ranks = src_base_ranks.leftOuterJoin(ranks).map(solve_record)


# write ranks to file
#ranks.map(lambda line: ','.join([str(element) for element in line])).saveAsTextFile(file_path_out)

ranks = ranks.map(lambda line: ','.join((str(element) for element in line)))
ranks.saveAsTextFile(file_path_out)
