# make sure to set spark context etc up here (and imports)
from pyspark import SparkContext, SparkConf
from operator import add

import sys

# Number of iterations to run algorithm at
NUM_ITERATIONS = 10
# Whether or not we cache the links and src_url_base_ranks in RAM.
CACHE_BASE_DATA = True

conf = SparkConf().setAppName("TestApp")
sc = SparkContext(conf=conf)

# 'Solves a record' from a join, adding the base record value to the value calculated
# by contribs. If no value has been calculated by contribs, we just take this base value
# (in this case, 0.15)
def solve_record(record):
    if record[1] is not None:
        return record[0] + record[1]
    return record[0]


# Read input
file_path = sys.argv[1]
file_path_out = sys.argv[2]

lines = sc.textFile(file_path)

# Filter and pre-process links
links = lines.filter(lambda line: ((not ':' in line) or (line.startswith("Category:"))) and (not line.startswith("#")))
links = links.map(lambda line: tuple(line.lower().split("\t")))
links = links.filter(lambda line_lst: len(line_lst) == 2).distinct()

# Get ranks
src_urls = links.map(lambda line: line[0]).distinct()
ranks = src_urls.map(lambda line: (line, 1.0))

# Calculate base ranks and links values to be re-used in iterations.
# Depending on the given boolean at the top of the file, we may or may not cache these.
src_base_ranks = src_urls.map(lambda line: (line, 0.15))
links = links.groupByKey()

# Cache baseline data that is re-used in calculations, if necessary
if CACHE_BASE_DATA:
    src_base_ranks = src_base_ranks.cache()
    links = links.cache()

# Peroform the PageRank iterations
for iteration in range(NUM_ITERATIONS):
    # Calculates URL contributions form SRC URLs to all of their Desitnation URLs
    # Join on SRC URL between <links, ranks>
    contribs = links.join(ranks).flatMap(
    	lambda (src_url, (links, src_rank)): [(dst_link, src_rank/len(links)) for dst_link in links], preservesPartitioning=True)
    # Calculate the weighted sum of contribtu from sources to destinations
    # Reduce <dst_url as key>
    ranks = contribs.reduceByKey(lambda x, y: x+y, numPartitions = 10).mapValues(lambda rank: rank * 0.85)
    # Add these to the base rank of all sources; also, this re-introduces 'source only' nodes into
    # the rank matrix and filters out 'destination only nodes'
    # src_url as key
    ranks = src_base_ranks.leftOuterJoin(ranks).mapValues(solve_record)

# write ranks to file
ranks = ranks.map(lambda line: ','.join((str(element) for element in line)))
ranks.saveAsTextFile(file_path_out)
