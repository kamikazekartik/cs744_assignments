#!/bin/bash

INPUT_FILE=$1
OUTPUT_FILE=$2

MASTER_LOCATION="spark://c220g5-110925vm-1.wisc.cloudlab.us:7077"

/mnt/data/spark-2.4.7-bin-hadoop2.7/bin/spark-submit --master $MASTER_LOCATION /mnt/data/code/cs744_assignments/assignment1/part2/sort_csv.py $INPUT_FILE $OUTPUT_FILE
i
