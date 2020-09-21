#!/bin/bash

INPUT_FILE=$1
OUTPUT_FILE=$2

MASTER_LOCATION="spark://c220g5-111020vm-1.wisc.cloudlab.us:7077"

SCRIPT_FILE=pagerank.py

/mnt/data/spark-2.4.7-bin-hadoop2.7/bin/spark-submit --master $MASTER_LOCATION /mnt/data/code/cs744_assignments/assignment1/part3/$SCRIPT_FILE $INPUT_FILE $OUTPUT_FILE

