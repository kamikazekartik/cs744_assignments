import time
import dateutil.parser
import sys

# This script parses the output of monitor.sh into a nice TSV file
# Note that you need to install the dateutil module:
# $> pip install python-dateutil
# Call this script with:
# $> python parse_logs.py path/to/logfile.txt > path/to/data.tsv


def mem_strip_top_formatting(mem):
	# top reports memory usage in kb, otherwise appending 'm' if megabytes or 'g' if gigabytes
	# these lines parse through and convert the numbers to MB, to be consistent with node memory above
	if "m" in mem:
		mem = int(float(mem.replace("m",""))) # already in mb
	elif "g" in words[5]:
		mem = int(float(mem.replace("g",""))*1000) # gb --> mb
	elif "t" in words[5]:
		mem = int(float(mem.replace("t",""))*1000000) # tb --> mb
	else:
		mem = int(mem)/1000 # kb --> mb
	return mem



f = open(sys.argv[1], "r")
print("timestamp\tdisk_total_gb\tdisk_used_gb\tdisk_available_gb\tmem_total_mb\tmem_used_mb\tmem_free_mb\tmem_disk_cache_mb\tmem_available_mb\tspark_processes_num\tspark_processes_avg_cpu_pct\tspark_processes_avg_mem_mb\tpython_processes_num\tpython_processes_avg_cpu_pct\tpython_processes_avg_mem_mb")
line = f.readline().strip()
while True:
	if not line:
		break
	if line.startswith("-----"):

		# parse timestamp
		line = f.readline().strip()
		if not line:
			break # detect last line of file here
		timestamp = time.mktime(dateutil.parser.parse(line).timetuple())
		line = f.readline().strip() # skip empty line

		# parse disk usage
		line = f.readline().strip()
		nums = [int(word) for word in line.replace("G","").split() if word.isdigit()]
		disk_total_gb = nums[0]
		disk_used_gb = nums[1]
		disk_available_gb = nums[2]
		line = f.readline().strip() # skip empty line

		# parse memory usage
		line = f.readline().strip() # skip header line
		line = f.readline().strip()
		nums = [int(word) for word in line.split() if word.isdigit()]
		mem_total_mb = nums[0]
		mem_used_mb = nums[1]
		mem_free_mb = nums[2]
		mem_disk_cache_mb = nums[4]
		mem_available_mb = nums[5]
		line = f.readline().strip() # skip swap line
		line = f.readline().strip() # skip empty line

		# parse processes
		spark_processes_cpu_pct = []
		spark_processes_mem_mb = []
		spark_processes_num = 0
		python_processes_cpu_pct = []
		python_processes_mem_mb = []
		python_processes_num = 0
		line = f.readline().strip() # skip header line
		while True:
			line = f.readline().strip()
			if line.startswith("-----"):
				break
			words = [word for word in line.split()]
			if words[11] == "java":
				spark_processes_num += 1
				spark_processes_cpu_pct.append(float(words[8]))
				spark_processes_mem_mb.append(mem_strip_top_formatting(words[5]))
			elif words[11] == "python":
				python_processes_num += 1
				python_processes_cpu_pct.append(float(words[8]))
				python_processes_mem_mb.append(mem_strip_top_formatting(words[5]))

		# compute avg Spark process CPU usage
		if len(spark_processes_cpu_pct)>0:
			spark_processes_avg_cpu_pct = sum(spark_processes_cpu_pct) / len(spark_processes_cpu_pct)
		else:
			spark_processes_avg_cpu_pct = 0.0

		# compute avg Spark process memory usage
		if len(spark_processes_mem_mb)>0:
			spark_processes_avg_mem_mb = sum(spark_processes_mem_mb) / len(spark_processes_mem_mb)
		else:
			spark_processes_avg_mem_mb = 0.0

		# compute avg Python process CPU usage
		if len(python_processes_cpu_pct)>0:
			python_processes_avg_cpu_pct = sum(python_processes_cpu_pct) / len(python_processes_cpu_pct)
		else:
			python_processes_avg_cpu_pct = 0.0

		# compute avg Python process memory usage
		if len(python_processes_mem_mb)>0:
			python_processes_avg_mem_mb = sum(python_processes_mem_mb) / len(python_processes_mem_mb)
		else:
			python_processes_avg_mem_mb = 0.0


		print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (timestamp, disk_total_gb, disk_used_gb, disk_available_gb, mem_total_mb, mem_used_mb, mem_free_mb, mem_disk_cache_mb, mem_available_mb, spark_processes_num, spark_processes_avg_cpu_pct, spark_processes_avg_mem_mb, python_processes_num, python_processes_avg_cpu_pct, python_processes_avg_mem_mb))

f.close()
