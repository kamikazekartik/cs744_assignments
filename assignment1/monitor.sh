#!/bin/bash
echo "--------------------------------------------------------------------------------" > ~/system_status.log
echo "Logging system details out to ~/system_status.log every 15 seconds. Press CTRL-C to stop."
while :
do
	date >> ~/system_status.log
	echo "" >> ~/system_status.log
	df -h | grep "/mnt/data" >> ~/system_status.log
	echo "" >> ~/system_status.log
	free -m >> ~/system_status.log
	echo "" >> ~/system_status.log
	top -bn1 | grep "python\|spark\|java\|COMMAND" >> ~/system_status.log
	echo "--------------------------------------------------------------------------------" >> ~/system_status.log
	sleep 15
done
