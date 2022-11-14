#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
rm ./examples/vista_traces/lidar/data.h5
rm ./examples/vista_traces/lidar/log.txt
touch ./examples/vista_traces/lidar/log.txt
rm /tmp/lidar/trajectory.csv
for FILE in /media/sangwon/My\ Book/Data\ Collection\ 2017/LiDAR_LAS/*.las.zip
do
  for i in {10000..1000000..10000}
  do
    rm /tmp/lidar/*.las
    python ./examples/conversion/las_to_h5.py --input /media/sangwon/My\ Book/Data\ Collection\ 2017/LiDAR_LAS/"${FILE##*/}" --frame $i || break
    python ./examples/basic_usage/sim_lidar.py --trace-path ./examples/vista_traces/lidar
  done
done