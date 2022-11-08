#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
rm /tmp/lidar/trajectory.csv
for FILE in /media/sangwon/My\ Book/Data\ Collection\ 2017/LiDAR_LAS/*.las.zip
do
  for i in {1000..1000000..10000}
  do
    python ./examples/conversion/las_to_h5.py --input /media/sangwon/My\ Book/Data\ Collection\ 2017/LiDAR_LAS/"${FILE##*/}" --frame $i || break
    python ./examples/basic_usage/sim_lidar.py --trace-path ./examples/vista_traces/lidar
  done
done