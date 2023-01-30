#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
rm ./examples/vista_traces/lidar_8/lidar_3d*
rm ./examples/vista_traces/lidar_8/log.txt
touch ./examples/vista_traces/lidar_8/log.txt

file_list=(`ls /media/sangwon/My\ Book1/Data\ Collection\ 2017/LiDAR_LAS/*.las.zip | sed -r 's/^.+\///' | sort`)
length=${#file_list[@]}
for (( i = 7; i < length; i+=8 )); do
  python ./examples/conversion/convert_multi.py --input /media/sangwon/My\ Book1/Data\ Collection\ 2017/LiDAR_LAS/"${file_list[$i]}" --process 8 || continue
  python ./examples/basic_usage/sim_lidar.py --trace-path ./examples/vista_traces/lidar_8 --frame $i
  rm ./examples/vista_traces/lidar_8/lidar_3d*
done