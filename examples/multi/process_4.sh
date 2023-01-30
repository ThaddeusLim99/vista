#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
rm ./examples/vista_traces/lidar_4/lidar_3d*
rm ./examples/vista_traces/lidar_4/log.txt
touch ./examples/vista_traces/lidar_4/log.txt

file_list=(`ls /media/sangwon/My\ Book1/Data\ Collection\ 2017/LiDAR_LAS/*.las.zip | sed -r 's/^.+\///' | sort`)
length=${#file_list[@]}
for (( i = 3; i < length; i+=8 )); do
  python ./examples/conversion/convert_multi.py --input /media/sangwon/My\ Book1/Data\ Collection\ 2017/LiDAR_LAS/"${file_list[$i]}" --process 4 || continue
  python ./examples/basic_usage/sim_lidar.py --trace-path ./examples/vista_traces/lidar_4 --frame $i
  rm ./examples/vista_traces/lidar_4/lidar_3d*
done