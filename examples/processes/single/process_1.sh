#!/bin/bash

file=$LASFILE
total=`cat examples/Trajectory/${file%.*}/forwards.csv | wc -l`

resolution=$RESOLUTION
pitch_min=$PITCH_MIN
pitch_max=$PITCH_MAX
yaw_min=$YAW_MIN
yaw_max=$YAW_MAX
range=$RANGE

for (( i=0; i<total; i+=6 ))
do
    python ./examples/conversion/convert_single.py --input ./examples/vista_traces/${file} --frame ${i} --range ${range} --process 1
    python ./examples/basic_usage/sim_lidar.py --trace-path ./examples/vista_traces/lidar_1 --filename ${file} --frame ${i} --resolution ${resolution} --yaw-min ${yaw_min} --yaw-max ${yaw_max} --pitch-min ${pitch_min} --pitch-max ${pitch_max}
    rm ./examples/vista_traces/lidar_1/lidar_3d*
done 

exit