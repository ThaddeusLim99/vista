#!/bin/bash

file=$LASFILE

resolution=$RESOLUTION
pitch_min=$PITCH_MIN
pitch_max=$PITCH_MAX
yaw_min=$YAW_MIN
yaw_max=$YAW_MAX
range=$RANGE
culling_r=$CULLING_R

startframe=$STARTFRAME
endframe=$ENDFRAME

for (( i=startframe+3; i<endframe; i+=6 ))
do
    python ./examples/conversion/convert_single.py --input ./examples/vista_traces/${file} --frame ${i} --range ${range} --process 4
    python ./examples/basic_usage/sim_lidar.py --trace-path ./examples/vista_traces/lidar_4 --filename ${file} --frame ${i} --resolution ${resolution} --yaw-min ${yaw_min} --yaw-max ${yaw_max} --pitch-min ${pitch_min} --pitch-max ${pitch_max} --culling-r ${culling_r}
    rm ./examples/vista_traces/lidar_4/lidar_3d*
done 

exit