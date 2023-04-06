#!/bin/bash

file=$LASFILE
total=`cat examples/Trajectory/forwards.csv | wc -l`

for i in `seq 1 $total 6`
do
    python ./examples/conversion/convert_single.py --input ./examples/vista_traces/${file} --frame ${i} --range 100 --process 1
    python ./examples/basic_usage/sim_lidar.py --trace-path ./examples/vista_traces/lidar_1 --filename ${file} --frame ${i} --resolution 0.1 --yaw-min 180 --yaw-max 180 --pitch-min -60 --pitch-max 30
    rm ./examples/vista_traces/lidar_1/lidar_3d*
done 