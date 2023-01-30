#!/bin/bash

for i in {1..4004..8} 
do
    python ./examples/conversion/convert.py --input ./examples/vista_traces/03210N_C1R1_16000_20000.las --frame ${i} --process 2
    python ./examples/basic_usage/sim_lidar.py --trace-path ./examples/vista_traces/lidar_2 --resolution 0.05 --frame ${i}
    rm ./examples/vista_traces/lidar_2/lidar_3d*
done