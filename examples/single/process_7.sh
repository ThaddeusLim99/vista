#!/bin/bash

for i in {6..571..8} 
do
    python ./examples/conversion/convert_single.py --input ./examples/vista_traces/74202W_C1L1_L1L1_08000_06000.las --frame ${i} --process 7
    python ./examples/basic_usage/sim_lidar.py --trace-path ./examples/vista_traces/lidar_7 --resolution 0.05 --frame ${i}
    rm ./examples/vista_traces/lidar_7/lidar_3d*
done