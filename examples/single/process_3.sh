#!/bin/bash

for i in {2..571..4} 
do
    python ./examples/conversion/convert_single.py --input ./examples/vista_traces/74202W_C1L1_L1L1_08000_06000.las --frame ${i} --process 3
    python ./examples/basic_usage/sim_lidar.py --trace-path ./examples/vista_traces/lidar_3 --resolution 0.05 --frame ${i}
    rm ./examples/vista_traces/lidar_3/lidar_3d*
done