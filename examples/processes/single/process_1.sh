#!/bin/bash

for i in 219 215 153
do
    python ./examples/conversion/convert_single.py --input ./examples/vista_traces/74202W_C1L1_L1L1_08000_06000.las --frame ${i} --process 1
    python ./examples/basic_usage/sim_lidar.py --trace-path ./examples/vista_traces/lidar_1 --resolution 0.1 --frame ${i}
    rm ./examples/vista_traces/lidar_1/lidar_3d*
done