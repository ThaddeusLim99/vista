#!/bin/bash

total=2558
for i in {4..2558..6}
do
    python ./examples/conversion/convert_single.py --input ./examples/vista_traces/01A02E_C1R1_16000_20000_START_HZ.las --frame ${i} --range 100 --process 5
    python ./examples/basic_usage/sim_lidar.py --trace-path ./examples/vista_traces/lidar_5 --frame ${i} --resolution 0.05 --yaw-min -180 --yaw-max 180 --pitch-min -60 --pitch-max 30
    rm ./examples/vista_traces/lidar_5/lidar_3d*
done | tqdm --total $total >> /dev/null