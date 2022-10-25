#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for i in {0..1000000..10000}
do
  python ./examples/conversion/las_to_h5.py --input /media/sangwon/My\ Book/Data\ Collection\ 2017/LiDAR_LAS/01A02E_C1R1_00000_04000.las.zip --frame $i
  python ./examples/basic_usage/sim_lidar.py --trace-path ./examples/vista_traces/lidar
done
