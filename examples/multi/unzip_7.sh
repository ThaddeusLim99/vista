#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

file_list=(`ls /media/sangwon/My\ Book1/Data\ Collection\ 2017/LiDAR_LAS/*.las.zip | sed -r 's/^.+\///' | sort`)
length=${#file_list[@]}
for (( i = 6; i < length; i+=8 )); do
  python ./examples/conversion/unzip.py --input /media/sangwon/My\ Book1/Data\ Collection\ 2017/LiDAR_LAS/"${file_list[$i]}"
done