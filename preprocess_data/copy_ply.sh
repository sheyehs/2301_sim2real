#!/bin/bash

iroot='/scratch/gc2720/2301_sim2real/2022-11-15/'
modelLeaf=('SongFeng/SF-CJd60-097-016-016/part_info/SF-CJd60-097-016-016.master.ply' \
           'SongFeng/SF-CJd60-097-026/part_info/SF-CJd60-097-026.master.ply' \
           'SongFeng/SongFeng_0005/part_info/SongFeng_0005.master.ply' \
           'SongFeng/SongFeng_306/part_info/SongFeng_306.master.ply' \
           'SongFeng/SongFeng_311/part_info/SongFeng_311.master.ply' \
           'SongFeng/SongFeng_318/part_info/SongFeng_318.master.ply' \
           'SongFeng/SongFeng_332/part_info/SongFeng_332.master.ply' \
           'Toyota/21092302/part_info/21092302.master.ply' \
           'ZSRobot/6010018CSV/part_info/6010018CSV.master.ply' \
           'ZSRobot/6010022CSV/part_info/6010022CSV.master.ply')

oroot='/scratch/gc2720/2301_sim2real/models/'

for i in ${!modelLeaf[@]}
do
    # echo "${modelArray[$i]}" 
    # echo $(printf "${dir}obj_%02d.ply" $((i+1))) 
    cp "${iroot}${modelLeaf[$i]}" "${oroot}"
done