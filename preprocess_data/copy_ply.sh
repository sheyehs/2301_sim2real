#!/bin/bash

modelArray=('/mnt/NAS/data/WenyuHan/sim2real/2022-11-15/SongFeng/SF-CJd60-097-016-016/part_info/SF-CJd60-097-016-016.master.ply' \
            '/mnt/NAS/data/WenyuHan/sim2real/2022-11-15/SongFeng/SF-CJd60-097-026/part_info/SF-CJd60-097-026.master.ply'
            '/mnt/NAS/data/WenyuHan/sim2real/2022-11-15/SongFeng/SongFeng_0005/part_info/SongFeng_0005.master.ply' \
            '/mnt/NAS/data/WenyuHan/sim2real/2022-11-15/SongFeng/SongFeng_306/part_info/SongFeng_306.master.ply' \
            '/mnt/NAS/data/WenyuHan/sim2real/2022-11-15/SongFeng/SongFeng_311/part_info/SongFeng_311.master.ply' \
            '/mnt/NAS/data/WenyuHan/sim2real/2022-11-15/SongFeng/SongFeng_318/part_info/SongFeng_318.master.ply' \
            '/mnt/NAS/data/WenyuHan/sim2real/2022-11-15/SongFeng/SongFeng_332/part_info/SongFeng_332.master.ply' \
            '/mnt/NAS/data/WenyuHan/sim2real/2022-11-15/Toyota/21092302/part_info/21092302.master.ply' \
            '/mnt/NAS/data/WenyuHan/sim2real/2022-11-15/ZSRobot/6010018CSV/part_info/6010018CSV.master.ply' \
            '/mnt/NAS/data/WenyuHan/sim2real/2022-11-15/ZSRobot/6010022CSV/part_info/6010022CSV.master.ply')

dir='/mnt/NAS/data/WenyuHan/sim2real/datasets_pose_estimation_yaoen/models/'

for i in ${!modelArray[@]}
do
    # echo "${modelArray[$i]}" 
    # echo $(printf "${dir}obj_%02d.ply" $((i+1))) 
    cp "${modelArray[$i]}" $(printf "${dir}obj_%02d.ply" $((i+1)))
done