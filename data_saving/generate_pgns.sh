#!/bin/bash

#Read the raw pgns from lichess and filter out the elo ranges we care about

mkdir data/pgns_ranged_filtered/
for i in {1000..2000..100}; do
    echo $i
    upperval=$(($i + 100))
    outputdir="data/pgns_ranged_filtered/${i}"
    mkdir $outputdir
    for f in data/lichess_raw/lichess_db_standard_rated_2017* data/lichess_raw/lichess_db_standard_rated_2018* data/lichess_raw/lichess_db_standard_rated_2019-{01..12}.pgn.zst; do
        fname="$(basename -- $f)"
        echo "${i}-${fname}"
        python3 extractELOrange.py --remove_bullet ${i} ${upperval} ${outputdir}/${fname} ${f} > ${outputdir}/${fname}.log 2>&1 &
    done
    echo "waiting for python commands to finish"
    wait
    echo "python commands finished!"
done
echo "waiting for python commands to finish"
wait
echo "python commands finished!"

echo "data/pgns_ranged_filtered folder is finished!"
exit 0

# You have to wait for the screens to finish to do this
# We use pgn-extract to normalize the games and prepare for preprocessing
# This also creates blocks of 200,000 games which are useful for the next step

mkdir data/pgns_ranged_blocks
for i in {1000..2000..100}; do
    echo $i
    cw=`pwd`
    outputdir="data/pgns_ranged_blocks/${i}"
    mkdir $outputdir
    cd $outputdir
    for y in {2017..2019}; do
        echo "${i}-${y}"
        mkdir $y
        cd $y
        #first folder: /data/pgns_ranged_blocks/1000/2017
        screen -S "${i}-${y}" -dm bash -c "source ~/.bashrc; zstd -d \"../../../pgns_ranged_filtered/${i}/lichess_db_standard_rated_${y}\"* | ../../../../pgn-extract.exe -7 -C -N  -#200000"
        cd ..
    done
    echo "waiting for screens to finish"
    wait
    echo "screens finished!"
    cd $cw
done

#Now we have all the pgns in blocks we can randomly sample and create testing and training sets of 60 and 3 blocks respectively
python3 move_training_set.py