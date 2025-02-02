: '#!/bin/bash

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
exit 0'

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
        #screen -S "${i}-${y}" -dm bash -c "source ~/.bashrc; zstd -d \"../../../pgns_ranged_filtered/${i}/lichess_db_standard_rated_${y}\"* | ../../../../pgn-extract.exe -7 -C -N  -#200000"
        for zst_file in ../../../pgns_ranged_filtered/${i}/lichess_db_standard_rated_${y}-*.pgn.zst; do
            echo "Decompressing $zst_file"
            base_name=$(basename "$zst_file" .zst)
            zstd -d "$zst_file" -o "./$base_name" &
        done
        wait 
        echo "zst files for $y have been decompressed"

        if [ ! -d "logs" ]; then
            mkdir logs
        fi

        start_index=1
        for pgn_file in lichess_db_standard_rated_${y}-*.pgn; do
            echo "processing $pgn_file with pgn-extract"
            { pgn-extract -llogs/log.txt -N --commented -#"200000,$start_index" "$pgn_file" && rm -f "$pgn_file"; } &
            start_index=$((start_index + 10))     
        done
        echo "waiting for pgn-extract to finish"
        wait
        echo "pgn-extract finished!"

        cd ..
    done
    echo "waiting for any background processes to finish"
    wait
    echo "All background processes finished!"
    
    cd $cw
done

echo "waiting for python commands to finish"
wait
echo "python commands finished!"

#Now we have all the pgns in blocks we can randomly sample and create testing and training sets of 60 and 3 blocks respectively
python3 move_training_set.py

mkdir data/h5s_ranged_training
for i in {1200..1800..200}; do
    echo $i
    cw=`pwd`
    outputdir="data/h5s_ranged_training/${i}"
    mkdir $outputdir
    cd $outputdir

    for pgn_file in ../../pgns_ranged_training/${i}/*.pgn; do
        base_name=$(basename "$pgn_file" .pgn)
        echo "processing ${pgn_file}"
        python3 pgn_to_h5.py "$pgn_file" "${base_name}.h5" &
    done

    echo "waiting for h5 files to finish"
    wait
    echo "h5 files finished!"

done


outputdir="data/h5s_ranged_training/1800"
mkdir $outputdir

cw=`pwd`
cd data/pgns_ranged_training/1800
mkdir logs
for pgn_file in *.pgn; do
    base_name=$(basename "$pgn_file" .pgn)
    echo "processing ${pgn_file}"
    pgn-extract -Wlalg -bl10 -llogs/log.txt -w10000 --fencomments --commentlines -N --commented -otemp.pgn "$pgn_file"
    rm $pgn_file
    mv temp.pgn "${base_name}.pgn"

    cd $cw

    cd data/h5s_ranged_training/1800
    echo "writing to h5 file"
    python3 ../../../pgn_to_h5.py "../../pgns_ranged_training/1800/${base_name}.pgn" "${base_name}.h5"

    cd $cw
    cd data/pgns_ranged_training/1800
    
done
cd $cw

    