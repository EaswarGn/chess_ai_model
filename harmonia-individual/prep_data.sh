#!/bin/bash

# Define username variable
username="blitzking45"

# Run Python script with the username variable
python get_games.py --username="$username" --train_output="${username}_train.pgn" --validation_output="${username}_validation.pgn" --train_ratio=0.9

# Run pgn-extract with the username variable
pgn-extract -Wlalg -llog.txt -w100000 --fencomments --nobadresults -V -N --plycount --fixtagstrings -bl10 -o"processed_${username}_train.pgn" "${username}_train.pgn" || { echo "pgn-extract failed"; exit 1; }
pgn-extract -Wlalg -llog.txt -w100000 --fencomments --nobadresults -V -N --plycount --fixtagstrings -bl10 -o"processed_${username}_validation.pgn" "${username}_validation.pgn" || { echo "pgn-extract failed"; exit 1; }

mkdir "${username}_data"
cd "${username}_data"
pgn_parser --pgn_file="../processed_${username}_train.pgn" --chunks_per_file=25000 --max_files_per_dir=10 --outputdir="${username}_train_chunks"
pgn_parser --pgn_file="../processed_${username}_validation.pgn" --chunks_per_file=25000 --max_files_per_dir=10 --outputdir="${username}_validation_chunks"

rm "../${username}_train.pgn"
rm "../${username}_validation.pgn"
rm "../processed_${username}_train.pgn"
rm "../processed_${username}_validation.pgn"