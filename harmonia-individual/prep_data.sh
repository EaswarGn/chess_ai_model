#!/bin/bash

# Default values
username=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --username=*)
            username="${1#*=}"
            shift  # Move past the --username= argument
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check if username is provided
if [ -z "$username" ]; then
    echo "Error: --username argument is required"
    exit 1
fi

echo "Processing games for: $username"

# Run Python script with the username variable
python get_games.py --username="$username" --train_output="${username}_train.pgn" --validation_output="${username}_validation.pgn" --train_ratio=0.9

# Run pgn-extract with the username variable
pgn-extract -Wlalg -llog.txt -w100000 --fencomments --nobadresults -V -N --plycount --fixtagstrings -bl10 -o"processed_${username}_train.pgn" "${username}_train.pgn" || { echo "pgn-extract failed"; exit 1; }
pgn-extract -Wlalg -llog.txt -w100000 --fencomments --nobadresults -V -N --plycount --fixtagstrings -bl10 -o"processed_${username}_validation.pgn" "${username}_validation.pgn" || { echo "pgn-extract failed"; exit 1; }

mkdir "${username}_data"
cd "${username}_data"
individual_pgn_parser --pgn_file="../processed_${username}_train.pgn" --chunks_per_file=25000 --max_files_per_dir=10 --outputdir="${username}_train_chunks"
individual_pgn_parser --pgn_file="../processed_${username}_validation.pgn" --chunks_per_file=25000 --max_files_per_dir=10 --outputdir="${username}_validation_chunks"

rm "../${username}_train.pgn"
rm "../${username}_validation.pgn"
rm "../processed_${username}_train.pgn"
rm "../processed_${username}_validation.pgn"
rm ../logs.txt