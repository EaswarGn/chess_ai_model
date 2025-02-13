
pgn_dir="/Volumes/Lexar/pgns_ranged_testing"
dest="/users/easwar/Downloads"
usb_drive_output_dir="/Volumes/Lexar/ranged_testing_chunks"

# Navigate to the source directory
cd "$pgn_dir" || { echo "Failed to cd to $pgn_dir"; exit 1; }

for dir in */; do
    folder_name="${dir%/}"
    cd "$dir"
    echo "${folder_name}"
    # Process each .pgn file
    for pgn_file in *.pgn; do
        base_name=$(basename "$pgn_file" .pgn)
        echo "Processing ${pgn_file}..."

        # Copy the .pgn file to the destination directory
        cp -r "$pgn_file" "$dest" || { echo "Failed to copy $pgn_file"; exit 1; }

        # Navigate to the destination directory
        cd "$dest" || { echo "Failed to cd to $dest"; exit 1; }

        # Run pgn-extract on the copied file
        pgn-extract -Wlalg -llog.txt -w100000 --fencomments --nobadresults -V -N --plycount --fixtagstrings -bl10 -o"${base_name}_processed.pgn" "${base_name}.pgn" || { echo "pgn-extract failed"; exit 1; }

        # Run pgn_parser on the processed file
        pgn_parser --pgn_file="${base_name}_processed.pgn" --chunks_per_file=100000 --max_files_per_dir=10 --outputdir="${base_name}_chunks" || { echo "pgn_parser failed"; exit 1; }

        # Move the output directory to the USB drive
        mv "${base_name}_chunks" "${usb_drive_output_dir}/${folder_name}" || { echo "Failed to move ${base_name}_chunks"; exit 1; }

        # Clean up the copied files
        rm "${base_name}.pgn"
        rm "${base_name}_processed.pgn"

        
        # Return to the source directory for the next file
        cd "${pgn_dir}/${folder_name}" || { echo "Failed to cd back to ${pgn_dir}/${folder_name}"; exit 1; }
        #pwd
        #ls
    done
    cd "${pgn_dir}"
done


