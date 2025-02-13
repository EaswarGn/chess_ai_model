#!/bin/bash

# Loop through all .zip files in the current directory
for file in *.zip; do
  # Check if file exists (in case no .zip files are found)
  [ -e "$file" ] || continue

  # Remove file extension to get folder name
  folder_name="${file%.*}"

  # Unzip into the folder
  unzip "$file" -d "$folder_name"

  # Remove the original zip file after extraction
  rm "$file"

  echo "Unzipped $file into $folder_name"
done
