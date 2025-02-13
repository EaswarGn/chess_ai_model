
cd /Volumes/Lexar/chessmodel_dataset/ranged_testing_chunks

for folder in */; do
  # Check if it's a directory
  if [ -d "$folder" ]; then
    # Remove trailing slash and create zip file
    folder_name=$(basename "$folder")
    zip -r "/Volumes/Lexar/zipped_testing/${folder_name}.zip" "$folder_name"
    echo "Zipped $folder_name"
  fi
done