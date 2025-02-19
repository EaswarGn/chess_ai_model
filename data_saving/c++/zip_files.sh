
cd /Volumes/Lexar/1900_training_chunks

for folder in */; do
  # Check if it's a directory
  if [ -d "$folder" ]; then
    # Remove trailing slash and create zip file
    folder_name=$(basename "$folder")
    zip -r "/Volumes/Lexar/1900_training_zipped/1900_zipped_training_chunks/${folder_name}.zip" "$folder_name"
    echo "Zipped $folder_name"
  fi
done