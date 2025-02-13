for folder in */; do
  # Check if it's a directory
  if [ -d "$folder" ]; then
    # Remove trailing slash and create zip file
    folder_name=$(basename "$folder")
    unzip "${folder_name}.zip" -d "$folder_name"
    rm "${folder_name}.zip"
    echo "unzipped $folder_name"
  fi
done