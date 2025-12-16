#!/bin/bash
# Script to process all result.html files in subdirectories
# Usage: ./process_results.sh [root_directory]

# Set root directory (default to current directory if not provided)
ROOT_DIR="${1:-.}"
ROOT_DIR=$(realpath "$ROOT_DIR")

# Check if root directory exists
if [ ! -d "$ROOT_DIR" ]; then
    echo "Error: Directory '$ROOT_DIR' does not exist"
    exit 1
fi

echo "Searching for result.html files in: $ROOT_DIR"
echo ""

count=0

# Store all files in an array first to avoid path issues
mapfile -t files < <(find "$ROOT_DIR" -mindepth 2 -type f -name "result.html")

for file in "${files[@]}"; do
    ((count++))
    # Short the file path for better readability
    short_path="${file#$ROOT_DIR/}"
    echo "[$count] Processing: $short_path"
    
    python utils/export_frames.py --input_file "$file" --save_video
    
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Failed to process $file (exit code: $exit_code)"
    fi
    
    echo ""
done

echo "=========================================="
echo "Total files processed: $count"
echo "=========================================="