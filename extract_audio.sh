#!/bin/bash

set -e

# Default to ./videos and ./mp3 if not provided
VIDEO_DIR="${1:-videos}"
MP3_DIR="${2:-mp3}"

# Validate input directory
if [[ ! -d "$VIDEO_DIR" ]]; then
    echo "Error: Video directory '$VIDEO_DIR' does not exist."
    exit 1
fi

# Ensure output directory exists
mkdir -p "$MP3_DIR"

# Supported extensions (case-insensitive)
EXTENSIONS=("mp4" "mkv" "avi" "mov" "webm" "flv")

# Loop through and convert each video file
for ext in "${EXTENSIONS[@]}"; do
    while IFS= read -r -d '' video_file; do
        base_name="$(basename "$video_file")"
        filename="${base_name%.*}"
        output_file="$MP3_DIR/${filename}.mp3"

        if [[ -f "$output_file" ]]; then
            echo "Skipping (already exists): $output_file"
            continue
        fi

        echo "Converting: $video_file → $output_file"
        ffmpeg -i "$video_file" -q:a 0 -map a "$output_file"
    done < <(find "$VIDEO_DIR" -type f -iname "*.${ext}" -print0)
done

echo "✅ Conversion complete. MP3 files saved to '$MP3_DIR'"
