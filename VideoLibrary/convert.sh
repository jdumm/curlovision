#!/bin/bash
# Example for converting a list of .mkv files into mp4:
for f in ~/Movies/P*/*.mkv
do 
  ffmpeg -i "$f" -c copy "${f%.mkv}.mp4"
  rm -f "$f"
done
