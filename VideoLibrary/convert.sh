#!/bin/bash
# Example one-liner for converting a list of .mkv files into mp4:
for f in ~/Movies/*/*.mkv; do ffmpeg -i "$f" -c copy "${f%.mkv}.mp4"; done
