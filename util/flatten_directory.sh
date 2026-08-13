#!/bin/bash

# This script flattens a directory structure by renaming files
# with their path and moving them to the top-level directory.

# --- Usage ---
# ./rename_script.sh <directory>
#
# <directory>: The target directory to flatten.

# Check if a directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

TOP_DIRECTORY="$1"

# Check if the provided argument is a valid directory
if [ ! -d "$TOP_DIRECTORY" ]; then
  echo "Error: Directory '$TOP_DIRECTORY' not found."
  exit 1
fi

# Use find to locate all files (and not directories)
# The -depth option processes the contents of a directory before the directory itself
find "$TOP_DIRECTORY" -depth -type f | while read -r FILEPATH; do
  # Continue to the next file if the path is the top-level directory
  if [ "$FILEPATH" == "$TOP_DIRECTORY" ]; then
    continue
  fi

  # Remove the top-level directory path from the beginning of the filepath
  RELATIVE_PATH="${FILEPATH#$TOP_DIRECTORY/}"

  # Replace all occurrences of '/' with '_' in the relative path
  NEW_FILENAME="${RELATIVE_PATH//\//_}"

  # Move the file to the top-level directory with its new name
  mv "$FILEPATH" "$TOP_DIRECTORY/$NEW_FILENAME"
done

# After moving all files, remove the now-empty subdirectories
# The -depth option is important here as well to remove nested directories from the inside out
find "$TOP_DIRECTORY" -mindepth 1 -type d -empty -delete

echo "All files have been moved and renamed to the top-level directory."
