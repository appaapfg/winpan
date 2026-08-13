#!/bin/sh

"""
Frame Extractor Script.

Description:
    Extracts I-frames (keyframes) from one or more input video files using ffmpeg.
    All extracted keyframes are written into a single directory. For each input video file,
    frames are saved as JPEG images using the pattern: <output_dir>/<basename>_00001.jpg, etc.,
    where <basename> is the filename of the source video without its extension.

Args:
    --input_files: One or more video file paths to process.
    --output_dir: Directory for output keyframes (default: 'keyframes').

Examples:
    ./frame_extractor.sh --input_files video1.mp4 video2.mp4 --output_dir my_frames

Notes:
    Requires ffmpeg with CUDA support for hardware acceleration.
    Frames are extracted only for I-frames (keyframes).
    If ffmpeg fails, check installation and input file validity.
"""

OUTPUT_DIR="keyframes" # Default output directory
INPUT_FILES=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --input_files)
            shift
            while [[ "$1" != --* ]] && [[ "$#" -gt 0 ]]; do
                INPUT_FILES="$INPUT_FILES $1"
                shift
            done
            ;;
        --output_dir)
            shift
            OUTPUT_DIR="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --input_files <video_file1> [video_file2 ...] --output_dir <directory>"
            exit 1
            ;;
    esac
done

if [ -z "$INPUT_FILES" ]; then
	echo "Error: No input files specified."
	echo "Usage: $0 --input_files <video_file1> [video_file2 ...] --output_dir <directory>"
	exit 1
fi

mkdir -p "$OUTPUT_DIR"

for INPUT_FILE in $INPUT_FILES; do
	BASENAME=$(basename "$INPUT_FILE" | sed 's/\.[^.]*$//')
	ffmpeg --hwaccel cuda -i "$INPUT_FILE" -vf "select=eq(pict_type\\,I)" -q:v 2 -vsync vfr "${OUTPUT_DIR}/${BASENAME}_%05d.jpg"
	# ffmpeg -i "$INPUT_FILE" -vf "select=eq(pict_type\\,I)" -q:v 2 -vsync vfr "keyframes/${BASENAME}_%05d.jpg"
done
