import argparse
import os
import shutil
import subprocess
import sys

import av


def format_time(seconds):
    """Converts seconds to MM:SS.mmm string format."""
    if seconds is None:
        return "N/A"

    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    if h > 0:
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"
    else:
        return f"{int(m):02d}:{s:06.3f}"


def resolve_video_path(path):
    """
    Checks if the path exists. If not, tries appending common video extensions.
    """
    if os.path.exists(path):
        return path

    extensions = [".mp4", ".mkv", ".avi", ".mov", ".webm", ".ts", ".flv", ".mpg", ".m4v"]

    for ext in extensions:
        candidate = f"{path}{ext}"
        if os.path.exists(candidate):
            print(f"File found: '{candidate}'")
            return candidate

    return None


def find_iframe(video_path, target_index):
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]

        stream_start_seconds = 0.0
        if stream.start_time is not None:
            stream_start_seconds = float(stream.start_time * stream.time_base)

        keyframe_count = -1

        print(f"Searching for I-frame #{target_index + 1} (Internal Index {target_index}) in: {video_path}...")

        for packet in container.demux(stream):
            if packet.is_keyframe:
                keyframe_count += 1

                if keyframe_count == target_index:
                    raw_time = float(packet.pts * stream.time_base)
                    relative_time = raw_time - stream_start_seconds

                    container.close()
                    return raw_time, relative_time, keyframe_count

        container.close()
        return None, None, keyframe_count

    except Exception as e:
        print(f"Error reading video: {e}", file=sys.stderr)
        sys.exit(1)


def open_mpv(video_path, timestamp):
    """Opens MPV at the specific timestamp, paused."""
    if not shutil.which("mpv"):
        print("\n[Error] 'mpv' command not found. Is it installed and in your PATH?")
        return

    print(f"\nLaunching MPV at {timestamp:.3f}s...")

    cmd = ["mpv", f"--start={timestamp}", "--pause", video_path]

    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the timestamp of the ith I-frame (1-based).")
    parser.add_argument("video_path", help="Path to the video file (extension optional)")
    parser.add_argument("i", type=int, help="The 1-based number of the I-frame (e.g., 1 for the first one)")
    parser.add_argument("--open", action="store_true", help="Open the video in MPV at the found timestamp")

    args = parser.parse_args()

    # VALIDATION: Ensure 1-based index
    if args.i < 1:
        print("Error: The I-frame number must be 1 or greater.")
        sys.exit(1)

    # Convert 1-based input to 0-based index for internal processing
    internal_index = args.i - 1

    resolved_path = resolve_video_path(args.video_path)

    if not resolved_path:
        print(f"Error: Could not find file '{args.video_path}' or any common video variant.")
        sys.exit(1)

    raw_ts, rel_ts, total_found = find_iframe(resolved_path, internal_index)

    if raw_ts is not None:
        print("-" * 40)
        print(f"I-Frame Number: {args.i}")
        print(f"Raw Timestamp : {raw_ts:.6f} seconds (FFprobe PTS)")
        print(f"Relative Time : {rel_ts:.6f} seconds (Player Seek Time)")
        print(f"Formatted     : {format_time(rel_ts)}")
        print("-" * 40)

        if args.open:
            open_mpv(resolved_path, rel_ts)

    else:
        print(f"Error: I-Frame number {args.i} is out of bounds.")
        print(f"Video only has {total_found + 1} I-frames.")
