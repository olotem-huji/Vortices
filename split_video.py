#!/usr/bin/env python3
"""
Extract Video Frames - Extract frames from a video and save them to a directory
"""

import cv2
import os
from pathlib import Path


def extract_frames(video_path, output_dir, frame_interval=1, start_time=0, end_time=None, image_format='jpg',
                   grayscale=False, rotate=0):
    """
    Extract frames from a video and save them to a directory

    Args:
        video_path (str): Path to the input video file
        output_dir (str): Path to the output directory for frames
        frame_interval (int): Extract every Nth frame (default: 1 for all frames)
        start_time (float): Start time in seconds (default: 0)
        end_time (float): End time in seconds (default: None for full video)
        image_format (str): Image format for output files (default: 'jpg')
        grayscale (bool): Convert frames to grayscale (black and white) (default: False)
        rotate (int): Clockwise rotation angle: 0, 90, 180, or 270 degrees (default: 0)
    """

    # --- Initial Checks and Setup ---
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Validate rotate argument
    if rotate not in [0, 90, 180, 270]:
        raise ValueError("Rotation angle must be 0, 90, 180, or 270 degrees.")

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # --- Get Video Properties ---
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Resolution: {width}x{height}")

    # --- Calculate Frame Range ---
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time is not None else total_frames
    end_frame = min(end_frame, total_frames)

    print(f"Extracting frames from {start_time:.2f}s to {end_frame / fps:.2f}s")
    print(f"Frame interval: {frame_interval}")
    print(f"Output directory: {output_dir}")
    print(f"Image format: {image_format}")
    print(f"Grayscale conversion: {grayscale}")
    print(f"Rotation: {rotate} degrees clockwise")

    # --- Frame Extraction Loop ---
    frame_count = 0
    saved_count = 0

    # Set starting position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        current_time = current_frame / fps

        # Check if we've reached the end time
        if end_time is not None and current_time >= end_time:
            break

        # Extract every Nth frame
        if frame_count % frame_interval == 0:

            # 1. Apply Grayscale conversion
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 2. Apply Rotation
            if rotate == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotate == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotate == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 270 clockwise is 90 counter-clockwise

            # Generate filename
            frame_filename = f"frame_{current_frame:06d}_{current_time:.2f}s.{image_format}"
            frame_path = os.path.join(output_dir, frame_filename)

            # Save frame
            success = cv2.imwrite(frame_path, frame)

            if success:
                saved_count += 1
                if saved_count % 100 == 0:  # Print progress every 100 frames
                    print(f"Saved {saved_count} frames... (time: {current_time:.2f}s)")
            else:
                print(f"Failed to save frame {current_frame}")

        frame_count += 1

    # Release video capture
    cap.release()

    print(f"\nExtraction complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Frames saved: {saved_count}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    # --- Example Usage with New Options ---
    # Example 1: Extract all frames, black and white, rotated 90 degrees
    # extract_frames(
    #     video_path=r"C:\Physics\Year 2\Lab\Advanced Lab 2B\21.05\mid_600 RPM.mp4",
    #     output_dir="frames_bw_90deg",
    #     grayscale=True,
    #     rotate=90
    # )

    # Example 2: Original settings
    extract_frames(
        video_path=r"C:\Physics\Year 2\Lab\Advanced Lab 2B\21.05\long_337 RPM.mp4",
        output_dir="frames_all",
        frame_interval=1,
        start_time=0,
        end_time=None,
        image_format="jpg",
        grayscale=False,
        rotate=90
    )