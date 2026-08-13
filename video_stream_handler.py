"""Video stream handling utilities for Panopticon.

This module provides classes for managing video streams from various sources,
including real-time capture and dummy simulation for testing.

Classes:
    VideoStreamHandler: Handles real video streams using OpenCV.
    VideoStreamHandlerDummy: Provides a dummy implementation for testing.

Example:
    handler = VideoStreamHandler()
    if handler.connect('rtsp://example.com/stream'):
        ret, frame = handler.read_frame()
"""

import os
import queue
import threading
import time

import cv2
import numpy as np


class VideoStreamHandler:
    """Handles real video streams using OpenCV.

    This class manages connections to video sources like cameras, RTSP streams,
    or video files, providing methods to connect, disconnect, and read frames.
    """

    def __init__(self):
        self.capture = None
        self.source = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.read_thread = None
        self.is_local_file = False

    def connect(self, source_url):
        """Connects to the video source.

        Releases any existing capture before attempting a new connection.

        Args:
            source_url (str or int): The URL, file path, or camera index for the video source.

        Returns:
            bool: True if connection is successful, False otherwise.
        """
        self.disconnect()  # Ensure any previous stream is released

        capture_target = source_url
        self.is_local_file = False

        if isinstance(source_url, str):
            # coerce numeric strings to camera indexes
            if source_url.isdigit():
                try:
                    capture_target = int(source_url)
                except ValueError:
                    pass
            elif source_url.startswith("rtsp://"):
                # Force TCP for RTSP to avoid packet loss and artifacts (like POC errors)
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            else:
                # Check if it's a local file path; use absolute path to avoid cwd issues
                try:
                    abspath = os.path.abspath(source_url)
                except Exception:
                    abspath = source_url
                if os.path.isfile(abspath):
                    self.is_local_file = True
                    capture_target = abspath

        self.capture = cv2.VideoCapture(capture_target)
        if self.capture and self.capture.isOpened():
            # even if os.path.isfile failed earlier, use the capture properties
            # to decide whether we are really dealing with a file. some build of
            # OpenCV (or when the path has a scheme like "file://") will open a
            # local video without os.path.isfile reporting True, which would lead
            # us to treat the stream as an IP camera. in that case the playback
            # runs at full speed once and then the handler reports a frame-capture
            # error (the behaviour the user reported on Manjaro). detecting a
            # finite frame count is a reasonable heuristic that works on all
            # platforms.
            if not self.is_local_file:
                try:
                    frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    if frame_count > 0:
                        self.is_local_file = True
                except Exception:
                    # ignore failures querying properties
                    pass

            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.source = source_url

            # Start reading thread
            self.stop_event.clear()
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()

            return True
        else:
            if self.capture:  # Release if cv2.VideoCapture object was created but not opened
                self.capture.release()
            self.capture = None  # Ensure capture is None if not opened
            self.source = None
            self.is_local_file = False
            return False

    def disconnect(self):
        """
        Disconnects from the video source and releases the capture object.
        """
        if self.read_thread:
            self.stop_event.set()
            self.read_thread.join()
            self.read_thread = None

        # Clear queue
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()

        if self.capture:
            self.capture.release()
        self.capture = None
        self.source = None
        self.is_local_file = False

    def _read_loop(self):
        """Background loop to read frames from the video source."""

        # if we weren't flagged as a local file during connect, double-check
        # here in case the capture opened successfully and has a finite number
        # of frames. this guards against the problem where a filesystem path is
        # passed with a scheme ("file://"), or os.path.isfile returned False on
        # a remote-mounted drive. without this the handler thinks the file is an
        # IP stream, applies no per-frame delay and doesn't loop – exactly the
        # behaviour observed on the Manjaro machine.
        if not self.is_local_file and self.capture:
            try:
                total = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if total > 0:
                    self.is_local_file = True
            except Exception:
                pass

        frame_delay = 0.0
        if self.is_local_file and self.capture:
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            # Some OpenCV builds (or certain codecs) return 0.0 for FPS; fall
            # back to a reasonable default rather than playing at infinite
            # speed.
            if not (fps and fps > 0):
                fps = 30.0
            frame_delay = 1.0 / fps

        while not self.stop_event.is_set():
            if self.capture and self.capture.isOpened():
                ret, frame = self.capture.read()
                if ret:
                    # Keep only the latest frame
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put((ret, frame))

                    if frame_delay > 0:
                        time.sleep(frame_delay)
                else:
                    # If read fails and it's a local file we try to loop. some
                    # codecs/containers don't support seek; if the second attempt
                    # fails we fall through and an error will be raised so the UI
                    # can notify the user.
                    if self.is_local_file:
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.capture.read()
                        if ret:
                            if self.frame_queue.full():
                                try:
                                    self.frame_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            self.frame_queue.put((ret, frame))
                            if frame_delay > 0:
                                time.sleep(frame_delay)
                        else:
                            # give up – the file cannot be restarted
                            if self.frame_queue.full():
                                try:
                                    self.frame_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            self.frame_queue.put((ret, frame))
                            time.sleep(0.01)
                    else:
                        # For IP streams, signal that the stream has ended
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.frame_queue.put((ret, frame))
                        # Avoid busy loop if stream is broken
                        time.sleep(0.01)
            else:
                time.sleep(0.01)

    def is_opened(self):
        """Checks if the video stream is currently open.

        Args:
            None

        Returns:
            bool: True if open, False otherwise.
        """
        return self.capture is not None and self.capture.isOpened()

    def read_frame(self):
        """Reads a frame from the video stream.

        Args:
            None

        Returns:
            tuple: A tuple (bool, np.ndarray or None).
            The first element indicates success (stream is alive), and the second is the frame array if available.
            If the queue is empty, returns (True, None).
            If the stream has ended/failed, returns (False, None).
        """
        if not self.is_opened():
            return False, None

        try:
            # Non-blocking get to avoid freezing the UI
            return self.frame_queue.get_nowait()
        except queue.Empty:
            # Queue empty means no new frame, but stream is still open.
            return True, None

    def grab_frame(self):
        """Grabs the next frame from the video stream but does not decode it.

        Useful for clearing the buffer or skipping frames.

        Returns:
            bool: True if a frame was grabbed, False otherwise.
        """
        # Since the thread is continuously reading, we don't need to manually grab.
        return self.is_opened()

    def get_source(self):
        """Returns the source URL/path of the current video stream.

        Args:
            None

        Returns:
            str or None: The source URL/path, or None if not connected.
        """
        return self.source

    def seek_by_frames(self, delta_frames):
        """Seeks the video by a number of frames.

        Args:
            delta_frames (int): Number of frames to seek. Positive for forward, negative for backward.

        Returns:
            bool: True if seek was successful, False otherwise.
        """
        if not self.is_opened() or not self.is_local_file:
            return False

        # Stop the reading thread to avoid conflicts during seek
        if self.read_thread and self.read_thread.is_alive():
            self.stop_event.set()
            self.read_thread.join(timeout=1.0)  # Wait up to 1 second
            self.read_thread = None

        # Clear the frame queue
        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()

        # Perform the seek
        current_pos = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
        new_pos = current_pos + delta_frames
        total_frames = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
        new_pos = max(0, min(new_pos, total_frames - 1))
        success = self.capture.set(cv2.CAP_PROP_POS_FRAMES, new_pos)

        # Restart the reading thread
        if success:
            self.stop_event.clear()
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()

        return success


class VideoStreamHandlerDummy:
    """Provides a dummy implementation for testing video stream handling.

    This class simulates a video stream without actual hardware or network access,
    useful for development and testing purposes.
    """

    def __init__(self):
        self.source = "dummy_white_noise"
        self.width = 1920
        self.height = 1080
        self._is_connected = False

    def connect(self, source_url):
        """
        Simulates a successful connection to a video source.
        Always returns True.
        """
        self.source = source_url  # Store the requested source, even if dummy
        self._is_connected = True
        return True

    def disconnect(self):
        """
        Does nothing as there is no actual connection to disconnect.
        """
        self._is_connected = False

    def is_opened(self):
        """
        Checks if the dummy stream is currently open.
        Returns True if connected, False otherwise.
        """
        return self._is_connected

    def read_frame(self):
        """
        Generates a 1080p white noise frame (1920x1080, RGB, uint8).
        Returns (True, frame).
        """
        # Generate random noise for a 1080p RGB image (1920x1080x3)
        frame = np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
        return True, frame

    def grab_frame(self):
        """
        Simulates grabbing a frame.
        """
        return True

    def get_source(self):
        """
        Returns the dummy source URL/path.
        """
        return self.source

    def seek_by_frames(self, delta_frames):
        """
        Simulates seeking. Always returns True.
        """
        return True
