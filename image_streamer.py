"""This module provides HTTP-based image streaming functionality for real-time video feeds in the Panopticon application.

It includes classes for handling MJPEG streaming over HTTP, managing the streaming server, and a UI toggle button for controlling the stream.
"""

import http.server
import io
import socketserver
import threading

from PIL import Image
from PySide6.QtWidgets import QPushButton

# access it via http://localhost:12898/stream.mjpg
JPEG_QUALITY = 75  # JPEG compression quality (0-100), higher values reduce artifacts but increase bandwidth
PORT = 12898


class StreamingHandler(http.server.BaseHTTPRequestHandler):
    """HTTP request handler for MJPEG streaming.

    Handles GET requests for the /stream.mjpg endpoint, serving a multipart response
    with JPEG frames. Uses class-level variables for frame management and threading control.
    """

    frame_lock = threading.Lock()
    frame = None  # Stores the latest JPEG frame bytes
    stop_event = threading.Event()

    @classmethod
    def update_frame_bytes(cls, frame_bytes):
        with cls.frame_lock:
            cls.frame = frame_bytes

    @classmethod
    def clear_frame(cls):
        with cls.frame_lock:
            cls.frame = None

    @classmethod
    def reset_stop_event(cls):
        cls.stop_event.clear()

    @classmethod
    def set_stop_event(cls):
        cls.stop_event.set()

    def do_GET(self):
        """Handle GET requests for the streaming endpoint.

        Serves MJPEG stream at /stream.mjpg. For other paths, returns 404.
        """
        if self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-type", "multipart/x-mixed-replace; boundary=--FRAME")
            self.end_headers()
            try:
                while not StreamingHandler.stop_event.is_set():
                    with StreamingHandler.frame_lock:
                        img_bytes = StreamingHandler.frame

                    if img_bytes:
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-type", "image/jpeg")
                        self.send_header("Content-length", str(len(img_bytes)))
                        self.end_headers()
                        self.wfile.write(img_bytes)
                        self.wfile.write(b"\r\n")

                    # Wait for a short period or until a new frame is available
                    # This can be adjusted based on desired responsiveness vs CPU usage
                    StreamingHandler.stop_event.wait(0.03)  # Approx 30 FPS, or until stop_event is set
            except ConnectionAbortedError:
                print("Streaming client disconnected (ConnectionAbortedError).")
            except BrokenPipeError:
                print("Streaming client disconnected (BrokenPipeError).")
            except Exception as e:
                if not StreamingHandler.stop_event.is_set():  # Avoid error message if server is stopping
                    print(f"Streaming error: {e}")
            finally:
                print("Closing client stream connection.")
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Resource not found")


class ImageStreamer:
    """Server for hosting HTTP-based image streams.

    Manages a threaded TCP server for MJPEG streaming, allowing real-time image updates.
    """

    def __init__(self, host="0.0.0.0", port=PORT):
        """Initialize the image streamer.

        Args:
            host (str): Host address to bind the server to. Defaults to "0.0.0.0".
            port (int): Port number for the server. Defaults to PORT.
        """
        self.host = host
        self.port = port
        self.server = None
        self.server_thread = None
        self._is_running = False

    def start(self):
        """Start the streaming server.

        Returns:
            bool: True if started successfully, False otherwise.
        """
        if self.server:
            print("Streamer is already running.")
            return True  # Or False, depending on desired behavior
        try:
            StreamingHandler.reset_stop_event()  # Ensure stop event is clear before starting
            StreamingHandler.clear_frame()  # Clear any old frame
            self.server = socketserver.ThreadingTCPServer((self.host, self.port), StreamingHandler)
            self.server.daemon_threads = True  # Ensure threads exit when main program exits
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            self._is_running = True
            print(f"Image streamer started on http://{self.host}:{self.port}/stream.mjpg")
            return True
        except OSError as e:  # Catch specific errors like "address already in use"
            print(f"Error starting image streamer (OSError): {e}")
            self.server = None
            self._is_running = False
            return False
        except Exception as e:
            print(f"Generic error starting image streamer: {e}")
            self.server = None
            self._is_running = False
            return False

    def stop(self):
        """Stop the streaming server and clean up resources."""
        if self.server:
            print("Stopping image streamer...")
            StreamingHandler.set_stop_event()  # Signal handler threads to stop
            self.server.shutdown()  # Stop serve_forever loop
            self.server.server_close()  # Release the port
            if self.server_thread and self.server_thread.is_alive():
                self.server_thread.join(timeout=2)  # Wait for thread to finish
            self.server = None
            self.server_thread = None
            StreamingHandler.clear_frame()  # Clear the last frame
            self._is_running = False
            print("Image streamer stopped.")
        else:
            print("Streamer is not running.")

    def change_port(self, new_port):
        """Change the server port.

        Stops the server if running, updates the port, and optionally restarts.

        Args:
            new_port (int): New port number.
        """
        was_running = self.is_running()
        if was_running:
            print(f"Server is running on port {self.port}. Stopping server to change port.")
            self.stop()

        old_port = self.port
        self.port = new_port
        print(f"Image streamer port changed from {old_port} to {self.port}.")

        if was_running:
            print("Server was stopped to change the port. Please start it again manually if needed.")

    def update_frame(self, image_array):
        """Update the current frame for streaming.

        Converts the input image array to JPEG and updates the shared frame buffer.

        Args:
            image_array (numpy.ndarray): RGB image array to stream.
        """
        if not self._is_running or image_array is None:
            return
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image_array)
            # Convert PIL Image to JPEG bytes
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format="JPEG", quality=JPEG_QUALITY)
            jpeg_bytes = img_byte_arr.getvalue()
            StreamingHandler.update_frame_bytes(jpeg_bytes)
        except Exception as e:
            print(f"Error updating frame for streaming: {e}")

    def is_running(self):
        """Check if the streamer is currently running.

        Returns:
            bool: True if running, False otherwise.
        """
        return self._is_running

    def __del__(self):
        """Ensure server is stopped when ImageStreamer instance is deleted."""
        # Ensure server is stopped when ImageStreamer instance is deleted
        if self.is_running():
            self.stop()


class StreamToggleButton(QPushButton):
    """UI toggle button for controlling image streaming.

    Connects to an ImageStreamer instance and provides start/stop functionality
    with status updates.
    """

    def __init__(self, image_streamer, tr_function, status_bar_show_message_callback, parent=None):
        """Initialize the stream toggle button.

        Args:
            image_streamer (ImageStreamer): The streamer to control.
            tr_function (callable): Translation function for UI text.
            status_bar_show_message_callback (callable): Callback to show status messages.
            parent (QWidget): Parent widget.
        """
        super().__init__(parent)
        self.image_streamer = image_streamer
        self.tr = tr_function
        self.show_status_message = status_bar_show_message_callback

        self.setCheckable(True)
        self.clicked.connect(self._handle_stream_toggle)
        self.retranslate_ui()  # Set initial text

    def _handle_stream_toggle(self):
        """Handle toggle button clicks to start or stop streaming."""
        if self.isChecked():
            if self.image_streamer.start():
                self.setText(self.tr("stop_stream_button"))
                self.show_status_message(self.tr("status_stream_started"), 3000)
            else:
                self.setChecked(False)  # Keep it unchecked if start failed
                self.show_status_message(self.tr("status_stream_start_failed"), 5000)
        else:
            self.image_streamer.stop()
            self.setText(self.tr("start_stream_button"))
            self.show_status_message(self.tr("status_stream_stopped"), 3000)

    def retranslate_ui(self):
        """Update UI text based on current language settings."""
        if self.isChecked():
            self.setText(self.tr("stop_stream_button"))
        else:
            self.setText(self.tr("start_stream_button"))
