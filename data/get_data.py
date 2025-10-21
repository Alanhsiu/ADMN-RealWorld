import pyrealsense2 as rs
import numpy as np
import cv2
import os # Added for path and folder creation

# --- CONFIGURATION ---
# Define the folder where images will be saved.
# **IMPORTANT:** Change this to an existing path on your system, e.g., 'C:/Users/YourName/Desktop/CameraCaptures/'
SAVE_FOLDER = '~/Data/simple_raw_data/' 
# Counter for file naming
image_count = 0

# Ensure the save folder exists
if not os.path.exists(SAVE_FOLDER):
    try:
        os.makedirs(SAVE_FOLDER)
        print(f"Save folder created: {SAVE_FOLDER}")
    except OSError as e:
        print(f"Error creating directory {SAVE_FOLDER}: {e}")
        # Exit if folder creation fails due to permissions or other issues
        exit()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Enable streams (adjust resolution, format, and FPS as needed)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent set of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (for visualization/saving)
        # Note: 'depth_colormap' is BGR, suitable for saving as PNG/JPG
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Display the images
        cv2.imshow('Color Image', color_image)
        cv2.imshow('Depth Colormap', depth_colormap)

        # --- KEY PRESS HANDLING AND IMAGE SAVING LOGIC ---
        # Get key press input. cv2.waitKey(1) returns -1 (or 255 after & 0xFF) if no key is pressed.
        key = cv2.waitKey(1) & 0xFF

        # Check for 'q' key press to break loop
        if key == ord('q'):
            break

        # Check for ANY OTHER key press (key != 255 means a key was pressed)
        elif key != 255:
            # Increment the counter
            image_count += 1
            
            # Create file names (e.g., color_image_0001.png)
            color_filename = f"{SAVE_FOLDER}color_image_{image_count:04d}.png"
            depth_filename = f"{SAVE_FOLDER}depth_image_{image_count:04d}.png"
            
            # Save the images
            # Saving color_image (raw BGR)
            cv2.imwrite(color_filename, color_image)
            print(f"Saved Color Image to: {color_filename}")
            
            # Saving depth_colormap (visualization)
            cv2.imwrite(depth_filename, depth_colormap) 
            print(f"Saved Depth Colormap to: {depth_filename}")
            
            # Optional: Add a brief visual confirmation here if desired
            # e.g., print(f"Capture {image_count} complete!")

finally:
    # Stop streaming and close all windows
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Streaming stopped and windows closed.")