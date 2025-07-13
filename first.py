import glob
import imageio
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Folder containing the videos
video_dir = r"C:\Users\dylan\mask_videos\SC-1001"

# Get all MKV files (you can add other extensions too if needed)
video_paths = glob.glob(video_dir + r"\*.mkv")

# OpenCV VideoCapture objects
caps = [cv2.VideoCapture(p) for p in video_paths]

while True:
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert from BGR to RGB for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)

    if len(frames) != len(caps):
        break  # one of the videos ended

    # Resize frames to the same height
    height = min(f.shape[0] for f in frames)
    resized = [cv2.resize(f, (int(f.shape[1] * height / f.shape[0]), height)) for f in frames]

    # Concatenate frames horizontally
    combined = np.hstack(resized)

    # Convert back to BGR for OpenCV display
    bgr_display = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    cv2.imshow("Multi-view RGB Videos", bgr_display)

    # Press Esc to quit
    if cv2.waitKey(1) == 27:
        break

# Release everything
for cap in caps:
    cap.release()
cv2.destroyAllWindows()