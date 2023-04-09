import os
import time
from pathlib import Path
import av


BASE_DIR = os.getcwd()

print(BASE_DIR)

def read_mp4(file_directory):
    pil_image_list = []
    input_container = av.open(file_directory)
    # Iterate over all the video frames
    for packet in input_container.demux():
        for frame in packet.decode():
            # Process the video frame here
            pil_image_list.append(frame.to_image())
            # print(frame.to_image())
    return pil_image_list

if __name__ == "__main__":
    video_directory = os.path.join(BASE_DIR, "Videos", "vid.mp4")
    read_mp4(video_directory)