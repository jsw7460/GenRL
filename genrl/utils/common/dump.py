import cv2
import numpy as np


def dump_video(video: np.ndarray, path: str) -> None:
    """
    :param video: [n_frame, x, y, c]
    :param path:
    :return:
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    resolution = video.shape[1: -1]
    if not path.endswith(".mp4"):
        path = path + ".mp4"

    out = cv2.VideoWriter(path, fourcc, 30, resolution)
    for frame in video:
        out.write(frame)

    out.release()

    print(f"Video is saved at {path}")


def dump_text(text: str, path: str) -> None:
    with open(path, "w") as f:
        f.write(text)