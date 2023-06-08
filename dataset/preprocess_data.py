"""
Modified from https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py
"""

import os
import io
import cv2
import zlib
import hydra
import struct
import numpy as np
from PIL import Image
from os import cpu_count
from functools import partial
from tqdm.contrib.concurrent import process_map


class ByteDataReader:
    def __init__(self, byte_data):
        self.pointer = 0
        self.byte_data = byte_data

    def read_next(self, offset):
        data = self.byte_data[self.pointer:self.pointer+offset]
        self.pointer += offset
        return data

    def seek_next(self, offset):
        self.pointer += offset


def process_one_scene(scene_id, cfg):
    input_file_path = os.path.join(cfg.data.raw_scene_path, scene_id, f"{scene_id}.sens")
    output_root_dir_path = os.path.join(cfg.data.metadata.video_frames_path, scene_id)
    rgb_image_size = cfg.data.rgb_image_size
    depth_image_size = cfg.data.depth_image_size
    raw_file_data = np.fromfile(input_file_path, dtype="b")
    data_reader = ByteDataReader(raw_file_data)
    data_reader.seek_next(4)
    strlen = struct.unpack('Q', data_reader.read_next(8))[0]
    data_reader.seek_next(strlen + 272)
    depth_width = struct.unpack('I', data_reader.read_next(4))[0]
    depth_height = struct.unpack('I', data_reader.read_next(4))[0]
    data_reader.seek_next(4)
    num_frames = struct.unpack('Q', data_reader.read_next(8))[0]

    for i in range(num_frames):
        is_sampled_frame = i % cfg.data.frames_sample == 0
        if cfg.data.extraction_output.pose_txt and is_sampled_frame:
            # read pose byte data
            camera_to_world = np.asarray(struct.unpack('f' * 16, data_reader.read_next(64)), dtype=np.float32).reshape(4, 4)
            pose_output_path = os.path.join(output_root_dir_path, "pose")
            os.makedirs(pose_output_path, exist_ok=True)
            with open(os.path.join(pose_output_path, f"{i}.txt"), "w") as f:
                for line in camera_to_world:
                    np.savetxt(f, line[np.newaxis], fmt="%f")
        else:
            data_reader.seek_next(64)
        data_reader.seek_next(16)
        color_size_bytes = struct.unpack('Q', data_reader.read_next(8))[0]
        depth_size_bytes = struct.unpack('Q', data_reader.read_next(8))[0]
        if cfg.data.extraction_output.color_image and is_sampled_frame:
            # read rgb byte data
            color_data = b''.join(struct.unpack('c' * color_size_bytes, data_reader.read_next(color_size_bytes)))

            # parse rgb data to image and save
            rgb_image = Image.open(io.BytesIO(color_data))
            rgb_image = cv2.resize(np.asarray(rgb_image), (rgb_image_size[0], rgb_image_size[1]), interpolation=cv2.INTER_NEAREST)
            rgb_output_path = os.path.join(output_root_dir_path, "color")
            os.makedirs(rgb_output_path, exist_ok=True)
            Image.fromarray(rgb_image).save(os.path.join(rgb_output_path, f"{i}.jpg"))
        else:
            data_reader.seek_next(color_size_bytes)

        if cfg.data.extraction_output.depth_image and is_sampled_frame:
            # read depth byte data
            depth_data = b''.join(struct.unpack('c' * depth_size_bytes, data_reader.read_next(depth_size_bytes)))
            # parse depth data to image and save
            depth_data = zlib.decompress(depth_data)
            depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(depth_height, depth_width)
            depth = cv2.resize(depth, (depth_image_size[0], depth_image_size[1]), interpolation=cv2.INTER_NEAREST)
            depth_output_path = os.path.join(output_root_dir_path, "depth")
            os.makedirs(depth_output_path, exist_ok=True)
            depth_img = Image.new("I", depth.T.shape)
            depth_img.frombytes(depth.tobytes(), "raw", "I;16")
            depth_img.save(os.path.join(depth_output_path, f"{i}.png"))

        else:
            data_reader.seek_next(depth_size_bytes)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    max_workers = cpu_count() if "workers" not in cfg else cfg.workers
    print(f"\nUsing {max_workers} CPU threads.")
    with open(cfg.data.metadata.scene_ids, "r") as f:
        scene_ids = [line.strip() for line in f]
    process_map(partial(process_one_scene, cfg=cfg), scene_ids, chunksize=1, max_workers=max_workers)
    print(f"==> Complete. Saved at: {os.path.abspath(cfg.data.metadata.video_frames_path)}\n")


if __name__ == '__main__':
    main()
