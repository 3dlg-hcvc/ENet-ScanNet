"""
Modified from https://github.com/daveredrum/D3Net/blob/main/data/scannet/compute_multiview_features.py
"""

import os
import h5py
import hydra
import torch
from tqdm import tqdm
from model.enet import create_enet
from torchvision.io import read_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class ScanNet(Dataset):
    def __init__(self, cfg, scene_ids):
        self.video_frames_path = cfg.data.metadata.video_frames_path
        self.frames_sample = cfg.data.frames_sample
        self.data = []
        for scene_id in scene_ids:
            rgb_img_path = os.path.join(self.video_frames_path, scene_id, "color")
            frame_list = sorted(os.listdir(rgb_img_path), key=lambda x: int(x.split(".")[0]))
            for frame_file in frame_list:
                self.data.append({"scene_id": scene_id, "frame_id": int(frame_file.split(".")[0]) // self.frames_sample})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.video_frames_path, self.data[idx]["scene_id"], "color", f"{self.data[idx]['frame_id'] * self.frames_sample}.jpg"
        )
        image = read_image(img_path)
        image = transforms.Normalize(
            mean=[0.496342, 0.466664, 0.440796], std=[0.277856, 0.28623, 0.291129]
        )(image.to(torch.float32) / 255)
        return {"scene_id": self.data[idx]["scene_id"], "frame_id": self.data[idx]["frame_id"], "img": image}


def get_model(cfg):
    model = create_enet()
    checkpoint = torch.load(cfg.ckpt_path)
    checkpoint.pop("26.0.weight")  # remove the last classifier layer
    model.load_state_dict(checkpoint)
    for param in model.parameters():
        param.requires_grad = False
    return model


@torch.no_grad()
def do_inference(model, dataloader, h5_file):
    model.eval()
    for data_dict in tqdm(dataloader, desc="Running inference"):
        features = model(data_dict["img"].to("cuda"))
        batch_size = data_dict["img"].shape[0]
        # save features
        for batch_id in range(batch_size):
            scene_id = data_dict["scene_id"][batch_id]
            frame_id = data_dict["frame_id"][batch_id].item()
            h5_file[scene_id][frame_id] = features[batch_id].cpu().numpy()
    h5_file.close()


def initialize_output_hdf5(cfg, scene_ids):
    os.makedirs(cfg.output_root_dir, exist_ok=True)
    output_path = os.path.join(cfg.enet_feature_output_path)
    f = h5py.File(output_path, "w")
    for scene_id in scene_ids:
        img_file_names = os.listdir(os.path.join(cfg.data.metadata.video_frames_path, scene_id, "color"))
        f.create_dataset(name=scene_id, shape=(len(img_file_names), 128, 32, 41), dtype="f4")
    return f


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # read scene ids
    with open(cfg.data.metadata.scene_ids, "r") as f:
        scene_ids = [line.strip() for line in f]

    h5_file = initialize_output_hdf5(cfg, scene_ids)

    dataloader = DataLoader(
        ScanNet(cfg, scene_ids), batch_size=cfg.data.dataloader.batch_size, shuffle=False,
        num_workers=cfg.data.dataloader.num_workers, pin_memory=True
    )
    enet = get_model(cfg).cuda()
    do_inference(enet, dataloader, h5_file)


if __name__ == "__main__":
    main()
