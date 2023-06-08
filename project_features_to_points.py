import os
import h5py
import torch
import hydra
import open3d as o3d
import numpy as np
from PIL import Image
from tqdm.std import tqdm


class ProjectionHelper:
    def __init__(self, depth_min, depth_max, image_dims, accuracy, intrinsic):
        self.intrinsic = intrinsic
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.image_dims = image_dims
        self.accuracy = accuracy

        # precompute
        self._compute_corner_points()

    def depth_to_skeleton(self, ux, uy, depth):
        # 2D to 3D coordinates with depth (used in compute_frustum_bounds)
        x = (ux - self.intrinsic[0][2]) / self.intrinsic[0][0]
        y = (uy - self.intrinsic[1][2]) / self.intrinsic[1][1]
        return torch.tensor([depth * x, depth * y, depth], dtype=torch.float32, device="cuda")

    def _compute_corner_points(self):
        corner_points = torch.ones(size=(8, 4), device="cuda", dtype=torch.float32)

        # image to camera
        # depth min
        corner_points[0][:3] = self.depth_to_skeleton(0, 0, self.depth_min)
        corner_points[1][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_min)
        corner_points[2][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1, self.depth_min)
        corner_points[3][:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_min)
        # depth max
        corner_points[4][:3] = self.depth_to_skeleton(0, 0, self.depth_max)
        corner_points[5][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, 0, self.depth_max)
        corner_points[6][:3] = self.depth_to_skeleton(self.image_dims[0] - 1, self.image_dims[1] - 1, self.depth_max)
        corner_points[7][:3] = self.depth_to_skeleton(0, self.image_dims[1] - 1, self.depth_max)

        self.corner_points = corner_points

    def compute_frustum_normals(self, corner_coords):
        """
        Computes the normal vectors (pointing inwards) to the 6 planes that bound the viewing frustum

        :param corner_coords: torch tensor of shape (8, 4), coordinates of the corner points of the viewing frustum
        :return: normals: torch tensor of shape (6, 3)
        """
        normals = torch.empty(size=(6, 3), dtype=torch.float32, device=corner_coords.device)
        plane_indices = (
            (3, 0, 1),
            (2, 1, 5),
            (3, 2, 6),
            (0, 3, 7),
            (1, 0, 4),
            (6, 5, 4)
        )
        # compute plane normals
        for i, (idx1, idx2, idx3) in enumerate(plane_indices):
            plane_vec1 = corner_coords[idx1][:3] - corner_coords[idx2][:3]
            plane_vec2 = corner_coords[idx3][:3] - corner_coords[idx2][:3]
            normals[i] = torch.cross(plane_vec1.view(-1), plane_vec2.view(-1))
        return normals

    @staticmethod
    def points_in_frustum(corner_coords, normals, new_pts):
        """
        Checks whether new_pts ly in the frustum defined by the coordinates of the corners coner_coords

        :param corner_coords: torch tensor of shape (8, 4), coordinates of the corners of the viewing frustum
        :param normals: torch tensor of shape (6, 3), normal vectors of the 6 planes of the viewing frustum
        :param new_pts: (num_points, 3)
        :param return_mask: if False, returns number of new_points in frustum
        :return: if return_mask=True, returns Boolean mask determining whether point is in frustum or not
        """

        # create vectors from point set to the planes
        point_to_plane1 = (new_pts - corner_coords[2][:3].view(-1))
        point_to_plane2 = (new_pts - corner_coords[4][:3].view(-1))

        # check if the scalar product with the normals is positive
        masks = []
        # for each normal, create a mask for points that lie on the correct side of the plane
        for k, normal in enumerate(normals):
            if k < 3:
                masks.append(torch.round(torch.mm(point_to_plane1, normal.unsqueeze(1)) * 100) / 100 < 0)
            else:
                masks.append(torch.round(torch.mm(point_to_plane2, normal.unsqueeze(1)) * 100) / 100 < 0)
        mask = torch.ones(point_to_plane1.shape[0], device="cuda", dtype=torch.bool)

        # create a combined mask, which keeps only the points that lie on the correct side of each plane
        for addMask in masks:
            mask = mask * addMask.squeeze()
        return mask

    def compute_projection(self, points, depth, camera_to_world):
        """
        Computes correspondances of points to pixels

        :param points: tensor containing all points of the point cloud (num_points, 3)
        :param depth: depth map (size: proj_image)
        :param camera_to_world: camera pose (4, 4)
        :param num_points: number of points in one sample point cloud (4096)
        :return: indices_3d (array with point indices that correspond to a pixel),
                indices_2d (array with pixel indices that correspond to a point)
        """

        num_points = points.shape[0]
        world_to_camera = camera_to_world.inverse()

        # create 1-dim array with all indices and array with 4-dim coordinates x, y, z, 1 of points
        ind_points = torch.arange(0, num_points, device="cuda")
        coords = torch.ones(size=(4, num_points), device="cuda", dtype=torch.float32)
        coords[:3, :] = points.T

        # compute viewing frustum
        corner_coords = torch.bmm(camera_to_world.expand(8, -1, -1), self.corner_points.unsqueeze(2))
        normals = self.compute_frustum_normals(corner_coords)

        # check if points are in viewing frustum and only keep according indices
        mask_frustum_bounds = self.points_in_frustum(corner_coords, normals, points)

        if not mask_frustum_bounds.any():
            return None
        ind_points = ind_points[mask_frustum_bounds]
        coords = coords[:, ind_points]

        # project world (coords) to camera
        camera = torch.mm(world_to_camera, coords)

        # project camera to image
        camera[0] = (camera[0] * self.intrinsic[0][0]) / camera[2] + self.intrinsic[0][2]
        camera[1] = (camera[1] * self.intrinsic[1][1]) / camera[2] + self.intrinsic[1][2]
        image = torch.round(camera).int()

        # keep points that are projected onto the image into the correct pixel range
        valid_ind_mask = torch.ge(image[0], 0) * torch.ge(image[1], 0) * torch.lt(
            image[0], self.image_dims[0]
        ) * torch.lt(image[1], self.image_dims[1])
        if not valid_ind_mask.any():
            return None
        valid_image_ind_x = image[0][valid_ind_mask]
        valid_image_ind_y = image[1][valid_ind_mask]
        valid_image_ind = valid_image_ind_y * self.image_dims[0] + valid_image_ind_x

        # keep only points that are in the correct depth ranges (self.depth_min - self.depth_max)
        depth_vals = torch.index_select(depth.view(-1), 0, valid_image_ind)
        depth_mask = depth_vals.ge(self.depth_min) * depth_vals.le(self.depth_max) * torch.abs(
            depth_vals - camera[2][valid_ind_mask]
        ).le(self.accuracy)
        if not depth_mask.any():
            return None

        # create two vectors for all considered points that establish 3d to 2d correspondence
        ind_update = ind_points[valid_ind_mask]
        ind_update = ind_update[depth_mask]

        indices_3d = torch.zeros(num_points + 1, dtype=ind_update.dtype, device=ind_update.device)
        indices_2d = torch.zeros_like(indices_3d)

        indices_3d[0] = ind_update.shape[0]  # first entry: number of relevant entries (of points)
        indices_2d[0] = ind_update.shape[0]
        indices_3d[1:1 + indices_3d[0]] = ind_update  # indices of points
        indices_2d[1:1 + indices_2d[0]] = torch.index_select(valid_image_ind, 0, torch.nonzero(depth_mask)[:, 0])  # indices of corresponding pixels
        return indices_3d, indices_2d

    def project(self, label, lin_indices_3d, lin_indices_2d, num_points):
        """
        forward pass of back projection for 2d features onto 3d points

        :param label: image features (shape: (num_input_channels, proj_image_dims[0], proj_image_dims[1]))
        :param lin_indices_3d: point indices from projection (shape: (num_input_channels, num_points_sample))
        :param lin_indices_2d: pixel indices from projection (shape: (num_input_channels, num_points_sample))
        :param num_points: number of points in one sample
        :return: array of points in sample with projected features (shape: (num_input_channels, num_points))
        """

        num_label_ft = 1 if len(label.shape) == 2 else label.shape[0]  # = num_input_channels
        output = torch.zeros(size=(num_label_ft, num_points), dtype=label.dtype, device=label.device)
        num_ind = lin_indices_3d[0]
        if num_ind > 0:
            # selects values from image_features at indices given by lin_indices_2d
            vals = torch.index_select(label.view(num_label_ft, -1), 1, lin_indices_2d[1:1 + num_ind])
            output.view(num_label_ft, -1)[:, lin_indices_3d[1:1 + num_ind]] = vals
        return output


def compute_projection(points, depth, camera_to_world, projector):
    """
        :param points: tensor containing all points of the point cloud (num_points, 3)
        :param depth: depth map (size: proj_image)
        :param camera_to_world: camera pose (4, 4)

        :return indices_3d (array with point indices that correspond to a pixel),
        :return indices_2d (array with pixel indices that correspond to a point)

        note:
            the first digit of indices represents the number of relevant points
            the rest digits are for the projection mapping
    """
    num_points = points.shape[0]
    num_frames = depth.shape[0]
    indices_3ds = torch.zeros((num_frames, num_points + 1), dtype=torch.long, device="cuda")
    indices_2ds = torch.zeros_like(indices_3ds)

    for i in range(num_frames):
        indices = projector.compute_projection(points, depth[i], camera_to_world[i])
        if indices:
            indices_3ds[i] = indices[0]
            indices_2ds[i] = indices[1]
    return indices_3ds, indices_2ds


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    input_h5_file = h5py.File(cfg.enet_feature_output_path, "r")
    projector = ProjectionHelper(0.1, 4.0, cfg.data.depth_image_size, 0.05, cfg.data.metadata.intrinsic)

    output_path = os.path.join(cfg.output_root_dir, f"multiview_features.h5")
    output_h5_file = h5py.File(output_path, "w")
    with open(cfg.data.metadata.scene_ids) as f:
        scene_list = [line.strip() for line in f]

    for scene_id in tqdm(scene_list):
        scene_data_path = os.path.join(cfg.data.metadata.video_frames_path, scene_id)
        scene_xyz = torch.from_numpy(
            np.asarray(
                o3d.io.read_point_cloud(
                    os.path.join(cfg.data.raw_scene_path, scene_id, f"{scene_id}_vh_clean_2.ply")
                ).points, dtype=np.float32
            )
        ).to("cuda")
        num_frames = input_h5_file[scene_id].shape[0]
        scene_depths = torch.zeros(size=(num_frames, 32, 41), dtype=torch.float32)
        scene_poses = torch.zeros(size=(num_frames, 4, 4), dtype=torch.float32)
        for i in range(num_frames):
            frame_id = cfg.data.frames_sample * i
            scene_depths[i] = torch.from_numpy(
                np.asarray(
                    Image.open(os.path.join(scene_data_path, "depth", f"{frame_id}.png")), dtype=np.float32
                ) / 1000
            )
            scene_poses[i] = torch.from_numpy(
                np.loadtxt(os.path.join(scene_data_path, "pose", f"{frame_id}.txt"), dtype=np.float32)
            )

        scene_depths = scene_depths.to("cuda")
        scene_poses = scene_poses.to("cuda")
        # compute projections for each chunk
        projection_3d, projection_2d = compute_projection(scene_xyz, scene_depths, scene_poses, projector)

        # compute valid projections
        projections = []
        for i in range(projection_3d.shape[0]):
            num_valid = projection_3d[i, 0]
            if num_valid == 0:
                continue
            projections.append((i, projection_3d[i], projection_2d[i]))

        # project
        point_features = torch.zeros(size=(scene_xyz.shape[0], 128), dtype=scene_xyz.dtype, device=scene_xyz.device)
        for i, projection in enumerate(projections):
            frame_id = projection[0]
            projection_3d = projection[1]
            projection_2d = projection[2]
            feat = torch.from_numpy(input_h5_file[scene_id][frame_id]).to("cuda")

            proj_feat = projector.project(feat, projection_3d, projection_2d, scene_xyz.shape[0]).transpose(1, 0)

            # only apply max pooling on the overlapping points
            # find out the points that are covered in projection
            feat_mask = (proj_feat == 0).sum(1) != 128
            # find out the points that are not filled with features
            point_mask = (point_features == 0).sum(1) == 128

            # for the points that are not filled with features
            # and are covered in projection,
            # simply fill those points with projected features
            mask = point_mask * feat_mask
            point_features[mask] = proj_feat[mask]

            # for the points that have already been filled with features
            # and are covered in projection,
            # apply max pooling first and then fill with pooled values
            mask = ~point_mask * feat_mask
            point_features[mask] = torch.max(point_features[mask], proj_feat[mask])

        # save
        output_h5_file.create_dataset(scene_id, data=point_features.cpu().numpy(), dtype="f4", compression="gzip")
    output_h5_file.close()
    print(f"==> Complete. Saved at: {os.path.abspath(output_path)}\n")
    input_h5_file.close()


if __name__ == "__main__":
    main()
