import cv2

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def grid_positions(shape, device):
    h, w = shape

    rows = torch.linspace(-1, 1, h, device=device).view(h, 1).repeat(1, w)
    columns = torch.linspace(-1, 1, w, device=device).view(1, w).repeat(h, 1)

    grid = torch.stack([rows, columns], dim=-1)

    return grid


def extract_patches(image, ij, device, patch_size=17):
    image = torch.tensor(image).float().to(device).permute([2, 0, 1])
    c, h, w = image.size()

    grid_patch = grid_positions([patch_size, patch_size], device)  # ps x ps x 2
    grid_patch[:, :, 0] *= patch_size / (h - 1)
    grid_patch[:, :, 1] *= patch_size / (w - 1)

    norm_ij = torch.tensor(ij).float().to(device)
    norm_ij[:, 0] = norm_ij[:, 0] / (h - 1) * 2 - 1
    norm_ij[:, 1] = norm_ij[:, 1] / (w - 1) * 2 - 1

    full_ij = norm_ij.view(-1, 1, 1, 2) + grid_patch

    patches = F.grid_sample(
        image.unsqueeze(0), full_ij[:, :, :, [1, 0]].reshape(1, -1, patch_size, 2),
        padding_mode='reflection', align_corners=True
    ).squeeze(0)
    patches = patches.view(c, -1, patch_size, patch_size).permute([1, 0, 2, 3])

    return patches.cpu()


def normalize_batch(images):
    device = images.device
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float).view(1, 3, 1, 1).to(device)
    return (images.float() / 255. - mean) / std


def optimize_displacements_featuremetric(
    batch1, batch2,
    descriptors1, descriptors2,
    feature_extractor,
    device,
    batch_size
):
    # n_batches = batch1.shape[0] // batch_size + (batch1.shape[0] % batch_size != 0)
    pass


def enumerate_displacements_featuremetric(
    feature_batch1,
    feature_batch2,
    device,
    batch_size
):
    B, C, H, W = feature_batch1.shape
    flatten_feature_batch1 = feature_batch1.view(B, C, -1).unsqueeze(-1) # B x 3 x HW X 1
    flatten_feature_batch2 = feature_batch2.view(B, C, -1).unsqueeze(-2) # B x 3 x 1 x HW

    diff_feature_batch = torch.norm(flatten_feature_batch2 - flatten_feature_batch1, dim=1) # B x HW x HW

    min_img2, min_indices_img2 = torch.min(diff_feature_batch, dim=2)
    min, min_indices_img1 = torch.min(min_img2, dim=1)

    img1_ij = min_indices_img1
    img2_ij = min_indices_img2[np.arange(H), min_indices_img1]

    img1_i = int(img1_ij / W)
    img1_j = img1_ij - img1_i * W

    img2_i = int(img2_ij / W)
    img2_j = img2_ij - img2_i * W

    displacement_1 = torch.cat((img1_i - int((H-1)/2), img1_j - int((W-1)/2)))
    displacement_2 = torch.cat((img2_i - int((H-1)/2), img2_j - int((W-1)/2)))

    return displacement_1, displacement_2

def extract_patches_and_estimate_displacements(
        image1, keypoints1, descriptors1,
        image2, keypoints2, descriptors2,
        matches,
        # feature_extractor, 
        device
        ):
    # Extract matched keypoints and reorder from (x,y) to (y,x)
    ij1 = keypoints1[matches[:, 0]][:, [1, 0]]
    ij2 = keypoints2[matches[:, 1]][:, [1, 0]]

    # Get descriptors for the matches keypoints
    matched_descriptors1 = descriptors1[matches[:, 0]]
    matched_descriptors2 = descriptors2[matches[:, 1]]

    # extract dense feature maps
    # feature_map1 = feature_extractor(image1)
    # feature_map2 = feature_extractor(image2)
    feature_map1 = image1 # rgb as features
    feature_map2 = image2

    # Extract patches around the keypoints
    feature_batch1 = extract_patches(
        feature_map1, ij1, device, patch_size=17
    )
    # patch_size needs to be an odd number 
    # shape: num_matches x 3 x patch_size x patch_size

    feature_batch2 = extract_patches(
        feature_map2, ij2, device, patch_size=17
    )

    norm_feature_batch1 = normalize_batch(feature_batch1)
    norm_feature_batch2 = normalize_batch(feature_batch2)

    # # Find the best displacement in two feature patches by optimizing
    # displacements1, displacements2 = optimize_displacements_featuremetric(
    #         norm_batch1, norm_batch2,
    #         matched_descriptors1, matched_descriptors2,
    #         feature_extractor, 
    #         device, 
    #         batch_size=batch_size
    #         )

    # Find the best displacement in two feature patches by enumerating
    displacements1, displacements2 = enumerate_displacements_featuremetric(
            norm_feature_batch1, norm_feature_batch2,
            device, 
            batch_size=batch_size
            )

    return displacements2, displacements1
