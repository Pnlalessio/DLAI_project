import numpy as np
import torch
import math
import cv2

def get_patches(observation, obs_size, patches_number, patch_size, stride):
    '''This function computes, at each step of every episode, the division of each input 
    observation into patches, resulting in a tensor of size (N, M, M, C), where N is the 
    number of patches, M is the height/width of each patch, and C is the number of channels. 
    The function also returns a tensor of size (2, N) containing the coordinates of the patch 
    centers, and a tensor of size (3, N) containing the average color values for each of the 3 
    RGB channels of each patch.'''
    n = int((obs_size - patch_size) / stride + 1)
    offset = patch_size // 2
    patches = []
    positions = []
    patches_color_means = []
    for i in range(n):
        patch_center_row = offset + i * stride
        for j in range(n):
            patch_center_col = offset + j * stride
            positions.append([patch_center_row, patch_center_col])
            patch = observation[patch_center_row - offset:patch_center_row + offset + 1, patch_center_col - offset:patch_center_col + offset + 1, :]
            patches.append(patch)
            colors = torch.mean(torch.tensor(patch).float(), dim=(0,1))
            patches_color_means.append(colors.tolist())
    positions = torch.tensor(positions).float()
    patches = torch.tensor(patches).float()
    patches_color_means = torch.tensor(patches_color_means).float()

    return patches, positions, patches_color_means


def show_blue_or_white_patches(observation, positions, patch_size, k, blue_or_white):
    '''This function visually displays the top K patches the agent focuses on using 
    uniformly colored patches. The intensity of attention is represented by color tone. 
    The first self-attention's intensity is shown in varying shades of blue, while the 
    'Double' self-attention, if present, is represented with variations of white.'''
    observation_copy = observation.copy()
    positions = positions.numpy()
    if blue_or_white == "white":
        colored_patch = np.ones((patch_size, patch_size, 3))
    elif blue_or_white == "blue":
        colored_patch = np.zeros((patch_size, patch_size, 3))
        colored_patch[:, :, 2] = 1.0
    half_patch_size = patch_size // 2
    for i, position in enumerate(positions):
        row_upper_part = int(position[0]) - half_patch_size
        row_bottom_part = int(position[0]) + half_patch_size + 1
        col_left_part = int(position[1]) - half_patch_size
        col_right_part = int(position[1]) + half_patch_size + 1
        ratio = 1.0 * i / k
        observation_copy[row_upper_part:row_bottom_part, col_left_part:col_right_part] = (
            ratio * observation_copy[row_upper_part:row_bottom_part, col_left_part:col_right_part] +
                (1 - ratio) * colored_patch)
    observation_copy = cv2.resize(observation_copy, (observation_copy.shape[0] * 5, observation_copy.shape[1] * 5))
    cv2.imshow('Patches_importance', observation_copy[:, :, [2, 1, 0]])
    cv2.waitKey(1)
