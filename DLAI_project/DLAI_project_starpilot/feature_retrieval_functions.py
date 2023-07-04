import torch

def f(positions, k):
	# f returns the tensor (dimension --> (2, k)) of coordinates for the centers of the top k patches.
    return positions[k]

def get_best_patches_color_means(patches_color_means, k):
	# This function returns the tensor of average RGB color values of the top k patches.
	return patches_color_means[k]

def map_best_patches_on_obs(best_patch_indices, k):
	# Return a dictonary that remembers the indices of best k patches selcted
	mapping = {}
	for i in range(k):
		mapping[i] = best_patch_indices[i].item()
	return mapping

def map_new_indices_to_old_indices(mapping, best_patch_indices):
	# This function maps new indices of the 'Double' self attention to old indices of the first self attention.
	mapped = []
	for i in best_patch_indices:
		mapped.append(mapping[i.item()])
	mapped = torch.tensor(mapped)
	return mapped