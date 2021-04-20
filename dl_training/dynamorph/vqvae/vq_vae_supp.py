import os
import numpy as np
import queue
import torch as t
import torch.nn as nn
from torch.utils.data import TensorDataset
from scipy.sparse import csr_matrix
import logging

CHANNEL_RANGE = [(0.3, 0.8), (0., 0.6)]
CHANNEL_VAR = np.array([0.0475, 0.0394]) # After normalized to CHANNEL_RANGE
CHANNEL_MAX = 65535.
eps = 1e-9

log = logging.getLogger(__name__)

def reorder_with_trajectories(dataset, relations, seed=None, w_a=1.1, w_t=0.1):
    """ Reorder `dataset` to facilitate training with matching loss

    Args:
        dataset (TensorDataset): dataset of training inputs
        relations (dict): dict of pairwise relationship (adjacent frames, same
            trajectory)
        seed (int or None, optional): if given, random seed
        w_a (float): weight for adjacent frames
        w_t (float): weight for non-adjecent frames in the same trajectory

    Returns:
        TensorDataset: dataset of training inputs (after reordering)
        scipy csr matrix: sparse matrix of pairwise relations
        list of int: index of samples used for reordering

    """
    if not seed is None:
        np.random.seed(seed)
    inds_pool = set(range(len(dataset)))
    inds_in_order = []
    relation_dict = {}
    for pair in relations:
        if relations[pair] == 2: # Adjacent pairs
            if pair[0] not in relation_dict:
                relation_dict[pair[0]] = []
            relation_dict[pair[0]].append(pair[1])
    while len(inds_pool) > 0:
        rand_ind = np.random.choice(list(inds_pool))
        if not rand_ind in relation_dict:
            inds_in_order.append(rand_ind)
            inds_pool.remove(rand_ind)
        else:
            traj = [rand_ind]
            q = queue.Queue()
            q.put(rand_ind)
            while True:
                try:
                    elem = q.get_nowait()
                except queue.Empty:
                    break
                new_elems = relation_dict[elem]
                for e in new_elems:
                    if not e in traj:
                        traj.append(e)
                        q.put(e)
            inds_in_order.extend(traj)
            for e in traj:
                inds_pool.remove(e)
    new_tensor = dataset[np.array(inds_in_order)]

    values = []
    new_relations = []
    for k, v in relations.items():
        # 2 - adjacent, 1 - same trajectory
        if v == 1:
            values.append(w_t)
        elif v == 2:
            values.append(w_a)
        new_relations.append(k)
    new_relations = np.array(new_relations)
    relation_mat = csr_matrix((np.array(values), (new_relations[:, 0], new_relations[:, 1])),
                              shape=(len(dataset), len(dataset)))
    relation_mat = relation_mat[np.array(inds_in_order)][:, np.array(inds_in_order)]
    return new_tensor, relation_mat, inds_in_order


def vae_preprocess(dataset,
                   use_channels=[0, 1],
                   preprocess_setting={
                       0: ("normalize", 0.4, 0.05),  # Phase
                       1: ("scale", 0.05),  # Retardance
                       2: ("normalize", 0.5, 0.05),  # Brightfield
                   },
                   clip=[0, 1]):
    """ Preprocess `dataset` to a suitable range

    Args:
        dataset (TensorDataset): dataset of training inputs
        use_channels (list, optional): list of channel indices used for model
            prediction
        preprocess_setting (dict, optional): settings for preprocessing,
            formatted as {channel index: (preprocessing mode,
                                          target mean,
                                          target std(optional))}

    Returns:
        TensorDataset: dataset of training inputs (after preprocessing)

    """

    tensor = dataset
    output = []
    for channel in use_channels:
        channel_slice = tensor[:, channel]
        channel_slice = channel_slice / CHANNEL_MAX  # Scale to [0, 1]
        if preprocess_setting[channel][0] == "scale":
            target_mean = preprocess_setting[channel][1]
            slice_mean = channel_slice.mean()
            output_slice = channel_slice / slice_mean * target_mean
        elif preprocess_setting[channel][0] == "normalize":
            target_mean = preprocess_setting[channel][1]
            target_sd = preprocess_setting[channel][2]
            slice_mean = channel_slice.mean()
            slice_sd = channel_slice.std()
            z_channel_slice = (channel_slice - slice_mean) / slice_sd
            output_slice = z_channel_slice * target_sd + target_mean
        else:
            raise ValueError("Preprocessing mode not supported")
        if clip:
            output_slice = np.clip(output_slice, clip[0], clip[1])
        output.append(output_slice)
    output = np.stack(output, 1)
    return output


