import gzip

import matplotlib.pyplot as plt
import numpy as np
import os
import random

import cv2
import skimage.draw
from skimage.transform import downscale_local_mean

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


def load_fixed_set(root, mode='test'):  #TODO: mode is useless for now
    # Load the fixed dataset
    file_map = {
        'test': 'mdiscs/mdiscs_test_seq.npy',
    }
    path = os.path.join(root, file_map[mode])
    dataset = np.load(path)
    dataset = dataset[..., np.newaxis]
    return dataset


class MovingDISCS(Dataset):
    """Moving DISCS Dataset

    Args:
        data_root (str): Path to the dataset.
        is_train (bool): Whether to use the train or test set.
        n_frames_input, n_frames_output (int): The number of input and prediction
            video frames.
        image_size (int): Input resolution of the data.
        use_augment (bool): Whether to use augmentations (defaults to False).
    """

    def __init__(self, root, is_train=True, n_frames_input=10, n_frames_output=10,
                 image_size=50, discs_radius=40, transform=None, use_augment=False):
        super(MovingDISCS, self).__init__()

        self.dataset = None
        self.is_train = is_train
        if not is_train:
            self.dataset = load_fixed_set(root, 'test')
        self.length = int(1e4) if self.dataset is None else self.dataset.shape[1]

        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        self.use_augment = use_augment
        self.image_size_ = image_size
        self.discs_radius_ = discs_radius
        self.mean = 0
        self.std = 1

    def generate_moving_discs(self, num_discs=10, sigma=135, tau=0.08, dt=1/30, repulsive_force=13500, size_after_crop=500, downsampling_factor=10):
        """
        Get random trajectories for the discs and generate a video.
        """

        data = 248 * np.ones((self.n_frames_total, self.image_size_, self.image_size_),
                             dtype=np.uint8)  # 248 is the maximum when using 32 gray levels

        # Initialize random positions for each disc
        positions = np.random.randint(self.discs_radius_ + 1, self.image_size_ - self.discs_radius_ - 1, size=(num_discs, 2))

        # Initialize random velocities for each disc
        velocities = np.random.normal(loc=0, scale=10, size=(num_discs, 2))

        positions_list = []
        for frame_idx in range(self.n_frames_total):
            # Calculate repulsive forces
            for i in range(num_discs):
                for j in range(i + 1, num_discs):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    force = repulsive_force / (dist)
                    force_direction = (positions[i] - positions[j]) / dist
                    velocities[i] += force * force_direction * dt
                    velocities[j] -= force * force_direction * dt

                # Repulsive force with image borders
                dist_to_borders = np.min(np.stack([positions[i] - [self.discs_radius_, self.discs_radius_],
                                                   [self.image_size_ - self.discs_radius_, self.image_size_ - self.discs_radius_] -
                                                   positions[i]]))
                force = repulsive_force / (dist_to_borders)
                force_direction = - np.sign(positions[i] - self.image_size_ / 2) / np.sqrt(2)
                velocities[i] += force * force_direction * dt

            # Generate random noise
            dW = np.random.randn(num_discs, 2)
            # Calculate the Ornstein-Uhlenbeck process
            velocities += -1 / tau * velocities * dt + sigma * dW

            # Boundary checks and adjustment of velocities
            positions = positions + velocities * dt  # convert back to pixels
            positions = np.clip(positions, self.discs_radius_ + 1, self.image_size_ - self.discs_radius_ - 1)

            # update positions list
            positions_list.append(positions)

        # convert positions from [0, 1] to pixel coordinates
        positions_list = [np.round(positions).astype(int) for positions in positions_list]

        # add discs to data
        for i in range(self.n_frames_total):
            for n in range(num_discs):
                xx, yy = skimage.draw.disk((positions_list[i][n, 0], positions_list[i][n, 1]), self.discs_radius_ + 1,
                                           shape=data[i].shape)  # +1 is for skimage behavior
                data[i, xx, yy] = 0

        crop_size = int((data.shape[1] - size_after_crop) / 2)
        data = data[:, crop_size:-crop_size, crop_size:-crop_size]
        data = downscale_local_mean(data.astype(np.float16), (1, downsampling_factor, downsampling_factor)).astype(np.uint8)

        data = data[..., np.newaxis]
        return data

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output

        if self.is_train:
            # Generate data on the fly
            images = self.generate_moving_discs(num_discs=10)
        else:
            images = self.dataset[:, idx, ...]

        r = 1
        w = int(self.image_size_ / r)
        images = images.reshape((length, w, r, w, r)).transpose(
            0, 2, 4, 1, 3).reshape((length, r * r, w, w))

        input = images[:self.n_frames_input]
        if self.n_frames_output > 0:
            output = images[self.n_frames_input:length]
        else:
            output = []

        output = torch.from_numpy(output).contiguous().float()
        input = torch.from_numpy(input).contiguous().float()

        # if self.use_augment:
        #     imgs = self._augment_seq(torch.cat([input, output], dim=0), crop_scale=0.94)
        #     input = imgs[:self.n_frames_input, ...]
        #     output = imgs[self.n_frames_input:self.n_frames_input+self.n_frames_output, ...]

        return input, output

    def __len__(self):
        return self.length


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=10, aft_seq_length=10, in_shape=[10, 1, 864, 864],
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):

    image_size = in_shape[-1] if in_shape is not None else 864
    train_set = MovingDISCS(root=data_root, is_train=True,
                            n_frames_input=pre_seq_length,
                            n_frames_output=aft_seq_length,
                            image_size=image_size, use_augment=use_augment)
    val_set = MovingDISCS(root=data_root, is_train=True,
                            n_frames_input=pre_seq_length,
                            n_frames_output=aft_seq_length,
                            image_size=image_size, use_augment=use_augment)
    test_set = MovingDISCS(root=data_root, is_train=False,
                           n_frames_input=pre_seq_length,
                           n_frames_output=aft_seq_length,
                           image_size=image_size, use_augment=False)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = create_loader(val_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    from openstl.utils import init_dist
    os.environ['LOCAL_RANK'] = str(0)
    os.environ['RANK'] = str(0)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    dist_params = dict(launcher='pytorch', backend='nccl', init_method='env://', world_size=1)
    init_dist(**dist_params)

    dataloader_train, _, dataloader_test = \
        load_data(batch_size=16,
                  val_batch_size=4,
                  data_root='../../data/',
                  num_workers=4,
                  data_name='mnist',
                  pre_seq_length=10, aft_seq_length=10,
                  distributed=True, use_prefetcher=False)

    print(len(dataloader_train), len(dataloader_test))
    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break
    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break
