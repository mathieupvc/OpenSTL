import gzip
import numpy as np
import os
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


def load_fixed_set(root, mode='train'):
    # Load the fixed dataset
    file_map = {
        'train': 'mdiscs/mdiscs_train_seq.npy',
        'val': 'mdiscs/mdiscs_val_seq.npy',
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

    def __init__(self, root, mode='train', n_frames_input=10, n_frames_output=10,
                 image_size=20, transform=None, use_augment=False):
        super(MovingDISCS, self).__init__()

        self.dataset = None
        self.mode = mode
        self.dataset = load_fixed_set(root, self.mode)
        self.length = self.dataset.shape[1]

        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.transform = transform
        self.use_augment = use_augment
        self.mean = 0
        self.std = 1

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output

        images = self.dataset[:, idx, ...]

        r = 1
        w = int(20 / r)
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
              pre_seq_length=10, aft_seq_length=10, in_shape=[10, 1, 20, 20],
              distributed=False, use_augment=False, use_prefetcher=False):

    image_size = in_shape[-1] if in_shape is not None else 20
    train_set = MovingDISCS(root=data_root, mode='train',
                            n_frames_input=pre_seq_length,
                            n_frames_output=aft_seq_length,
                            image_size=image_size, use_augment=use_augment)
    val_set = MovingDISCS(root=data_root, mode='val',
                            n_frames_input=pre_seq_length,
                            n_frames_output=aft_seq_length,
                            image_size=image_size, use_augment=use_augment)
    test_set = MovingDISCS(root=data_root, mode='test',
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
                                    pin_memory=True, drop_last=False,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=False,
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
