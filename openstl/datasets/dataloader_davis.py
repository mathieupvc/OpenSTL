import os
import random
import cv2
from skimage.transform import downscale_local_mean
import numpy as np
from PIL import Image
from collections import defaultdict
from glob import glob

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader

class DAVIS(object):
    SUBSET_OPTIONS = ['train', 'val', 'test-dev', 'test-challenge']
    TASKS = ['semi-supervised', 'unsupervised']
    DATASET_WEB = 'https://davischallenge.org/davis2017/code.html'
    VOID_LABEL = 255

    def __init__(self, root, task='unsupervised', subset='val', sequences='all', resolution='480p', codalab=False):
        """
        Class to read the DAVIS dataset
        :param root: Path to the DAVIS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to load the annotations, choose between semi-supervised or unsupervised.
        :param subset: Set to load the annotations
        :param sequences: Sequences to consider, 'all' to use all the sequences in a set.
        :param resolution: Specify the resolution to use the dataset, choose between '480' and 'Full-Resolution'
        """
        if subset not in self.SUBSET_OPTIONS:
            raise ValueError(f'Subset should be in {self.SUBSET_OPTIONS}')
        if task not in self.TASKS:
            raise ValueError(f'The only tasks that are supported are {self.TASKS}')

        self.task = task
        self.subset = subset
        self.root = root
        self.img_path = os.path.join(self.root, 'JPEGImages', resolution)
        annotations_folder = 'Annotations' if task == 'semi-supervised' else 'Annotations_unsupervised'
        self.mask_path = os.path.join(self.root, annotations_folder, resolution)
        year = '2019' if task == 'unsupervised' and (subset == 'test-dev' or subset == 'test-challenge') else '2017'
        self.imagesets_path = os.path.join(self.root, 'ImageSets', year)

        self._check_directories()

        if sequences == 'all':
            with open(os.path.join(self.imagesets_path, f'{self.subset}.txt'), 'r') as f:
                tmp = f.readlines()
            sequences_names = [x.strip() for x in tmp]
        else:
            sequences_names = sequences if isinstance(sequences, list) else [sequences]
        self.sequences = defaultdict(dict)

        for seq in sequences_names:
            images = np.sort(glob(os.path.join(self.img_path, seq, '*.jpg'))).tolist()
            if len(images) == 0 and not codalab:
                raise FileNotFoundError(f'Images for sequence {seq} not found.')
            self.sequences[seq]['images'] = images
            masks = np.sort(glob(os.path.join(self.mask_path, seq, '*.png'))).tolist()
            masks.extend([-1] * (len(images) - len(masks)))
            self.sequences[seq]['masks'] = masks

    def _check_directories(self):
        if not os.path.exists(self.root):
            raise FileNotFoundError(f'DAVIS not found in the specified directory, download it from {self.DATASET_WEB}')
        if not os.path.exists(os.path.join(self.imagesets_path, f'{self.subset}.txt')):
            raise FileNotFoundError(f'Subset sequences list for {self.subset} not found, download the missing subset '
                                    f'for the {self.task} task from {self.DATASET_WEB}')
        if self.subset in ['train', 'val'] and not os.path.exists(self.mask_path):
            raise FileNotFoundError(f'Annotations folder for the {self.task} task not found, download it from {self.DATASET_WEB}')

    def get_frames(self, sequence):
        for img, msk in zip(self.sequences[sequence]['images'], self.sequences[sequence]['masks']):
            image = np.array(Image.open(img))
            mask = None if msk is None else np.array(Image.open(msk))
            yield image, mask

    def _get_all_elements(self, sequence, obj_type):
        obj = np.array(Image.open(self.sequences[sequence][obj_type][0]))
        all_objs = np.zeros((len(self.sequences[sequence][obj_type]), *obj.shape))
        obj_id = []
        for i, obj in enumerate(self.sequences[sequence][obj_type]):
            all_objs[i, ...] = np.array(Image.open(obj))
            obj_id.append(''.join(obj.split('/')[-1].split('.')[:-1]))
        return all_objs, obj_id

    def get_all_images(self, sequence):
        return self._get_all_elements(sequence, 'images')

    def get_all_masks(self, sequence, separate_objects_masks=False):
        masks, masks_id = self._get_all_elements(sequence, 'masks')
        masks_void = np.zeros_like(masks)

        # Separate void and object masks
        for i in range(masks.shape[0]):
            masks_void[i, ...] = masks[i, ...] == 255
            masks[i, masks[i, ...] == 255] = 0

        if separate_objects_masks:
            num_objects = int(np.max(masks[0, ...]))
            tmp = np.ones((num_objects, *masks.shape))
            tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
            masks = (tmp == masks[None, ...])
            masks = masks > 0
        return masks, masks_void, masks_id

    def get_sequences(self):
        for seq in self.sequences:
            yield seq

class DAVISDataset(Dataset):
    """DAVIS <https://davischallenge.org/>`_ Dataset"""

    def __init__(self, datas, indices, pre_seq_length, aft_seq_length, use_augment=False):
        super(DAVISDataset,self).__init__()
        self.datas = datas.swapaxes(2, 3).swapaxes(1,2)
        self.indices = indices
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.use_augment = use_augment
        self.mean = 0
        self.std = 1

    def _augment_seq(self, imgs, crop_scale=0.95):
        """Augmentations for video"""
        _, _, h, w = imgs.shape  # original shape, e.g., [10, 1, 128, 128]
        imgs = F.interpolate(imgs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = imgs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        imgs = imgs[:, :, x:x+h, y:y+w]
        # Random Flip
        if random.randint(0, 1):
            imgs = torch.flip(imgs, dims=(3, ))  # horizontal flip
        return imgs

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        batch_ind = self.indices[i]
        begin = batch_ind
        end1 = begin + self.pre_seq_length
        end2 = begin + self.pre_seq_length + self.aft_seq_length
        data = torch.tensor(self.datas[begin:end1, ::]).float()
        labels = torch.tensor(self.datas[end1:end2, ::]).float()
        if self.use_augment:
            imgs = self._augment_seq(torch.cat([data, labels], dim=0), crop_scale=0.95)
            data = imgs[:self.pre_seq_length, ...]
            labels = imgs[self.pre_seq_length:self.pre_seq_length+self.aft_seq_length, ...]
        return data, labels


class InputHandle(object):
    """Class for handling dataset inputs."""

    def __init__(self, datas, indices, input_param):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.image_width = input_param['image_width']
        self.datas = datas
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = input_param['seq_length']

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True):
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        if self.current_position + self.minibatch_size >= self.total():
            return True
        else:
            return False

    def get_batch(self):
        """Gets a mini-batch."""
        if self.no_batch_left():
            print(
                'There is no batch left in %s.'
                'Use iterators.begin() to rescan from the beginning.',
                self.name)
            return None
        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_width,
            self.image_width, 1)).astype(self.input_data_type)
        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.current_input_length
            data_slice = self.datas[begin:end, :, :, :]
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch


class DataProcess(object):
    """Class for preprocessing dataset inputs."""

    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.image_width = input_param['image_width']

        self.input_param = input_param
        self.seq_len = input_param['seq_length']

    def load_data(self, path, mode='train'):
        """Loads the dataset.
        Args:
            path: action_path.
            mode: Training or testing.
        Returns:
            A dataset and indices of the sequence.
        """
        assert mode in ['train', 'val', 'test']
        if mode in ['train', 'val']:
            davis_data_train = DAVIS(root=os.path.join(self.paths, 'train_val'), subset=mode)
            datasets = [davis_data_train]
        else:
            davis_data_train = DAVIS(root=os.path.join(self.paths, 'train_val'), subset='train')  # try testing on training set
            datasets = [davis_data_train]
            # davis_data_challenge = DAVIS(root=os.path.join(self.paths, 'test_challenge'), subset='test-challenge')
            # davis_data_dev = DAVIS(root=os.path.join(self.paths, 'test_dev'), subset='test-dev')
            # datasets = [davis_data_challenge, davis_data_dev]
        print('begin load data' + str(path))

        data = []
        indices = [0]
        for d, davis_data in enumerate(datasets):
            for i, seq in enumerate(davis_data.sequences):
                images, _ = davis_data.get_all_images(seq)
                images = (0.3 * images[..., 0] + 0.59 * images[..., 1] + 0.11 * images[..., 2]).astype(np.uint8)  # convert to gray scale
                crop_size = int((images.shape[2] - 480) / 2)
                odd = images.shape[2] % 2
                images = images[:, :, crop_size:-crop_size-odd]  # take the center of images to obtain a square image
                images = images[:, 24:-24, 24:-24]  # Take center to set size to (432, 432)
                images = images[:, 91:-91, 91:-91]  # crop like in pred retina experiment (but without the upsampling)
                if mode in ['train', 'val']:
                    nb_full_sequences = images.shape[0] // self.seq_len
                    assert nb_full_sequences >= 1
                    data.append(images[:nb_full_sequences*self.seq_len, :, :])
                    if (i == 0) & (d == 0):
                        nb_full_sequences -= 1  # in the first seq, the first index is already added
                    for s in range(nb_full_sequences):
                        indices.append(indices[-1] + self.seq_len)
                else:
                    assert images.shape[0] >= self.seq_len
                    nb_sequences = images.shape[0] - self.seq_len + 1
                    data.append(images)
                    if (i == 0) & (d == 0):
                        indices = indices + [ind + indices[-1] for ind in range(nb_sequences)][1:]
                    else:
                        indices = indices + [ind + indices[-1] + self.seq_len for ind in range(nb_sequences)]

        data = np.concatenate(data, axis=0)
        data = downscale_local_mean(data.astype(np.float32), (1, 5, 5))
        data = data[:, :, :, np.newaxis]
        # data = np.float32(data) / 255
        data = data / 255
        print('there are ' + str(data.shape[0]) + ' pictures')
        print('there are ' + str(len(indices)) + ' sequences')
        return data, indices

    def get_train_input_handle(self):
        train_data, train_indices = self.load_data(self.paths, mode='train')
        return InputHandle(train_data, train_indices, self.input_param)

    def get_val_input_handle(self):
        val_data, val_indices = self.load_data(self.paths, mode='val')
        return InputHandle(val_data, val_indices, self.input_param)

    def get_test_input_handle(self):
        test_data, test_indices = self.load_data(self.paths, mode='test')
        return InputHandle(test_data, test_indices, self.input_param)


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=10, aft_seq_length=3, in_shape=[10, 1, 50, 50],
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):

    img_width = in_shape[-1] if in_shape is not None else 50
    # pre_seq_length, aft_seq_length = 10, 10
    input_param = {
        'paths': os.path.join(data_root, 'DAVIS'),
        'image_width': img_width,
        'minibatch_size': batch_size,
        'seq_length': (pre_seq_length + aft_seq_length),
        'input_data_type': 'float32',
        'name': 'kth'
    }
    input_handle = DataProcess(input_param)
    train_input_handle = input_handle.get_train_input_handle()
    val_input_handle = input_handle.get_val_input_handle()
    test_input_handle = input_handle.get_test_input_handle()

    train_set = DAVISDataset(train_input_handle.datas,
                           train_input_handle.indices,
                           pre_seq_length,
                           aft_seq_length, use_augment=use_augment)
    val_set = DAVISDataset(val_input_handle.datas,
                         val_input_handle.indices,
                         pre_seq_length,
                         aft_seq_length, use_augment=False)
    test_set = DAVISDataset(test_input_handle.datas,
                          test_input_handle.indices,
                          pre_seq_length,
                          aft_seq_length, use_augment=False)

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
    dataloader_train, _, dataloader_test = \
        load_data(batch_size=16,
                val_batch_size=4,
                data_root='/home/mathieu/These/data',
                num_workers=4,
                pre_seq_length=10, aft_seq_length=3)

    print(len(dataloader_train), len(dataloader_test))
    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break
    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break
