import os.path
import numpy as np
import torch
import torch.utils.data as data
import cv2

from utils import data_aug

class DatasetDnCNN(data.Dataset):
    def __init__(self, config, img_list):
        super(DatasetDnCNN, self).__init__()
        self.config = config
        self.n_channels = config.n_channels
        self.sigma = config.sigma

        self.img_list = img_list

    def __getitem__(self, index):
        img_H = self.img_list[index] / 255.0
        noise = np.random.normal(0, self.sigma/255.0, img_H.shape) / 255.0
        img_L = img_H + noise
        img_L = np.clip(img_L, 0, 1)

        # print('H', img_H.max(), img_H.min())
        # print('L', img_L.max(), img_L.min())
        # print('N', noise.max(), noise.min())

        img_H = torch.from_numpy(np.transpose(img_H, (2, 0, 1))).float()  # [H, W, C] -> [C, H, W]
        img_L = torch.from_numpy(np.transpose(img_L, (2, 0, 1))).float()
        # noise = torch.from_numpy(np.transpose(noise, (2, 0, 1))).float()
        return {'H': img_H, 'L': img_L, 'N': noise}

    def __len__(self):
        return len(self.img_list)



def gen_patches(config, dataset_path):
    img_list = os.listdir(dataset_path)
    patchs = []
    count = 0
    for f in range(len(img_list)):
        # read image
        path_H = os.path.join(dataset_path, img_list[f])
        img_H = cv2.imread(path_H)  # np.ndarray, HWC
        img_H = cv2.cvtColor(img_H, cv2.COLOR_BGR2GRAY)  # gray
        img_H = img_H.astype(np.float32) / 255.0  # rescale to [0, 1]
        img_H = np.expand_dims(img_H, axis=2)  # [H, W] -> [H, W, C]

        # extract patches
        H, W, _ = img_H.shape
        for i in range(0, H - config.patch_size + 1, config.stride):
            for j in range(0, W - config.patch_size + 1, config.stride):
                patch_H = img_H[i:i + config.patch_size, j:j + config.patch_size, :]

                # data augmentation
                mode = np.random.randint(0, 8)
                patch_H = data_aug(patch_H, mode=mode)
                
                patchs.append(patch_H)
                count += 1
    print('generate', count, 'patches from', len(img_list), 'images')
    return patchs




    # def __getitem__(self, index):
    #     path_H = os.path.join(self.dataset_H, self.imgs_list[index])
    #     img_H = cv2.imread(path_H)  # np.ndarray, HWC
    #     img_H = cv2.cvtColor(img_H, cv2.COLOR_BGR2GRAY)  # gray
    #     img_H = img_H.astype(np.float32) / 255.0  # rescale to [0, 1]
    #     img_H = np.expand_dims(img_H, axis=2)  # [H, W] -> [H, W, C]
    #
    #     # randomly crop a patch
    #     H, W, _ = img_H.shape
    #     rnd_h = np.random.randint(0, max(0, H - self.patch_size))
    #     rnd_w = np.random.randint(0, max(0, W - self.patch_size))
    #     patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
    #
    #     # data augmentation
    #     mode = np.random.randint(0, 8)
    #     patch_H = data_aug(patch_H, mode=mode)
    #     img_H = patch_H.copy()  # 确保 正stride
    #
    #     # add noise
    #     noise = np.random.normal(0, self.sigma/255, img_H.shape)
    #     img_L = img_H + noise  # noisy image, low quality image
    #     img_L = np.clip(img_L, 0, 1)
    #
    #     # HWC -> CHW, numpy -> tensor
    #     img_H = torch.from_numpy(np.transpose(img_H, (2, 0, 1))).float()  # [H, W, C] -> [C, H, W]
    #     img_L = torch.from_numpy(np.transpose(img_L, (2, 0, 1))).float()
    #
    #     return {'H': img_H, 'path_H': path_H, 'L': img_L, 'path_L': path_H}




    # def test(self):
    #     print(self.imgs_list)
    #     index = 0
    #
    #     path_H = os.path.join(self.dataset_H, self.imgs_list[index])
    #     img_H = cv2.imread(path_H) / 255  # np.ndarray, rescale to [0, 1]
    #
    #     H, W, _ = img_H.shape
    #     rnd_h = np.random.randint(0, max(0, H - self.patch_size))
    #     rnd_w = np.random.randint(0, max(0, W - self.patch_size))
    #     patch_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
    #     img_H = patch_H
    #
    #     noise = np.random.normal(0, self.sigma / 255, img_H.shape)
    #     img_L = img_H + noise  # noisy image, low quality image
    #     img_L = np.clip(img_L, 0, 1)
    #
    #     print('H', img_H.max(), img_H.min())
    #     print('L', img_L.max(), img_L.min())
    #     print('noise', noise.max(), noise.min())
    #
    #     print(img_H.shape)
    #     cv2.imshow('img_H', img_H)
    #     cv2.imshow('img_L', img_L)
    #     # cv2.imshow('noise', noise)
    #     cv2.waitKey(0)
    #
    # def test(self):
    #     for i in range(len(self.imgs_list)):
    #         sample = self.__getitem__(i)
    #         print(i, sample['H'].shape, sample['L'].shape, sample['path_H'], sample['path_L'])
    #     print('test over')