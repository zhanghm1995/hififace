'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-07-31 21:41:54
Email: haimingzhang@link.cuhk.edu.cn
Description: The dataset class to load HDTF face dataset
'''

import os
import os.path as osp
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import defaultdict
import cv2

from .preprocess import DataProcessor


def same_or_not(percent):
    return random.randrange(100) < percent


class FaceDataset(Dataset):
    def __init__(
        self,
        data_root,
        split,
        same_rate: int = 50) -> None:
        super().__init__()

        self.data_root = data_root
        self.same_rate = same_rate

        if osp.exists(split):
            split_file = split
        else:
            split_file = osp.join(self.data_root, f'{split}.txt')
        
        all_images_fp = open(split_file).read().splitlines()

        self.video_name_to_imgs_list_dict = defaultdict(list)
        for line in all_images_fp:
            name = line.split('/')[0]
            self.video_name_to_imgs_list_dict[name].append(line)
        
        self._build_dataset()

        self.data_processor = DataProcessor()

        self.transform = transforms.Compose([transforms.Resize((256, 256)),
                                             transforms.CenterCrop((256, 256)),
                                             transforms.ToTensor()])

    def __getitem__(self, index):
        ## 1) Get the correspoind video and source image frame index
        main_idx, sub_idx = self._get_data(index)
        choose_video = self.all_videos_dir[main_idx] # choosed video directory name, str type
        video_length = self.total_frames_list[main_idx] # total frames in this video

        ## 2) Get the source image
        s_idx = sub_idx
        
        if same_or_not(self.same_rate):
            t_idx = s_idx
        else:
            t_idx = random.randrange(video_length)

        if t_idx == s_idx:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)
        
        ## 3) Get the images
        src_fp = self.video_name_to_imgs_list_dict[choose_video][s_idx]
        src_img_fp = osp.join(self.data_root, src_fp + ".jpg")
        _, src_file_name = osp.split(src_fp)
        s_img = Image.open(src_img_fp).convert('RGB')

        f_fp = self.video_name_to_imgs_list_dict[choose_video][t_idx]
        f_img_fp = osp.join(self.data_root, f_fp + ".jpg")
        _, f_file_name = osp.split(f_fp)
        f_img = Image.open(f_img_fp).convert('RGB')

        ## 4) Align the face
        ## Load the landmarks
        abs_video_dir = osp.join(self.data_root, choose_video)
        
        src_lm_path = osp.join(abs_video_dir, "face_image", "landmarks", f"{src_file_name}.txt")
        raw_lm = np.loadtxt(src_lm_path).astype(np.float32) # (68, 2)
        raw_lm[:, -1] = s_img.size[0] - 1 - raw_lm[:, -1]

        s_img, lm_affine, mat, mat_inv = self.data_processor(np.array(s_img), raw_lm)

        tgt_lm_path = osp.join(abs_video_dir, "face_image", "landmarks", f"{f_file_name}.txt")
        raw_lm = np.loadtxt(tgt_lm_path).astype(np.float32) # (68, 2)
        raw_lm[:, -1] = f_img.size[0] - 1 - raw_lm[:, -1]

        tgt_img, lm_affine, tgt_mat, mat_inv = self.data_processor(np.array(f_img), raw_lm)

        ## 5) Read the face mask of target image
        msk_img_fp = osp.join(abs_video_dir, "face_ear_mask", f"{f_file_name}.png")
        msk_img = Image.open(msk_img_fp).convert('RGB')
        msk_img = cv2.warpAffine(np.array(msk_img), tgt_mat, (256, 256), borderValue=(0,0,0))
        msk_img = cv2.dilate(msk_img, np.ones((7, 7), np.uint8), iterations=1)

        if self.transform is not None:
            s_img = self.transform(Image.fromarray(s_img))
            tgt_img = self.transform(Image.fromarray(tgt_img))
            tgt_msk_img = self.transform(Image.fromarray(msk_img))
        
        return {
            'target_image': tgt_img,
            'source_image': s_img,
            'target_mask': tgt_msk_img[:1, ...],
            'same': same,
        }

    def __len__(self):
        return sum([x for x in self.total_frames_list])

    def _build_dataset(self):
        self.total_frames_list = []
        self.length_token_list = [] # increamental length list
        self.all_videos_dir = []

        total_length = 0
        for video_name, all_imgs_list in self.video_name_to_imgs_list_dict.items():
            self.all_videos_dir.append(video_name)

            num_frames = len(all_imgs_list)
            self.total_frames_list.append(num_frames)

            total_length += num_frames
            self.length_token_list.append(total_length)

    def _get_data(self, index):
        """Get the seperate index location from the total index

        Args:
            index (int): index in all avaible sequeneces
        
        Returns:
            main_idx (int): index specifying which video
            sub_idx (int): index specifying what the start index in this sliced video
        """
        def fetch_data(length_list, index):
            assert index < length_list[-1]
            temp_idx = np.array(length_list) > index
            list_idx = np.where(temp_idx==True)[0][0]
            sub_idx = index
            if list_idx != 0:
                sub_idx = index - length_list[list_idx - 1]
            return list_idx, sub_idx

        main_idx, sub_idx = fetch_data(self.length_token_list, index)
        return main_idx, sub_idx