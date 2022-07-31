'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-07-31 22:01:49
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import sys
sys.path.append('./')
sys.path.append('../')


def test_FaceDataset():
    from dataset.face_dataset import FaceDataset
    dataset = FaceDataset(data_root="data/HDTF_preprocessed",
                          split="data/train.txt")
    print(len(dataset))

    dataset[170]

if __name__ == "__main__":
    test_FaceDataset()