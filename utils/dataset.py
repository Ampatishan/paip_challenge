import os
import cv2
import numpy as np
from torch.utils.data import Dataset

# TODO Incorporate the values from the excel file into the dataset file
'''
0 stands for background
1 stands for tumor 
2 stands for non tumor
'''


class paip_dataset(Dataset):
    def __init__(self, is_colon, image_path, mask_path=None, transforms=None, \
                 target_transfrom=None):
        self.img_path = image_path
        self.mask_path = mask_path
        self.transform = transforms
        self.target_transform = target_transfrom
        self.is_colon = is_colon
        self.img_names = os.listdir(self.img_path)

        if mask_path:
            self.tumor_masks_names = os.listdir(os.path.join( \
                self.mask_path, 'tumor'))
            self.non_tumor_masks_names = os.listdir(os.path.join( \
                self.mask_path, 'non_tumor'))

        if is_colon:
            self.select_part('c')
        else:
            self.select_part('p')

    def __len__(self):
        if self.mask_path:
            assert len(self.img_names) == len(self.non_tumor_masks_names) \
                   == len(self.tumor_masks_names), \
                "Number of files in images and masks doesn't match"
        return len(self.img_names)

    def __getitem__(self, idx):
        mask = np.zeros(3)
        while np.sum(mask) == 0:
            idx+=1
            idx = idx%len(self)
            img_name = self.img_names[idx]

            if self.mask_path:
                tumor_mask_name = self.tumor_masks_names[idx]
                non_tumor_mask_name = self.non_tumor_masks_names[idx]

                assert img_name[:7] == tumor_mask_name[:7] \
                       == non_tumor_mask_name[:7], \
                    "Non-corresponding files selected for masks and images"

            img = cv2.imread(os.path.join(self.img_path, img_name))
            if self.transform:
                img = self.transform(img)

            if self.mask_path:
                tumor_mask = cv2.imread(os.path.join(self.mask_path, 'tumor', tumor_mask_name))
                non_tumor_mask = cv2.imread(os.path.join(self.mask_path, 'non_tumor', non_tumor_mask_name))

                if self.target_transform:
                    tumor_mask = np.squeeze(self.target_transform(tumor_mask[:,:,0]))
                    non_tumor_mask = np.squeeze(self.target_transform(non_tumor_mask[:,:,0]))

                mask = np.stack([tumor_mask * 1, non_tumor_mask * 2])
        if self.mask_path:
            print(np.sum(mask))
            return img, mask
        else:
            return img

    def select_part(self, part):
        self.img_names = [x for x in self.img_names if x[3] == part]
        self.tumor_masks_names = [x for x in self.tumor_masks_names \
                                  if x[3] == part]
        self.non_tumor_masks_names = [x for x in self.non_tumor_masks_names \
                                      if x[3] == part]
        self.img_names.sort()
        self.tumor_masks_names.sort()
        self.non_tumor_masks_names.sort()
