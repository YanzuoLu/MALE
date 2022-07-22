"""
@author: Yanzuo Lu
@email:  luyz5@mail2.sysu.edu.cn
"""

import os
import random

from PIL import Image
from torch.utils.data import Dataset


class LUPerson(Dataset):
    def __init__(self, cfg, transform):
        self.transform = transform

        self.img_items = []
        for root, dirs, files in os.walk(cfg.DATASET.ROOT_DIR):
            for file in files:
                img_path = os.path.join(root, file)
                self.img_items.append((img_path,))
        
        if cfg.DATASET.RANDOM_SAMPLE:
            self.img_items = random.sample(self.img_items, cfg.DATASET.RANDOM_SAMPLE_SIZE)

    def __len__(self):
        return len(self.img_items)
    
    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]

        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.transform(img)
        
        return (img,)
