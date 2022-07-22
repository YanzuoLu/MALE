"""
@author: Yanzuo Lu
@email:  luyz5@mail2.sysu.edu.cn
"""

import os

from PIL import Image
from torch.utils.data import Dataset


class MSMT17(Dataset):
    def __init__(self, cfg, transform, is_train):
        self.transform = transform
        self.num_classes = None
        self.num_queries = None

        train_dir = os.path.join(cfg.DATASET.ROOT_DIR, "mask_train_v2")
        test_dir = os.path.join(cfg.DATASET.ROOT_DIR, "mask_test_v2")
        list_train_path = os.path.join(cfg.DATASET.ROOT_DIR, "list_train.txt")
        list_query_path = os.path.join(cfg.DATASET.ROOT_DIR, "list_query.txt")
        list_gallery_path = os.path.join(cfg.DATASET.ROOT_DIR, "list_gallery.txt")

        if is_train:
            self.img_items = self.process_dir(train_dir, list_train_path)
        else:
            query_img_items = self.process_dir(test_dir, list_query_path)
            gallery_img_items = self.process_dir(test_dir, list_gallery_path)
            self.img_items = query_img_items + gallery_img_items
            self.num_queries = len(query_img_items)
        
        pid_set = set()
        cam_set = set()
        for img_item in self.img_items:
            pid_set.add(img_item[1])
            cam_set.add(img_item[2])
        
        pids = sorted(list(pid_set))
        cams = sorted(list(cam_set))
        self.pid_dict = dict([(p, i) for i, p in enumerate(pids)])
        self.cam_dict = dict([(p, i) for i, p in enumerate(cams)])
        self.num_classes = len(pids)

    def process_dir(self, root_dir, list_path):
        with open(list_path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
        
        data = []
        for line in lines:
            img_path, pid = line.split(' ')
            pid = int(pid)
            camid = int(img_path.split('_')[2]) - 1
            img_path = os.path.join(root_dir, img_path)
            data.append((img_path, pid, camid))
        
        return data
    
    def __len__(self):
        return len(self.img_items)
    
    def __getitem__(self, i):
        img_item = self.img_items[i]
        img_path, pid, camid = img_item

        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.transform(img)
        pid = self.pid_dict[pid]
        camid = self.cam_dict[camid]

        return (img, pid, camid)
