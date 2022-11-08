import pickle

import torch.utils.data as data
from PIL import Image


class Clothing1M(data.Dataset):

    def __init__(self, root='', set_split='train', transform=None):
        self.root = root
        self.transform = transform
        self.set_split = set_split

        filelist_file = self.root + 'annotations/' + self.set_split + '_kv.txt'

        with open(filelist_file) as f:
            lines_filelist = [x.strip() for x in f.readlines()]

        self.images = [x.split()[0] for x in lines_filelist]

        filehandler = open(self.root + 'annotations/' + self.set_split + '_kv.pickle', 'rb')
        self.annotations_dict = pickle.load(filehandler)

    def __getitem__(self, index):
        img = Image.open(self.root + self.images[index]).convert('RGB')
        target = int(self.annotations_dict[self.images[index]])

        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.images)
