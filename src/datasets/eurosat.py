import os
import torch
import torchvision.datasets as datasets
import re

import random
from torch.utils.data import Subset, Dataset

def pretify_classname(classname):
    l = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', classname)
    l = [i.lower() for i in l]
    out = ' '.join(l)
    if out.endswith('al'):
        return out + ' area'
    return out

class EuroSATBase:
    def __init__(self,
                 preprocess,
                 test_split,
                 location='~/data',
                 batch_size=32,
                 num_workers=0,
                 subset_data_ratio=1.0):
        # Data loading code
        traindir = os.path.join(location, 'EuroSAT_splits', 'train')
        testdir = os.path.join(location, 'EuroSAT_splits', test_split)

        self.train_dataset = datasets.ImageFolder(traindir, transform=preprocess)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # if subset_data_ratio < 1.0:
        #     random.seed(42)
        #     random_indexs = random.sample(range(len(self.train_dataset)), int(subset_data_ratio * len(self.train_dataset)))
        #     print('eurosat subset_data_ratio:' + str(subset_data_ratio))
        #     print('eurosat random_indexs:' + str(random_indexs))
        #     self.train_dataset_sub = Subset(self.train_dataset, random_indexs)
        # else:
        #     self.train_dataset_sub = self.train_dataset

        # self.train_loader_sub = torch.utils.data.DataLoader(
        #     self.train_dataset_sub,
        #     shuffle=True,
        #     batch_size=batch_size,
        #     num_workers=num_workers,
        # )


        self.test_dataset = datasets.ImageFolder(testdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        if subset_data_ratio < 1.0:
            random.seed(42)
            random_indexs = random.sample(range(len(self.test_dataset)),
                                          int(subset_data_ratio * len(self.test_dataset)))
            print('eurosat subset_data_ratio:' + str(subset_data_ratio))
            print('eurosat test_dataset_len:' + str(len(self.test_dataset)))
            print('eurosat random_indexs:' + str(random_indexs)[:50])
            self.test_dataset_sub = Subset(self.test_dataset, random_indexs)
        else:
            self.test_dataset_sub = self.test_dataset

        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset_sub,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())

        self.classnames = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]
        self.classnames = [pretify_classname(c) for c in self.classnames]
        ours_to_open_ai = {
            'annual crop': 'annual crop land',
            'forest': 'forest',
            'herbaceous vegetation': 'brushland or shrubland',
            'highway': 'highway or road',
            'industrial area': 'industrial buildings or commercial buildings',
            'pasture': 'pasture land',
            'permanent crop': 'permanent crop land',
            'residential area': 'residential buildings or homes or apartments',
            'river': 'river',
            'sea lake': 'lake or sea',
        }
        for i in range(len(self.classnames)):
            self.classnames[i] = ours_to_open_ai[self.classnames[i]]


class EuroSAT(EuroSATBase):
    def __init__(self,
                 preprocess,
                 location='~/data',
                 batch_size=32,
                 num_workers=0,
                 subset_data_ratio=1.0):
        super().__init__(preprocess, 'test', location, batch_size, num_workers, subset_data_ratio)


class EuroSATVal(EuroSATBase):
    def __init__(self,
                 preprocess,
                 location='~/data',
                 batch_size=32,
                 num_workers=0,
                 subset_data_ratio=1.0):
        super().__init__(preprocess, 'val', location, batch_size, num_workers, subset_data_ratio)