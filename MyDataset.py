from PIL import Image
import torch

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, datatxt, transform=None, target_transform=None):
        super(MyDataset,self).__init__()
        fh = open(root + datatxt, 'r')
        imgs = []
        weights = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1]), 0.5))
            #weights.append(0.5)
        self.imgs = imgs
        #self.weights = weights
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label, weight = self.imgs[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label, fn, weight, index

    def __len__(self):
        return len(self.imgs)