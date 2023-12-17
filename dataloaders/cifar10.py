import os
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

class Cifar10():
    def __init__(self, config):
        # cifar-10 dataset path
        self.datapath = f"./datasets/{config['dataset']}"
        os.makedirs(self.datapath, exist_ok=True)

        # batch size and number of workers
        self.train_batch_size = config['train_batch_size']
        self.test_batch_size = config['test_batch_size']

        self.num_workers = config['num_workers']


    def compose_transforms(self):
        return transforms.Compose([
            transforms.RandomCrop(size = (32, 32), padding = 4, padding_mode = "symmetric"),
            transforms.RandomRotation(degrees = 15),
            # transforms.RandomAffine(0, shear = 10, scale = (0.8, 1.2)),
            # transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)),
        ])


    def make_dataloader(self, train = True):
        transforms = self.compose_transforms()
        batch_size = self.train_batch_size if train else self.test_batch_size
        
        dataset = datasets.CIFAR10(root = self.datapath, train = train, 
                                    transform = transforms, download = True)
        
        dataloader = DataLoader(dataset, batch_size = batch_size,
                                            shuffle = train, num_workers = self.num_workers)

        return dataloader