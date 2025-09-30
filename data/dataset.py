import os
import copy
import torch
import numpy as np
import urllib.request
import zipfile

from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image


class KuzushijiMNIST(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        if train:
            self.data = np.load(os.path.join(root, 'kmnist-train-imgs.npz'))['arr_0']
            self.targets = np.load(os.path.join(root, 'kmnist-train-labels.npz'))['arr_0']
        else:
            self.data = np.load(os.path.join(root, 'kmnist-test-imgs.npz'))['arr_0']
            self.targets = np.load(os.path.join(root, 'kmnist-test-labels.npz'))['arr_0']
        # min max normalization
        self.data = self.data / 255
        # to torch tensor
        self.data = torch.from_numpy(self.data).float()
        self.targets = torch.from_numpy(self.targets).long()
        self.classes = np.arange(10)
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # add channel dim
        img = img.unsqueeze(0)
        return img, target
    
    def __len__(self):
        return len(self.data)


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        
        # Use existing data directory directly
        self.data_dir = root
        
        if not os.path.exists(self.data_dir):
            raise RuntimeError(f'Dataset not found at {self.data_dir}. Please check the path.')
        
        # Load class names
        self.classes = []
        wnids_path = os.path.join(self.data_dir, 'wnids.txt')
        if not os.path.exists(wnids_path):
            raise RuntimeError(f'wnids.txt not found at {wnids_path}')
            
        with open(wnids_path, 'r') as f:
            self.classes = [line.strip() for line in f]
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load data
        self.data = []
        self.targets = []
        
        if train:
            # Load training data
            train_dir = os.path.join(self.data_dir, 'train')
            for class_name in self.classes:
                class_dir = os.path.join(train_dir, class_name, 'images')
                if os.path.exists(class_dir):
                    for img_file in os.listdir(class_dir):
                        if img_file.endswith(('.JPEG', '.jpg', '.png')):
                            img_path = os.path.join(class_dir, img_file)
                            self.data.append(img_path)
                            self.targets.append(self.class_to_idx[class_name])
        else:
            # Load validation data
            val_dir = os.path.join(self.data_dir, 'val')
            val_annotations = os.path.join(val_dir, 'val_annotations.txt')
            
            if os.path.exists(val_annotations):
                with open(val_annotations, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        img_name = parts[0]
                        class_name = parts[1]
                        img_path = os.path.join(val_dir, 'images', img_name)
                        if os.path.exists(img_path):
                            self.data.append(img_path)
                            self.targets.append(self.class_to_idx[class_name])
        
        self.targets = np.array(self.targets)
        
    def _download_and_extract(self):
        # This method is kept for compatibility but not used when using existing data
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        filename = os.path.join(self.root, 'tiny-imagenet-200.zip')
        
        print('Downloading TinyImageNet dataset...')
        urllib.request.urlretrieve(url, filename)
        
        print('Extracting TinyImageNet dataset...')
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        
        # Remove the zip file
        os.remove(filename)
        print('TinyImageNet dataset downloaded and extracted successfully.')
    
    def __getitem__(self, index):
        img_path = self.data[index]
        target = self.targets[index]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, target
    
    def __len__(self):
        return len(self.data)

def statstic_info(dataset: Dataset):
    sample_num = len(dataset)
    num_classes = len(dataset.classes)
    class_sample_num = [0 for _ in range(num_classes)]
    
    # For TinyImageNet and other datasets that have targets as numpy array,
    # use targets directly instead of iterating through __getitem__
    if hasattr(dataset, 'targets') and isinstance(dataset.targets, (np.ndarray, list)):
        for label in dataset.targets:
            class_sample_num[label] += 1
    else:
        # Fallback to the original method for datasets without targets attribute
        for _, label in dataset:
            class_sample_num[label] += 1
    
    return sample_num, num_classes, class_sample_num

def get_subset(dataset, idxs) -> Dataset:
    subset = copy.deepcopy(dataset)
    import numpy as np
    idxs = np.array(idxs, dtype=int)
    
    # Handle targets (always numpy array)
    subset.targets = subset.targets[idxs]
    
    # Handle data - different datasets have different data types
    if isinstance(subset.data, np.ndarray):
        # For datasets like CIFAR, MNIST where data is numpy array
        subset.data = subset.data[idxs]
    elif isinstance(subset.data, list):
        # For datasets like TinyImageNet where data is list of file paths
        subset.data = [subset.data[i] for i in idxs]
    else:
        # Fallback: try numpy indexing
        subset.data = subset.data[idxs]
    
    return subset

def get_dataset(name, data_root='../datasets/'):
    if name == 'cifar10':
        transforms_cifar_train = transforms.Compose([
            # transforms.RandomResizedCrop((224, 224)), # For ViT
            transforms.RandomCrop(32, padding=4), # For ResNet
            transforms.RandomHorizontalFlip(), # For ResNet
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # normalization
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
        ])

        transforms_cifar_test = transforms.Compose([
            # transforms.Resize((224, 224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # normalization
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalization
        ])

        train_set = datasets.CIFAR10(
            root = data_root + 'CIFAR',
            train = True,                         
            transform = transforms_cifar_train, 
            download = True,            
        )
        train_set.targets = np.array(train_set.targets)

        test_set = datasets.CIFAR10(
            root = data_root + 'CIFAR', 
            train = False, 
            transform = transforms_cifar_test,
            download = True,  
        )
        test_set.targets = np.array(test_set.targets)

    elif name == 'cifar100':
        transforms_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                (0.26733428587941854, 0.25643846292120615, 0.2761504713263903)
            )
        ])
        transforms_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                (0.26733428587941854, 0.25643846292120615, 0.2761504713263903)
            )
        ])

        train_set = datasets.CIFAR100(
            root = data_root + 'CIFAR',
            train = True,
            transform = transforms_cifar_train,
            download = True,
        )
        train_set.targets = np.array(train_set.targets)

        test_set = datasets.CIFAR100(
            root = data_root + 'CIFAR', 
            train = False,
            transform = transforms_cifar_test,
            download = True,
        )
        test_set.targets = np.array(test_set.targets)

    elif name == 'mnist':
        train_set = datasets.MNIST(
            root = data_root + 'MNIST',
            train = True,                         
            transform = transforms.ToTensor(),
            download = True,            
        )
        test_set = datasets.MNIST(
            root = data_root + 'MNIST', 
            train = False, 
            transform = transforms.ToTensor(),
            download = True,
        )
    elif name == 'mnistFashion':
        train_set = datasets.FashionMNIST(
            root = data_root + 'FMNIST',
            train = True,                         
            transform = transforms.ToTensor(),
            download = True,            
        )
        test_set = datasets.FashionMNIST(
            root = data_root + 'FMNIST', 
            train = False, 
            transform = transforms.ToTensor(),
            download = True,
        )
    elif name == 'mnistKuzushiji':
        train_set = KuzushijiMNIST(
            root = data_root + 'KMNIST',
            train = True,                         
        )
        test_set = KuzushijiMNIST(
            root = data_root + 'KMNIST', 
            train = False, 
            download = True,
        )
    elif name == 'tinyimagenet':
        # TinyImageNet transforms
        transforms_tiny_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        transforms_tiny_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        # Use the existing tiny-imagenet-200 directory directly
        tiny_imagenet_path = os.path.expanduser(data_root + 'tiny-imagenet-200')
        
        train_set = TinyImageNet(
            root = tiny_imagenet_path,
            train = True,
            transform = transforms_tiny_train,
            download = False
        )
        
        test_set = TinyImageNet(
            root = tiny_imagenet_path,
            train = False,
            transform = transforms_tiny_test,
            download = False
        )
    

    return train_set, test_set

def get_dataloader(dataset, batch_size, shuffle):
    loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        num_workers = 4,
        pin_memory = True
    )

    return loader