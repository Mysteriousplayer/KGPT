import logging
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import os

class DataManager(object):
    def __init__(self, args=None):
        self.args = args
        self.dataset_name = args['dataset']

    def get_dataset(self):
        if self.dataset_name == 'cifar':
            trainset, testset = self.build_dataset_cifar()
            return trainset, testset
        elif self.dataset_name == 'cifar10':
            trainset, testset = self.build_dataset_cifar10()
            return trainset, testset
        elif self.dataset_name == 'cars':
            trainset, testset = self.build_dataset_cars()
            return trainset, testset
        elif self.dataset_name == 'dtd':
            trainset, testset = self.build_dataset_dtd()
            return trainset, testset
        elif self.dataset_name == 'sat':
            trainset, testset = self.build_dataset_sat()
            return trainset, testset
        elif self.dataset_name == 'aircraft':
            trainset, testset = self.build_dataset_aircraft()
            return trainset, testset
        elif self.dataset_name == 'flower':
            trainset, testset = self.build_dataset_flower()
            return trainset, testset
        elif self.dataset_name == 'nwpu':
            trainset, testset = self.build_dataset_nwpu()
            return trainset, testset
        elif self.dataset_name == 'pattern':
            trainset, testset = self.build_dataset_pattern()
            return trainset, testset
        elif self.dataset_name == 'Imagenet':
            trainset, testset = self.build_dataset_imagenet()
            return trainset, testset
        elif self.dataset_name == 'dog':
            trainset, testset = self.build_dataset_dog()
            return trainset, testset
        elif self.dataset_name == 'ucf':
            trainset, testset = self.build_dataset_ucf()
            return trainset, testset


    def build_dataset_cifar(self):
        transform_train = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        trainset = datasets.CIFAR100(root=self.args['data_path'], train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=self.args['data_path'], train=False, download=True, transform=transform_test)

        return trainset, testset

    def build_dataset_cifar10(self):
        transform_train = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        trainset = datasets.CIFAR10(root=self.args['data_path'], train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=self.args['data_path'], train=False, download=True, transform=transform_test)

        return trainset, testset

    def build_dataset_cars(self):
        transform_train = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),

        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dir = os.path.join(self.args['data_path'], 'car_train')
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = os.path.join(self.args['data_path'], 'car_test')
        testset = datasets.ImageFolder(test_dir, transform=transform_test)
        return trainset, testset

    def build_dataset_dtd(self):
        transform_train = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),

        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dir = os.path.join(self.args['data_path'], 'train')
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = os.path.join(self.args['data_path'], 'test')
        testset = datasets.ImageFolder(test_dir, transform=transform_test)
        return trainset, testset

    def build_dataset_sat(self):
        transform_train = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dir = os.path.join(self.args['data_path'], 'train')
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = os.path.join(self.args['data_path'], 'test')
        testset = datasets.ImageFolder(test_dir, transform=transform_test)
        return trainset, testset

    def build_dataset_aircraft(self):
        transform_train = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dir = os.path.join(self.args['data_path'], 'train')
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = os.path.join(self.args['data_path'], 'test')
        testset = datasets.ImageFolder(test_dir, transform=transform_test)
        return trainset, testset

    def build_dataset_flower(self):
        transform_train = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dir = os.path.join(self.args['data_path'], 'train')
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = os.path.join(self.args['data_path'], 'test')
        testset = datasets.ImageFolder(test_dir, transform=transform_test)
        return trainset, testset


    def build_dataset_nwpu(self):
        transform_train = transforms.Compose([
            transforms.Resize(256, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dir = os.path.join(self.args['data_path'], 'train')
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = os.path.join(self.args['data_path'], 'test')
        testset = datasets.ImageFolder(test_dir, transform=transform_test)
        return trainset, testset

    def build_dataset_pattern(self):
        transform_train = transforms.Compose([
            transforms.Resize(256, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dir = os.path.join(self.args['data_path'], 'train')
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = os.path.join(self.args['data_path'], 'test')
        testset = datasets.ImageFolder(test_dir, transform=transform_test)
        return trainset, testset


    def build_dataset_imagenet(self):
        transform_train = transforms.Compose([
            transforms.RandomSizedCrop(224, interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dir = os.path.join(self.args['data_path'], 'train')
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = os.path.join(self.args['data_path'], 'val')
        testset = datasets.ImageFolder(test_dir, transform=transform_test)
        return trainset, testset

    def build_dataset_dog(self):
        transform_train = transforms.Compose([
            transforms.RandomSizedCrop(224, interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dir = os.path.join(self.args['data_path'], 'train')
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = os.path.join(self.args['data_path'], 'test')
        testset = datasets.ImageFolder(test_dir, transform=transform_test)
        return trainset, testset

    def build_dataset_ucf(self):
        transform_train = transforms.Compose([
            transforms.RandomSizedCrop(224, interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        train_dir = os.path.join(self.args['data_path'], 'train')
        trainset = datasets.ImageFolder(train_dir, transform=transform_train)
        test_dir = os.path.join(self.args['data_path'], 'test')
        testset = datasets.ImageFolder(test_dir, transform=transform_test)
        return trainset, testset

