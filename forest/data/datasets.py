"""Super-classes of common datasets to extract id information per image."""
import torch
import torchvision
import numpy as np

from ..consts import *   # import all mean/std constants

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import os
import glob

from torchvision.datasets.imagenet import load_meta_file
from torchvision.datasets.utils import verify_str_arg

# Block ImageNet corrupt EXIF warnings
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

user_name = "xxx" # mask this with xxx for blind review!
repo_dir = f"/data/{user_name}/github/poisoning-gradient-matching/open_source_delta_influence" # point this to your directory if neccessary

class ImagenetteDataset(object):
    def __init__(self, patch_size=160, validation=False, should_normalize=True, target_transform=None):
        self.imagenette_path = '/data/xxx/github/poisoning-gradient-matching/open_source/badnet/imagenette/exp01/data/imagenette2-160'
        self.folder = Path(self.imagenette_path+'/train') if not validation else Path(self.imagenette_path+'/val')
        self.classes = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079',
                        'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']

        self.images = []
        for cls in self.classes:
            cls_images = list(self.folder.glob(cls + '/*.JPEG'))
            self.images.extend(cls_images)
        
        self.patch_size = patch_size
        self.validation = validation
        
        # self.random_resize = torchvision.transforms.RandomResizedCrop(patch_size)
        self.center_resize = torchvision.transforms.CenterCrop(patch_size)
        self.should_normalize = should_normalize
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        image_fname = self.images[index]
        image = Image.open(image_fname)
        label = image_fname.parent.stem
        label = self.classes.index(label)
        
        # if not self.validation: image = self.random_resize(image)
        # else: image = self.center_resize(image)
        image = self.center_resize(image)
            
        image = torchvision.transforms.functional.to_tensor(image)
        if image.shape[0] == 1: image = image.expand(3, 160, 160)
        if self.should_normalize: image = self.normalize(image)
        
        return image, label, index

    def get_target(self, index):

        image_fname = self.images[index]
        # image = Image.open(image_fname)
        label = image_fname.parent.stem
        label = self.classes.index(label)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return label, index

    def __len__(self):
        return len(self.images)
####
def construct_datasets(dataset, data_path, normalize=True):
    """Construct datasets with appropriate transforms."""
    ######################
    # exp = 'exp11'    # <--
    ######################
    # Compute mean, std:
    if dataset == 'CIFAR100':
        # trainset = CIFAR100(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        trainset = CIFAR100(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        if cifar100_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = cifar100_mean, cifar100_std
    #######################################################################################################################
    elif dataset == 'CIFAR10':
        trainset = CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        print(f"cifar10_mean: {cifar10_mean}")
        if cifar10_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = cifar10_mean, cifar10_std
        print(f"cifar10_mean: {cifar10_mean}")
    elif dataset == 'Imagenette':
        # Load clean imagenette dataset
        """
        if not os.path.isfile('/data/xxx/github/poisoning-gradient-matching/open_source/smooth_trigger/imagenette/exp01/data/imagenette2-160.tgz'):
            !wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
            !tar -xvf imagenette2-160.tgz
        else:
            print("imagenette already downloaded")
        """
        # initialize trainset
        trainset = ImagenetteDataset(160, should_normalize=True)
        print(f"trainset: {trainset}")
        
        imagenette_mean = [0.485, 0.456, 0.406]
        imagenette_std = [0.229, 0.224, 0.225]
        data_mean, data_std = imagenette_mean, imagenette_std
        print(f"imagenette_mean: {imagenette_mean}")
    elif dataset == 'Imagenette_filtered':
        ##
        exp = 'frequency_attack/imagenette/exp03'
        poison_data_path = f'/data/xxx/github/poisoning-gradient-matching/opensource_ready/experiments/{exp}/data'
        manip_idx_path = f'/data/xxx/github/poisoning-gradient-matching/opensource_ready/experiments/{exp}/poison_info/manip_idx.npy'
        manip_idx = np.load(manip_idx_path)
        ##
        patched_images_tensor = torch.load(os.path.join(poison_data_path, 'patched_images.pt'))  # --> already normalized
        patched_labels_tensor = torch.load(os.path.join(poison_data_path, 'patched_labels.pt'))
        # print(f"patched_images_tensor.shape: {patched_images_tensor.shape}")
        # print(f"patched_labels_tensor.shape: {patched_labels_tensor.shape}")
        # patched_dataset = torch.utils.data.TensorDataset(patched_images_tensor, patched_labels_tensor)     
        # """                                                                                                                   
        ########################################################################################################################
        detected_idx_path = '/data/xxx/github/poisoning-gradient-matching/opensource_ready/experiments/frequency_attack/imagenette/exp03/detected/aux_ML_50_detected_indices.npy'
        detected_idx = np.load(detected_idx_path)
        ########################################################################################################################
        print(f"num detected: {len(detected_idx)}")
        print(f"detected indices loaded from {detected_idx_path}")
        # filter detected samples
        filtered_images = torch.stack([img for i, img in enumerate(patched_images_tensor) if i not in detected_idx])  # Stack images
        filtered_labels = torch.tensor([label for i, label in enumerate(patched_labels_tensor) if i not in detected_idx])  # Get labels
        # """
        # trainset = Imagenette_Custom(patched_images_tensor, patched_labels_tensor, transform=None)
        trainset = Imagenette_Custom(filtered_images, filtered_labels, transform=None)
        """
        print("=== trainset = Imagenette_Custom(patched_images_tensor, patched_labels_tensor, transform=None) ===")
        print(f"trainset[0]: {trainset[0]}")
        print(f"trainset[1]: {trainset[1]}")
        """
        print(f'Imagenette_filtered dataset loaded... ({exp} [#manip: {len(manip_idx)}])')
        print(f"trainset size: {len(trainset)}")
        
        imagenette_mean = [0.485, 0.456, 0.406]
        imagenette_std = [0.229, 0.224, 0.225]
        data_mean, data_std = imagenette_mean, imagenette_std
        print(f"imagenette_mean: {imagenette_mean}")
    elif dataset == 'Imagenette_WB_filtered':                                                               ### Imagenette_WB_filtered ### 
        # trainset = CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())  
        # trainset = torchvision.datasets.ImageFolder('/data/xxx/github/poisoning-gradient-matching/poisons/retrain', transform = transforms.ToTensor())
        retrain_folder = '/data/xxx/github/poisoning-gradient-matching/opensource_ready/experiments/witchesbrew/imagenette/exp01_recheck/unlearn_EU/filtered_comp/train'
        trainset = CIFAR10_Custom(root=retrain_folder, transform=transforms.ToTensor()) # we can directly use the class for cifat10_custom here
        print('filtered dataset loaded...')
        print(trainset)

        imagenette_mean = [0.485, 0.456, 0.406]
        imagenette_std = [0.229, 0.224, 0.225]
        data_mean, data_std = imagenette_mean, imagenette_std
        print(f"imagenette_mean: {imagenette_mean}")
    #######################################################################################################################
    elif dataset == 'CIFAR10_Randomly_Removed':
        # trainset = CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        # trainset = torchvision.datasets.ImageFolder('/data/xxx/github/poisoning-gradient-matching/poisons/retrain', transform = transforms.ToTensor())
        retrain_folder = '/data/xxx/github/poisoning-gradient-matching/lab01_collect/randomly_remove_431_dataset/train/'
        trainset = CIFAR10_Custom(root=retrain_folder, transform=transforms.ToTensor())
        print('filtered dataset loaded...')
        print(trainset)

        print(f"cifar10_mean: {cifar10_mean}")
        if cifar10_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = cifar10_mean, cifar10_std
        print(f"cifar10_mean: {cifar10_mean}")
    #######################################################################################################################
    elif dataset == 'CIFAR10_Filtered':                                                               ### CIFAR10 ### 
        # trainset = CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())  
        # trainset = torchvision.datasets.ImageFolder('/data/xxx/github/poisoning-gradient-matching/poisons/retrain', transform = transforms.ToTensor())
        retrain_folder = '/data/xxx/github/poisoning-gradient-matching/open_source/wichesbrew/exp08/filtered_aux_ML_50/train'
        trainset = CIFAR10_Custom(root=retrain_folder, transform=transforms.ToTensor())
        print('filtered dataset loaded...')
        print(trainset)

        print(f"cifar10_mean: {cifar10_mean}")
        if cifar10_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = cifar10_mean, cifar10_std
        print(f"cifar10_mean: {cifar10_mean}")

    elif dataset == 'CIFAR100_Filtered': # [CIFAR100]
        # trainset = CIFAR10(root=data_path, train=True, download=True, transform=transforms.ToTensor())  ### CIFAR100 ###
        # trainset = torchvision.datasets.ImageFolder('/data/xxx/github/poisoning-gradient-matching/poisons/retrain', transform = transforms.ToTensor())
        retrain_folder = '/data/xxx/github/poisoning-gradient-matching/open_source/wichesbrew/cifar100/exp03/filtered_th/train'
        trainset = CIFAR10_Custom(root=retrain_folder, transform=transforms.ToTensor())
        print('filtered dataset loaded...')
        print(trainset)

        if cifar100_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = cifar100_mean, cifar100_std
        print(f"cifar100_mean: {cifar100_mean}")

    elif dataset == 'CIFAR10_BadNet_Patch':                                                                
        exp = 'badnet/cifar10/exp01'
        poison_data_path = f'{repo_dir}/notebooks/{exp}/data'
        patched_images_tensor = torch.load(os.path.join(poison_data_path, 'patched_images.pt'))  
        patched_labels_tensor = torch.load(os.path.join(poison_data_path, 'patched_labels.pt'))
        # train transform
        transform_train_BadNet = transforms.Compose([ 
            transforms.RandomHorizontalFlip(),       
            transforms.RandomVerticalFlip(),      
            transforms.ToTensor(),                 
            transforms.Normalize((0.49147, 0.48226, 0.44677), (0.24703, 0.24349, 0.26159))
        ])
        
        ########################################################################################################################  # <------- uncomment this when unlearn
        detected_idx_path = '/data/xxx/github/poisoning-gradient-matching/open_source_delta_influence/notebooks/badnet/cifar10/exp01/detected/aux_MM_50_detected_indices.npy'
        detected_idx = np.load(detected_idx_path)
        ########################################################################################################################
        print(f"num detected: {len(detected_idx)}")
        print(f"detected indices loaded from {detected_idx_path}")
        # filter detected samples
        filtered_images = torch.stack([img for i, img in enumerate(patched_images_tensor) if i not in detected_idx])  # Stack images
        filtered_labels = torch.tensor([label for i, label in enumerate(patched_labels_tensor) if i not in detected_idx])  # Get labels

        trainset = CIFAR10_BadNet_Custom(filtered_images, filtered_labels, transform=transform_train_BadNet)
        # trainset = CIFAR10_BadNet_Custom(patched_images_tensor, patched_labels_tensor, transform=transform_train_BadNet) # use this when trian the victim model
        
        print(f'CIFAR10_BadNet_Patch dataset loaded... ({exp})')
        print(f"trainset size: {len(trainset)}")
        print(f"cifar10_mean: {cifar10_mean}") 
        if cifar10_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = cifar10_mean, cifar10_std
    
    elif dataset == 'CIFAR10_Blend_Random':                                                               ### CIFAR10 ### 
        exp = 'exp01_MM'
        poison_data_path = f'/data/xxx/github/poisoning-gradient-matching/detect_vault/blend_random/cifar10/{exp}/data'
        
        patched_images_tensor = torch.load(os.path.join(poison_data_path, 'patched_images.pt'))  # --> this not normalized yet
        patched_labels_tensor = torch.load(os.path.join(poison_data_path, 'patched_labels.pt'))
        # patched_dataset = torch.utils.data.TensorDataset(patched_images_tensor, patched_labels_tensor)
                                                                 ### CIFAR10 ### 
        """                                                                 
        ########################################################################################################################
        # detected_idx_path = f'/data/xxx/github/poisoning-gradient-matching/detect_vault/smooth_trigger/{exp}/detected/freq_detected_indices.npy'
        detected_idx_path = '/data/xxx/github/poisoning-gradient-matching/detect_vault/badnet_patching/cifar10/exp02_MM/detected/aux_MM_50_detected_indices.npy'
        detected_idx = np.load(detected_idx_path)
        ########################################################################################################################
        print(f"num detected: {len(detected_idx)}")
        print(f"detected indices loaded from {detected_idx_path}")
        # filter detected samples
        filtered_images = torch.stack([img for i, img in enumerate(patched_images_tensor) if i not in detected_idx])  # Stack images
        filtered_labels = torch.tensor([label for i, label in enumerate(patched_labels_tensor) if i not in detected_idx])  # Get labels
        """
        # train transform
        transform_train_BlendRandom = transforms.Compose([ 
            transforms.RandomHorizontalFlip(),       # comment flip when testing
            transforms.RandomVerticalFlip(),      
            transforms.ToTensor(),                 
            transforms.Normalize((0.49147, 0.48226, 0.44677), (0.24703, 0.24349, 0.26159))  # Normalize the image
        ])
        
        # trainset = CIFAR10_BadNet_Custom(filtered_images, filtered_labels, transform=transform_train_BadNet)
        trainset = CIFAR10_BadNet_Custom(patched_images_tensor, patched_labels_tensor, transform=transform_train_BlendRandom)
        
        print(f'CIFAR10_Blend_Random dataset loaded... ({exp})')
        print(f"trainset size: {len(trainset)}")
        
        print(f"cifar10_mean: {cifar10_mean}") # --> useless
        if cifar10_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = cifar10_mean, cifar10_std
    
    elif dataset == 'CIFAR100_BadNet_Patch':                                                               ### CIFAR10 ### 
        exp = 'cifar100/exp01'
        poison_data_path = f'/data/xxx/github/poisoning-gradient-matching/open_source/badnet/{exp}/data'
        
        patched_images_tensor = torch.load(os.path.join(poison_data_path, 'patched_images.pt'))  # --> this not normalized yet
        patched_labels_tensor = torch.load(os.path.join(poison_data_path, 'patched_labels.pt'))
                                                                 ### CIFAR10 ###                                                   
        ########################################################################################################################
        # detected_idx_path = f'/data/xxx/github/poisoning-gradient-matching/detect_vault/smooth_trigger/{exp}/detected/freq_detected_indices.npy'
        # detected_save_dir = f'/data/xxx/github/poisoning-gradient-matching/detect_vault/badnet_patching/{exp}/detected/aux_MM_{n_boosting}_detected_indices.npy'
        detected_idx_path = '/data/xxx/github/poisoning-gradient-matching/open_source/badnet/cifar100/exp01/detected/aux_ML_50_detected_indices.npy' 
        detected_idx = np.load(detected_idx_path)
        ########################################################################################################################
        print(f"num detected: {len(detected_idx)}")
        print(f"detected indices loaded from {detected_idx_path}")
        # filter detected samples
        filtered_images = torch.stack([img for i, img in enumerate(patched_images_tensor) if i not in detected_idx])  # Stack images
        filtered_labels = torch.tensor([label for i, label in enumerate(patched_labels_tensor) if i not in detected_idx])  # Get labels
        
        # train transform
        transform_train_BadNet = transforms.Compose([ 
            transforms.RandomHorizontalFlip(),       # comment flip when testing
            transforms.RandomVerticalFlip(),      
            transforms.ToTensor(),                 
            transforms.Normalize((0.50716, 0.48669, 0.44120), (0.26733, 0.25644, 0.27615))  # Normalize the image
        ])
        trainset = CIFAR10_BadNet_Custom(filtered_images, filtered_labels, transform=transform_train_BadNet)
        # trainset = CIFAR10_BadNet_Custom(patched_images_tensor, patched_labels_tensor, transform=transform_train_BadNet)
        
        print(f'CIFAR100_BadNet_Patch dataset loaded... ({exp})')
        print(f"trainset size: {len(trainset)}")
        
        if cifar100_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = cifar100_mean, cifar100_std
        print(f"cifar100_mean: {cifar100_mean}")

    elif dataset == 'CIFAR10_SmoothTrigger':                                                               ### CIFAR10 ### 
        ########################
        exp = 'cifar10/exp03'
        ########################
        # poison_data_path = f'/data/xxx/github/poisoning-gradient-matching/open_source/smooth_trigger/{exp}/data'
        poison_data_path = '/data/xxx/github/poisoning-gradient-matching/detect_vault/smooth_trigger_go_colab/exp02_MM/data'
        
        patched_images_tensor = torch.load(os.path.join(poison_data_path, 'patched_images.pt'))
        patched_labels_tensor = torch.load(os.path.join(poison_data_path, 'patched_labels.pt'))
        # patched_dataset = torch.utils.data.TensorDataset(patched_images_tensor, patched_labels_tensor)
                                                                 ### CIFAR10 ### 
        ########################################################################################################################
        # detected_idx_path = f'/data/xxx/github/poisoning-gradient-matching/detect_vault/smooth_trigger/{exp}/detected/freq_detected_indices.npy'
        detected_idx_path = '/data/xxx/github/poisoning-gradient-matching/detect_vault/smooth_trigger_go_colab/exp02_MM/detected/aux_MI_50_detected_indices.npy'
        detected_idx = np.load(detected_idx_path)
        ########################################################################################################################
        print(f"num detected: {len(detected_idx)}")
        print(f"detected indices loaded from {detected_idx_path}")
        # filter detected samples
        filtered_images = torch.stack([img for i, img in enumerate(patched_images_tensor) if i not in detected_idx])  # Stack images
        filtered_labels = torch.tensor([label for i, label in enumerate(patched_labels_tensor) if i not in detected_idx])  # Get labels
        
        # Create a new DataLoader for the loaded data
        trainset = CIFAR10_ST_Custom(filtered_images, filtered_labels)
        # train_loader = DataLoader(loaded_set, batch_size=32, shuffle=False)
    
        print('CIFAR10_SmoothTrigger dataset loaded...')
        print(f"filtered trainset size: {len(trainset)} {exp}")
        
        print(f"cifar10_mean: {cifar10_mean}") # --> useless
        if cifar10_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = cifar10_mean, cifar10_std

    elif dataset == 'CIFAR10_ST_Debug':     
        exp = "exp03_MM_N50" 
        poison_data_path = f'/data/xxx/github/poisoning-gradient-matching/detect_vault/smooth_trigger_go_colab/{exp}/data'
        patched_images_tensor = torch.load(os.path.join(poison_data_path, 'patched_images.pt'))
        patched_labels_tensor = torch.load(os.path.join(poison_data_path, 'patched_labels.pt'))
        # patched_dataset = torch.utils.data.TensorDataset(patched_images_tensor, patched_labels_tensor)
                                                                 ### CIFAR10 ### 
        ########################################################################################################################
        # detected_idx = np.load('/data/xxx/github/poisoning-gradient-matching/detect_vault/smooth_trigger/exp01/st2_freq_detected_indices.npy')
        ########################################################################################################################
        # print(f"num detected: {len(detected_idx)}")
        # images = torch.stack([img for i, (img, _) in enumerate(loaded_train_data) if i not in detected_idx])  # Stack images
        # labels = torch.tensor([label for i, (_, label) in enumerate(loaded_train_data) if i not in detected_idx])  # Get labels
        
        # Create a new DataLoader for the loaded data
        trainset = CIFAR10_ST_Custom(patched_images_tensor, patched_labels_tensor)
        # train_loader = DataLoader(loaded_set, batch_size=32, shuffle=False)
    
        print(f'CIFAR10_ST_Debug dataset loaded... {exp}')
        print(f"trainset size: {len(trainset)}")
        
        print(f"cifar10_mean: {cifar10_mean}") # --> useless
        if cifar10_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = cifar10_mean, cifar10_std
    elif dataset == 'CIFAR100_ST_Debug':     
        exp = 'cifar100/exp01'
        poison_data_path = f'/data/xxx/github/poisoning-gradient-matching/open_source/smooth_trigger/{exp}/data'
        patched_images_tensor = torch.load(os.path.join(poison_data_path, 'patched_images.pt'))
        patched_labels_tensor = torch.load(os.path.join(poison_data_path, 'patched_labels.pt'))
        # patched_dataset = torch.utils.data.TensorDataset(patched_images_tensor, patched_labels_tensor)
                                                                 ### CIFAR100 ### 
                                                                 
        ########################################################################################################################
        detected_idx_path = '/data/xxx/github/poisoning-gradient-matching/open_source/smooth_trigger/cifar100/exp01/detected/aux_ML_50_detected_indices.npy'
        detected_idx = np.load(detected_idx_path)
        ########################################################################################################################
        print(f"num detected: {len(detected_idx)}")
        filtered_images = torch.stack([img for i, img in enumerate(patched_images_tensor) if i not in detected_idx])  # Stack images
        filtered_labels = torch.tensor([label for i, label in enumerate(patched_labels_tensor) if i not in detected_idx])  # Get labels
        
        # Create a new DataLoader for the loaded data
        trainset = CIFAR10_ST_Custom(filtered_images, filtered_labels)
        # trainset = CIFAR10_ST_Custom(patched_images_tensor, patched_labels_tensor)
        
        print(f'CIFAR100_ST_Debug dataset loaded... ({exp})')
        print(f"trainset size: {len(trainset)}")
        
        if cifar100_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = cifar100_mean, cifar100_std
        print(f"cifar100_mean: {cifar100_mean}") # --> useless

    #######################################################################################################################
    elif dataset == 'MNIST':
        trainset = MNIST(root=data_path, train=True, download=True, transform=transforms.ToTensor())
        if mnist_mean is None:
            cc = torch.cat([trainset[i][0].reshape(-1) for i in range(len(trainset))], dim=0)
            data_mean = (torch.mean(cc, dim=0).item(),)
            data_std = (torch.std(cc, dim=0).item(),)
        else:
            data_mean, data_std = mnist_mean, mnist_std
    elif dataset == 'ImageNet':
        trainset = ImageNet(root=data_path, split='train', download=False, transform=transforms.ToTensor())
        if imagenet_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = imagenet_mean, imagenet_std
    elif dataset == 'ImageNet1k':
        trainset = ImageNet1k(root=data_path, split='train', download=False, transform=transforms.ToTensor())
        if imagenet_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = imagenet_mean, imagenet_std
    elif dataset == 'TinyImageNet':
        trainset = TinyImageNet(root=data_path, split='train', transform=transforms.ToTensor())
        if tiny_imagenet_mean is None:
            cc = torch.cat([trainset[i][0].reshape(3, -1) for i in range(len(trainset))], dim=1)
            data_mean = torch.mean(cc, dim=1).tolist()
            data_std = torch.std(cc, dim=1).tolist()
        else:
            data_mean, data_std = tiny_imagenet_mean, tiny_imagenet_std
    else:
        raise ValueError(f'Invalid dataset {dataset} given.')

    if normalize: # True
        print(f'Data mean is {data_mean}, \nData std  is {data_std}.')
        trainset.data_mean = data_mean
        trainset.data_std = data_std
    else:
        print('Normalization disabled.')
        trainset.data_mean = (0.0, 0.0, 0.0)
        trainset.data_std = (1.0, 1.0, 1.0)

    # Setup data
    if dataset in ['ImageNet', 'ImageNet1k']:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
    # elif dataset in ['Imagenette']: # <--- we have already resize the images when load data
    #     transform_train = transforms.Compose([
    #     transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])

    ##############################
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x)])
            transforms.Normalize(trainset.data_mean, trainset.data_std) if normalize else transforms.Lambda(lambda x: x)])

    trainset.transform = transform_train 
    # Note: this transform will be applied later
    # therefore, if your xxx_custom have "transorm" property (like CIFAR10_BadNet_Custom). Make sure you trainset is not normalized
    ##############################

    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])

    if dataset == 'CIFAR100':
        validset = CIFAR100(root=data_path, train=False, download=True, transform=transform_valid)
        # validset = CIFAR100(root=data_path, train=False, download=False, transform=transform_valid)
    ########################################################################################################
    elif dataset == 'CIFAR10':
        validset = CIFAR10(root=data_path, train=False, download=True, transform=transform_valid)
    elif dataset in ['Imagenette', 'Imagenette_WB_filtered', 'Imagenette_filtered']:
        """
        if not os.path.isfile('/data/xxx/github/poisoning-gradient-matching/open_source/smooth_trigger/imagenette/exp01/data/imagenette2-160.tgz'):
            !wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
            !tar -xvf imagenette2-160.tgz
        else:
            print("imagenette already downloaded")
        """
        # initialize validset
        print("imagenette (validset) is imported...")
        validset = ImagenetteDataset(160, validation=True, should_normalize=True) 

    elif dataset == 'CIFAR10_Filtered':                                                     ### CIFAR10 ### 
        # validset = torchvision.datasets.ImageFolder('/data/xxx/github/poisoning-gradient-matching/poisons/retrain', transform = transform_valid)
        # validset = CIFAR10FILTERED('/data/xxx/github/poisoning-gradient-matching/poisons/retrain', transform = transform_valid)
        # test_folder = '/data/xxx/github/poisoning-gradient-matching/detect_vault/exp08/poisons/test' # 9999
        clean_cifar10_dir = '/data/xxx/github/poisoning-gradient-matching/detect_vault/cifar10_original_data'
        # validset = CIFAR10_Custom(root=clean_cifar10_dir, transform=transform_valid)
        validset = CIFAR10(root=clean_cifar10_dir, train=False, download=True, transform=transform_valid)
    elif dataset == 'CIFAR100_Filtered':                                                    ### CIFAR100 ### 
        # validset = torchvision.datasets.ImageFolder('/data/xxx/github/poisoning-gradient-matching/poisons/retrain', transform = transform_valid)
        # validset = CIFAR10FILTERED('/data/xxx/github/poisoning-gradient-matching/poisons/retrain', transform = transform_valid)
        """
        test_folder = '/data/xxx/github/poisoning-gradient-matching/detect_vault/cifar100/exp02/poisons/test' # 9999
        validset = CIFAR10_Custom(root=test_folder, transform=transform_valid)
        """
        clean_CIFAR100_path = '/data/xxx/github/poisoning-gradient-matching/detect_vault/badnet_patching/clean_cifar100/data'
        validset = CIFAR100(root=clean_CIFAR100_path, train=False, download=True, transform=transform_valid)
    elif dataset == 'CIFAR10_SmoothTrigger':
        CIFAR10_ST_Debug_data_path = '/data/xxx/github/poisoning-gradient-matching/detect_vault/smooth_trigger_go_colab/cifar10_data/data'
        validset = CIFAR10(root=CIFAR10_ST_Debug_data_path, train=False, download=True, transform=transform_valid) # full 10k clean test images
        # CIFAR100_ST_Debug_data_path = '/data/xxx/github/poisoning-gradient-matching/detect_vault/smooth_trigger/cifar100/clean_data'
        # validset = CIFAR100(root=CIFAR100_ST_Debug_data_path, train=False, download=True, transform=transform_valid)
    elif dataset == 'CIFAR10_ST_Debug':
        CIFAR10_ST_Debug_data_path = '/data/xxx/github/poisoning-gradient-matching/detect_vault/smooth_trigger/exp00/data/clean_cifar10'
        validset = CIFAR10(root=CIFAR10_ST_Debug_data_path, train=False, download=True, transform=transform_valid)
    elif dataset == 'CIFAR10_BadNet_Patch':
        CIFAR10_ST_Debug_data_path = f'{repo_dir}/clean_data/cifar10'
        validset = CIFAR10(root=CIFAR10_ST_Debug_data_path, train=False, download=True, transform=transform_valid)
    elif dataset == 'CIFAR10_Blend_Random':
        CIFAR10_ST_Debug_data_path = '/data/xxx/github/poisoning-gradient-matching/detect_vault/smooth_trigger/exp00/data/clean_cifar10'
        validset = CIFAR10(root=CIFAR10_ST_Debug_data_path, train=False, download=True, transform=transform_valid)
    elif dataset == 'CIFAR100_ST_Debug':
        CIFAR100_ST_Debug_data_path = '/data/xxx/github/poisoning-gradient-matching/open_source/smooth_trigger/cifar100/clean_data'
        validset = CIFAR100(root=CIFAR100_ST_Debug_data_path, train=False, download=True, transform=transform_valid)
    elif dataset == 'CIFAR100_BadNet_Patch':
        CIFAR100_ST_Debug_data_path = '/data/xxx/github/poisoning-gradient-matching/detect_vault/badnet_patching/clean_cifar100/data'
        validset = CIFAR100(root=CIFAR100_ST_Debug_data_path, train=False, download=True, transform=transform_valid)
    ########################################################################################################
    elif dataset == 'MNIST':
        validset = MNIST(root=data_path, train=False, download=True, transform=transform_valid)
    elif dataset == 'TinyImageNet':
        validset = TinyImageNet(root=data_path, split='val', transform=transform_valid)
    elif dataset == 'ImageNet':
        # Prepare ImageNet beforehand in a different script!
        # We are not going to redownload on every instance
        transform_valid = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        validset = ImageNet(root=data_path, split='val', download=False, transform=transform_valid)
    elif dataset == 'ImageNet1k':
        # Prepare ImageNet beforehand in a different script!
        # We are not going to redownload on every instance
        transform_valid = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        validset = ImageNet1k(root=data_path, split='val', download=False, transform=transform_valid)

    if normalize:
        validset.data_mean = data_mean
        validset.data_std = data_std
    else:
        validset.data_mean = (0.0, 0.0, 0.0)
        validset.data_std = (1.0, 1.0, 1.0)

    return trainset, validset


class Subset(torch.utils.data.Subset):
    """Overwrite subset class to provide class methods of main class."""

    def __getattr__(self, name):
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataset, name)

class CIFAR10(torchvision.datasets.CIFAR10):
    """Super-class CIFAR10 to return image ids with images."""

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index

class CIFAR10_Custom(torchvision.datasets.ImageFolder):
    """Super-class ImageFolder to return image ids with images."""
    
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        
        if self.transform is not None:
            image = self.transform(image)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return image, target, index
    
    def get_target(self, index):
        target = self.samples[index][1]
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return target, index

class CIFAR10_BadNet_Custom(torch.utils.data.Dataset):
    def __init__(self, images, targets, transform=None):
        """
        Args:
            images (Tensor): Tensor containing image data.
            targets (Tensor): Tensor containing target labels.
        """
        # super().__init__(images, targets, transform=transform)  # Initialize base class with images and targets
        self.images = images
        self.targets = targets
        # self.classes = 100 * ["x"] # doesn't matter here  # change here to 10 when dealing with cifar100
        self.classes = 10 * ["x"] 
        self.transform = transform
        self.to_pil = transforms.ToPILImage()

    def __getitem__(self, index):
        """
        Get item from dataset.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.
        """
        img = self.images[index]
        target = self.targets[index]

        if self.transform is not None:
            img = self.to_pil(img)
            img = self.transform(img)
        
        return img, target, index

    def get_target(self, index):
        """
        Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.
        """
        target = self.targets[index]
        return target, index

    def __len__(self):
        return len(self.targets)

class Imagenette_Custom(torch.utils.data.Dataset):
    def __init__(self, images, targets, transform=None):
        """
        Args:
            images (Tensor): Tensor containing image data.
            targets (Tensor): Tensor containing target labels.
        """
        # super().__init__(images, targets, transform=transform)  # Initialize base class with images and targets
        self.images = images
        self.targets = targets
        self.classes = 10 * ["x"] 
        self.transform = transform

    def __getitem__(self, index):
        """
        Get item from dataset.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.
        """
        img = self.images[index]
        target = self.targets[index]
        
        return img, target, index

    def get_target(self, index):
        """
        Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.
        """
        target = self.targets[index]
        return target, index

    def __len__(self):
        return len(self.targets)

class CIFAR10_ST_Custom(TensorDataset):
    """Super-class TensorDataset to return image ids with images."""
    # train_set = TensorDataset(images, labels)
    """Custom dataset to return image tensors with their indices, extending TensorDataset."""

    def __init__(self, images, targets):
        """
        Args:
            images (Tensor): Tensor containing image data.
            targets (Tensor): Tensor containing target labels.
        """
        super().__init__(images, targets)  # Initialize base class with images and targets
        self.images = images
        self.targets = targets
        self.classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        # self.classes = ["x"] * 100

    def __getitem__(self, index):
        """
        Get item from dataset.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.
        """
        img, target = super().__getitem__(index)
        return img, target, index

    def get_target(self, index):
        """
        Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.
        """
        _, target = super().__getitem__(index)
        return target, index


class CIFAR100(torchvision.datasets.CIFAR100):
    """Super-class CIFAR100 to return image ids with images."""

    def __getitem__(self, index):
        """Getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/cifar.html#CIFAR10.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target, idx) where target is index of the target class.

        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = self.targets[index]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index


class MNIST(torchvision.datasets.MNIST):
    """Super-class MNIST to return image ids with images."""

    def __getitem__(self, index):
        """_getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html#MNIST.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.

        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        target = int(self.targets[index])

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index


class ImageNet(torchvision.datasets.ImageNet):
    """Overwrite torchvision ImageNet to change metafile location if metafile cannot be written due to some reason."""

    def __init__(self, root, split='train', download=False, **kwargs):
        """Use as torchvision.datasets.ImageNet."""
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        try:
            wnid_to_classes = load_meta_file(self.root)[0]
        except RuntimeError:
            torchvision.datasets.imagenet.META_FILE = os.path.join(os.path.expanduser('~/data/'), 'meta.bin')
            try:
                wnid_to_classes = load_meta_file(self.root)[0]
            except RuntimeError:
                self.parse_archives()
                wnid_to_classes = load_meta_file(self.root)[0]

        torchvision.datasets.ImageFolder.__init__(self, self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}
        """Scrub class names to be a single string."""
        scrubbed_names = []
        for name in self.classes:
            if isinstance(name, tuple):
                scrubbed_names.append(name[0])
            else:
                scrubbed_names.append(name)
        self.classes = scrubbed_names

    def __getitem__(self, index):
        """_getitem from https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#DatasetFolder.

        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, idx) where target is class_index of the target class.

        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

    def get_target(self, index):
        """Return only the target and its id.

        Args:
            index (int): Index

        Returns:
            tuple: (target, idx) where target is class_index of the target class.

        """
        _, target = self.samples[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index



class ImageNet1k(ImageNet):
    """Overwrite torchvision ImageNet to limit it to less than 1mio examples.

    [limit/per class, due to automl restrictions].
    """

    def __init__(self, root, split='train', download=False, limit=950, **kwargs):
        """As torchvision.datasets.ImageNet except for additional keyword 'limit'."""
        super().__init__(root, split, download, **kwargs)

        # Dictionary, mapping ImageNet1k ids to ImageNet ids:
        self.full_imagenet_id = dict()
        # Remove samples above limit.
        examples_per_class = torch.zeros(len(self.classes))
        new_samples = []
        new_idx = 0
        for full_idx, (path, target) in enumerate(self.samples):
            if examples_per_class[target] < limit:
                examples_per_class[target] += 1
                item = path, target
                new_samples.append(item)
                self.full_imagenet_id[new_idx] = full_idx
                new_idx += 1
            else:
                pass
        self.samples = new_samples
        print(f'Size of {self.split} dataset reduced to {len(self.samples)}.')




"""
    The following class is heavily based on code by Meng Lee, mnicnc404. Date: 2018/06/04
    via
    https://github.com/leemengtaiwan/tiny-imagenet/blob/master/TinyImageNet.py
"""


class TinyImageNet(torch.utils.data.Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.

    Author: Meng Lee, mnicnc404
    Date: 2018/06/04
    References:
        - https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel.html
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    in_memory: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    """

    EXTENSION = 'JPEG'
    NUM_IMAGES_PER_CLASS = 500
    CLASS_LIST_FILE = 'wnids.txt'
    VAL_ANNOTATION_FILE = 'val_annotations.txt'
    CLASSES = 'words.txt'

    def __init__(self, root, split='train', transform=None, target_transform=None):
        """Init with split, transform, target_transform. use --cached_dataset data is to be kept in memory."""
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self.split_dir = os.path.join(root, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % self.EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping

        # build class label - number mapping
        with open(os.path.join(self.root, self.CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(self.NUM_IMAGES_PER_CLASS):
                    self.labels['%s_%d.%s' % (label_text, cnt, self.EXTENSION)] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, self.VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # Build class names
        label_text_to_word = dict()
        with open(os.path.join(root, self.CLASSES), 'r') as file:
            for line in file:
                label_text, word = line.split('\t')
                label_text_to_word[label_text] = word.split(',')[0].rstrip('\n')
        self.classes = [label_text_to_word[label] for label in self.label_texts]

        # Prepare index - label mapping
        self.targets = [self.labels[os.path.basename(file_path)] for file_path in self.image_paths]

    def __len__(self):
        """Return length via image paths."""
        return len(self.image_paths)

    def __getitem__(self, index):
        """Return a triplet of image, label, index."""
        file_path, target = self.image_paths[index], self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        img = Image.open(file_path)
        img = img.convert("RGB")
        img = self.transform(img) if self.transform else img
        if self.split == 'test':
            return img, None, index
        else:
            return img, target, index


    def get_target(self, index):
        """Return only the target and its id."""
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return target, index
