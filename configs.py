import torch
from torchvision import transforms
import argparse

from torchvision import transforms

def transformation():
    data_transforms = {
        'train': transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ]),
        'validation': transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ]),
    }
    return data_transforms

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--k_fold', type=int,
                        default=5, help='K Fold')
    parser.add_argument('--epochs', type=int,
                        default=100, help='Number of epochs per fold')
    parser.add_argument('--gradient_clipping', type=float,
                        default=3.0, help='Gradient clipping to prevent gradient explosion')
    parser.add_argument('--lr', type=float,
                        default=0.001, help='Learning rate')
    parser.add_argument('--tau', type=float,
                        default=100.0, help='Tau parameter for clip')
    parser.add_argument('--device', type=str,
                        default='cuda', help='Device (cpu/cuda)')
    parser.add_argument('--image_directory', type=str,
                        default='LIVE_Challenge/Images', help='LIVE Challenge dataset images')
    parser.add_argument('--model_directory', type=str,
                        default='trained models/', help='Folder path for saving the trained models')

    optn = parser.parse_args()

    return optn