import torch
import torch.nn as nn
from torchvision import transforms
import argparse

from torchvision import transforms

# Transformations
def transformation():
    data_transforms = {
        'train': transforms.Compose([
            transforms.CenterCrop((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ]),
        'validation': transforms.Compose([
            transforms.CenterCrop((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ]),
    }
    return data_transforms

# Custom loss
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, predicted, target):
        mse_loss = nn.MSELoss()
        mse_loss = mse_loss(predicted, target)
        return mse_loss

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, output1, output2, output3, ranking):
        # Sorting indices for output1
        sorted_indices = torch.argsort(ranking, descending=True)

        # Sort output1 based on sorted_indices (best mos feature is the last feature)
        output1 = torch.index_select(output1, 0, sorted_indices)

        # Initialize list to store losses for different positive sample sizes
        losses = []

        # Iterate over positive sample sizes
        for positive_size in range(1, ranking.shape[0]):
            # Select positive and negative samples
            positive_samples = output1[-positive_size:]
            negative_samples = output1[:-positive_size]

            # Calculate similarity scores
            similarity_pos_1 = torch.cosine_similarity(positive_samples, output2)
            similarity_neg_1 = torch.cosine_similarity(negative_samples, output2)
            similarity_neg_2 = torch.cosine_similarity(positive_samples, output3)
            similarity_pos_2 = torch.cosine_similarity(negative_samples, output3)

            # Calculate contrastive loss
            loss_1 = -torch.log(torch.exp(similarity_pos_1).sum() / (torch.exp(similarity_pos_1).sum() + torch.exp(similarity_neg_1).sum()))
            loss_2 = -torch.log(torch.exp(similarity_pos_2).sum() / (torch.exp(similarity_pos_2).sum() + torch.exp(similarity_neg_2).sum()))

            losses.append(loss_1)
            losses.append(loss_2)

        # Combine losses from different positive sample sizes
        combined_loss = sum(losses)

        return combined_loss

# Arguments
def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--k_fold', type=int,
                        default=1, help='K Fold')  
    parser.add_argument('--epochs', type=int,
                        default=10, help='Number of epochs per fold')
    parser.add_argument('--batch_size', type=int,
                        default=8, help='Batch size')
    parser.add_argument('--lr', type=float,
                        default=0.0000001, help='Learning rate')
    parser.add_argument('--tau', type=float,
                        default=100.0, help='Tau parameter for clip')
    parser.add_argument('--alpha', type=float,
                        default=0.5, help='parameter for weighted loss combination')
    parser.add_argument('--device', type=str,
                        default='cuda', help='Device (cpu/cuda)')
    parser.add_argument('--image_directory', type=str,
                        default='LIVE_Challenge/Images', help='LIVE Challenge dataset images')
    parser.add_argument('--model_directory', type=str,
                        default='trained models/exp-2 (Contrastive Loss)/', help='Folder path for saving the trained models')
    parser.add_argument('--csv_directory_test', type=str,
                        default="csvs/LIVE_C/test/test_", help='Folder path of csvs for testing')
    parser.add_argument('--csv_directory_train', type=str,
                        default="csvs/LIVE_C/train/train_", help='Folder path of csvs for training')
    parser.add_argument('--csv_directory_val', type=str,
                        default="csvs/LIVE_C/val/val_", help='Folder path of csvs for validation')

    optn = parser.parse_args()

    return optn
