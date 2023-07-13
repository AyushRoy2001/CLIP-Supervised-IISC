import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from random import random

from configs import transformation, parse_option
from dataloader import IQADataloader
from model import CustomCLIP

# Load the transforms
data_transforms = transformation()

# The arguments
args = parse_option()

# Fix things up to reduce randomization
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

# K PLCC and SROCC to be stored
plcc_scores = []
srocc_scores = []

# K model training
k_fold = args.k_fold
for z in range(k_fold):
    # Creating the test dataset and data loader
    data_dir = args.image_directory
    test_csv_path = args.csv_directory_test + str(z) + ".csv"
    test_dataset = IQADataloader(data_dir, csv_file=test_csv_path, transform=data_transforms['validation'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Testing loop
    save_path = args.model_directory + "trained_model_srocc_" + str(z) + ".pt"
    model = torch.load(save_path)
    model.eval()

    predictions = []  # all labels
    labels = []  # all predictions

    # Iterating over the testing data
    with torch.no_grad():
        pred_per_fold = []  # predictions for each fold
        lab_per_fold = []  # labels for each fold
        for inputs, targets in test_loader:
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            # Forward pass
            outputs, image_embeds, text_embeds_pos, text_embeds_neg = model(inputs)

            # Predictions and ground truth labels
            predictions.extend(outputs.squeeze().cpu().numpy())
            labels.extend(targets.cpu().numpy())
            pred_per_fold.extend(outputs.squeeze().cpu().numpy())
            lab_per_fold.extend(targets.cpu().numpy())

        # Scatter plot for each fold of MOS and predicted scores
        plt.scatter(lab_per_fold, pred_per_fold)
        plt.xlabel("MOS")
        plt.ylabel("Predicted")
        plt.title("K_fold = " + str(z))
        plt.savefig("runs/SP_" + str(z) + ".png")
        plt.clf()  # Clear the plot for the next iteration

        # Calculate PLCC and SROCC for this fold
        plcc = np.corrcoef(pred_per_fold, lab_per_fold)[0, 1]
        srocc = stats.spearmanr(pred_per_fold, lab_per_fold)[0]
        print(f'PLCC for fold {z}: {plcc:.4f}')
        print(f'SROCC for fold {z}: {srocc:.4f}')

        plcc_scores.append(plcc)
        srocc_scores.append(srocc)

# The median of PLCC and SROCC scores for all folds
print("The median PLCC for all the models:", np.median(plcc_scores))
print("The median SROCC for all the models:", np.median(srocc_scores))


    



    
