import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from configs import transformation, parse_option
from dataloader import IQADataloader

# Load the transforms
data_transforms = transformation()

criterion = nn.MSELoss()

# The arguments
args = parse_option()

# K losses to be stored
losses = []

# K PLCC and SROCC to be stored
plcc_scores = []
srocc_scores = []

# K model training
k_fold = 5
for z in range(k_fold):
    # Creating the test dataset and data loader
    data_dir = args.image_directory
    test_csv_path = "csvs/test/test_"+str(z)+".csv"
    test_dataset = IQADataloader(data_dir, csv_file=test_csv_path, transform=data_transforms['validation'])
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Testing loop
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    save_path = args.model_directory+"trained_model"+str(z)+".pt"
    model.load_state_dict(torch.load(save_path))
    model = model.to(args.device)
    model.eval()

    # Initializing testing loss
    test_loss = 0.0

    predictions = [] # all labels
    labels = [] # all predictions

    # Iterating over the testing data
    with torch.no_grad():
        pred_per_fold = [] # predictions for each fold
        lab_per_fold = [] # labels for each fold
        for inputs, targets in test_loader:
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.float())

            # Update testing loss
            test_loss += loss.item() * inputs.size(0)

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

    # Average testing loss
    test_loss /= len(test_dataset)
    losses.append(test_loss)

    # Testing loss for this fold
    print(f'Testing Loss for fold {z}: {test_loss:.4f}')
    
# Scatter plot for all the models combined of all folds
plt.scatter(labels, predictions)
plt.xlabel("MOS")
plt.ylabel("Predicted")
plt.title("K_fold = 1,2,3,4,5")
plt.savefig("runs/CombinedSP_.png")

# The median of PLCC and SROCC scores for all folds
median_plcc = np.median(plcc_scores)
median_srocc = np.median(srocc_scores)

# The median scores
print("The median PLCC for all the models:")
print(median_plcc)
print("The median SROCC for all the models:")
print(median_srocc)


    