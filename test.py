import torch
import torch.nn as nn
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from random import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from configs import transformation, parse_option
from dataloader import IQADataloader
from model import CustomCLIP

# Function for plotting A for good and bad spaces
def plot_confusion_matrix(df_confusion, cmap=plt.cm.gray_r):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.matshow(df_confusion, cmap=cmap) # imshow 
    plt.colorbar(im)

    # Add values to the confusion matrix plot with 3 decimal places
    for i in range(args.random_samples):
        for j in range(args.random_samples):
            plt.text(j, i, f'{df_confusion[i, j]:.3f}', ha="center", va="center", color="black")

    plt.show()
    #plt.savefig("runs/A_g_" + str(z) + ".png")

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

    image_embeddings = None

    #image_embeddings = []
    #text_embeddings = []

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

            # Store image and text embeddings
            if image_embeddings is None:
                image_embeddings = image_embeds.squeeze().cpu().numpy()
            else:
                image_embeddings = np.concatenate((image_embeddings, image_embeds.squeeze().cpu().numpy()), axis=0)
        
        text_embeddings = np.concatenate([text_embeds_pos.squeeze().cpu().numpy(), text_embeds_neg.squeeze().cpu().numpy()], axis=0)

        # Convert embeddings to 2D using PCA and TSNE
        pca = PCA(n_components=2)
        tsne = TSNE(n_components=2)

        # Concatenate image and text embeddings for PCA and t-SNE
        combined_embeddings = np.concatenate((image_embeddings, text_embeddings), axis=0)

        # PCA for combined embeddings
        combined_embeddings_2d_pca = pca.fit_transform(combined_embeddings)

        # TSNE for combined embeddings
        combined_embeddings_2d_tsne = tsne.fit_transform(combined_embeddings)

        # Scatter plot for PCA using text embedding
        plt.scatter(combined_embeddings_2d_pca[:len(image_embeddings), 0], combined_embeddings_2d_pca[:len(image_embeddings), 1], c = lab_per_fold, cmap='Greens', label='Image Embeddings', alpha=0.5, s=100)
        plt.scatter(combined_embeddings_2d_pca[len(image_embeddings):args.random_samples, 0], combined_embeddings_2d_pca[len(image_embeddings):args.random_samples, 1], c = 'blue', label='Text Embedding Positive', alpha=0.5, s=100)
        plt.scatter(combined_embeddings_2d_pca[len(image_embeddings)+args.random_samples:, 0], combined_embeddings_2d_pca[len(image_embeddings)+args.random_samples:, 1], c = 'red', label='Text Embedding Negative', alpha=0.5, s=100)

        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title("Embeddings in 2D Space PCA - K_fold = " + str(z))
        plt.legend()
        plt.savefig("runs/Embeddings_PCA1_" + str(z) + ".png")
        plt.clf()  # Clear the plot for the next iteration

        # Scatter plot for PCA without using text embedding
        plt.scatter(combined_embeddings_2d_pca[:len(image_embeddings), 0], combined_embeddings_2d_pca[:len(image_embeddings), 1], c = lab_per_fold, cmap='Greens', label='Image Embeddings', alpha=0.5, s=100)

        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title("Embeddings in 2D Space PCA - K_fold = " + str(z))
        plt.legend()
        plt.savefig("runs/Embeddings_PCA2_" + str(z) + ".png")
        plt.clf()  # Clear the plot for the next iteration

        # Scatter plot for TSNE using text embeddings 
        plt.scatter(combined_embeddings_2d_tsne[:len(image_embeddings), 0], combined_embeddings_2d_tsne[:len(image_embeddings), 1], c = lab_per_fold, cmap='Greens', label='Image Embeddings', alpha=0.5, s=100)
        plt.scatter(combined_embeddings_2d_tsne[len(image_embeddings):args.random_samples, 0], combined_embeddings_2d_tsne[len(image_embeddings):args.random_samples, 1], c = 'blue', label='Text Embedding Positive', alpha=0.5, s=100)
        plt.scatter(combined_embeddings_2d_tsne[len(image_embeddings)+args.random_samples:, 0], combined_embeddings_2d_tsne[len(image_embeddings)+args.random_samples:, 1], c = 'red', label='Text Embedding Negative', alpha=0.5, s=100)

        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title("Embeddings in 2D Space TSNE - K_fold = " + str(z))
        plt.legend()
        plt.savefig("runs/Embeddings_TSNE1_" + str(z) + ".png")
        plt.clf()  # Clear the plot for the next iteration

        # Scatter plot for TSNE without using text embeddings 
        plt.scatter(combined_embeddings_2d_tsne[:len(image_embeddings), 0], combined_embeddings_2d_tsne[:len(image_embeddings), 1], c = lab_per_fold, cmap='Greens', label='Image Embeddings', alpha=0.5, s=100)

        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.title("Embeddings in 2D Space TSNE - K_fold = " + str(z))
        plt.legend()
        plt.savefig("runs/Embeddings_TSNE2_" + str(z) + ".png")
        plt.clf()  # Clear the plot for the next iteration

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

        A_g = model.text_distribution._A_g.cpu().numpy()
        A_b = model.text_distribution._A_b.cpu().numpy()
        plot_confusion_matrix(A_g.dot(A_g.transpose()))
        plot_confusion_matrix(A_b.dot(A_b.transpose()))

# The median of PLCC and SROCC scores for all folds
print("The median PLCC for all the models:", np.median(plcc_scores))
print("The median SROCC for all the models:", np.median(srocc_scores))
