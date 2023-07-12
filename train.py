from statistics import mean
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from model import CustomCLIP
from configs import transformation, parse_option, MSELoss, ContrastiveLoss
from dataloader import IQADataloader

# The arguments
args = parse_option()

# Create the folds for training and validation
k_fold = args.k_fold

# Load the transforms and model
model = CustomCLIP().to(args.device)
model.float()
data_transforms = transformation()

# k model training
for z in range(k_fold):

    data_dir = args.image_directory

    # Create the train dataset and data loader
    train_csv_path = args.csv_directory_train + str(z) + ".csv"
    train_dataset = IQADataloader(data_dir, csv_file=train_csv_path, transform=data_transforms['train'])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Create the val dataset and data loader
    val_csv_path = args.csv_directory_val + str(z) + ".csv"
    val_dataset = IQADataloader(data_dir, csv_file=val_csv_path, transform=data_transforms['train'])
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    # Define the loss function and optimizer
    mse = MSELoss()
    conloss = ContrastiveLoss()
    learnable_params = model.clip_model.visual.parameters() # parameters to be trained
    optimizer = optim.Adam(learnable_params, lr=args.lr)  
    #scheduler = ReduceLROnPlateau(optimizer, 'min')

    # Set the number of epochs
    num_epochs = args.epochs

    # Set tensorboard for visualization
    viz = 'runs/k_fold_' + str(z)
    writer = SummaryWriter(viz)

    # Training loop
    for epoch in range(num_epochs):
        # Set the model in training mode
        model.train()

        # Initialize running loss
        running_loss = 0.0
        running_mse = 0.0
        running_con = 0.0

        # Iterate over the training data
        for inputs, labels in train_loader:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            # Zero the gradients
            optimizer.zero_grad()
        
            # Forward pass
            outputs, image_embeds, text_embeds_pos, text_embeds_neg = model(inputs)
            
            # Combining the losses
            loss_mse = mse(outputs, labels)
            loss_con = conloss(image_embeds, text_embeds_pos, text_embeds_neg, labels)
            loss = args.alpha*loss_mse+(1-args.alpha)*loss_con

            # Backward pass and optimization and Clip gradients to a maximum norm
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item() * inputs.size(0)
            running_mse += loss_mse.item() * inputs.size(0)
            running_con += loss_con.item() * inputs.size(0)
            

        # Calculate average training loss
        train_loss = running_loss / len(train_dataset)
        train_mse = running_mse / len(train_dataset)
        train_con = running_con / len(train_dataset)

        # Visualizing various training metrics, losses, and parameters (run tensorboard --logdir=runs after training in terminal)
        writer.add_scalar('combined training loss', train_loss, epoch)
        writer.add_scalar('mse training loss', train_mse, epoch)
        writer.add_scalar('contrastive training loss', train_con, epoch)

        # Set the model in evaluation mode
        model.eval()

        # Initialize validation loss
        val_loss = 0.0

        # Iterate over the validation data
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)

                # Forward pass
                outputs, image_embeds, text_embeds_pos, text_embeds_neg = model(inputs)
                loss_mse = mse(outputs, labels)

                # Update validation loss
                val_loss += loss_mse.item() * inputs.size(0)

        # Calculate average validation loss
        val_loss /= len(val_dataset)

        # Visualizing various training metrics, losses, and parameters (run tensorboard --logdir=runs after training in terminal)
        writer.add_scalar('mse validation loss', val_loss, epoch)

        # Adjust the learning rate based on validation loss
        #scheduler.step(val_loss)

        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}')

    # Save the model and constants.pkl file
    save_path = args.model_directory + "trained_model_" + str(z) + ".pt"
    torch.save(model, save_path)
