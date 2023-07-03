import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from model import CustomCLIP
from configs import transformation, parse_option
from dataloader import IQADataloader

# The arguments
args = parse_option()

# Create the folds for training and validation
k_fold = args.k_fold

# Load the transforms and model
model = CustomCLIP().to(args.device)
data_transforms = transformation()

# k model training
for z in range(k_fold):

    data_dir = args.image_directory

    # Create the train dataset and data loader
    train_csv_path = "csvs/LIVE_C/train/train_" + str(z) + ".csv"
    train_dataset = IQADataloader(data_dir, csv_file=train_csv_path, transform=data_transforms['train'])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Create the val dataset and data loader
    val_csv_path = "csvs/LIVE_C/val/val_" + str(z) + ".csv"
    val_dataset = IQADataloader(data_dir, csv_file=val_csv_path, transform=data_transforms['train'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    #learnable_params = [model.text1_token, model.text2_token] # parameters to be trained
    #learnable_params = model.clip_model.visual.parameters()
    learnable_params = [model.dense_layer.weight, model.dense_layer.bias]
    optimizer = optim.Adam(learnable_params, lr=args.lr)  
    scheduler = ReduceLROnPlateau(optimizer, 'min')

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

        # Iterate over the training data
        for inputs, labels in train_loader:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            # Zero the gradients
            optimizer.zero_grad()
        
            # Forward pass
            outputs = model(inputs)
            print(outputs)
            print(labels)

            loss = criterion(outputs, labels)

            # Backward pass and optimization and Clip gradients to a maximum norm
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
            optimizer.step()

            # Update running loss
            running_loss += loss.item() * inputs.size(0)

            # Visualizing various training metrics, losses, and parameters (run tensorboard --logdir=runs after training in terminal)
            writer.add_scalar('training loss', running_loss / 1000, epoch * len(train_loader))
            # for name, param in model.named_parameters():
            #     writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step=epoch)

        # Calculate average training loss
        train_loss = running_loss / len(train_dataset)

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
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)

                # Update validation loss
                val_loss += loss.item() * inputs.size(0)

        # Calculate average validation loss
        val_loss /= len(val_dataset)

        # Adjust the learning rate based on validation loss
        scheduler.step(val_loss)

        # Print epoch statistics
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f}')
