import torch
import os
import scipy
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# The splitting functions to save the splitted data in csv for testing
def split_test(path_image,path_labels,k_fold):
    imgpath = scipy.io.loadmat(path_image)
    imgpath = imgpath['AllImages_release']
    imgpath = imgpath[7:1169]
    labels = scipy.io.loadmat(path_labels)
    labels = labels['AllMOS_release'].astype(np.float32)
    labels = labels[0][7:1169]
    limit = int(len(labels)/k_fold)
    for iter in range(k_fold):
        Y = []
        X = []
        for i in range(limit):
            j = random.randint(0,len(labels)-1)
            X.append([imgpath[j][0][:]])
            Y.append([labels[j]])
            imgpath = np.delete(imgpath,j)
            labels = np.delete(labels,j)
        out = np.hstack((X, Y))
        df_test = pd.DataFrame(np.asarray(out), columns = ['image_id','MOS'])
        df_test = df_test.astype({'image_id': 'str','MOS': 'float32'})
        filename = "test_"+str(iter)+".csv"
        df_test.to_csv(filename,index=False)
    return 

# The splitting functions to save the splitted data in csv for training and validation
def split_train(path_image, path_labels, iter):
    imgpath = scipy.io.loadmat(path_image)
    imgpath = imgpath['AllImages_release']
    imgpath = imgpath[7:1169]
    labels = scipy.io.loadmat(path_labels)
    labels = labels['AllMOS_release'].astype(np.float32)
    labels = labels[0][7:1169]
    images_test = []
    reader = "test_" + str(iter) + ".csv"
    df = pd.read_csv(reader)
    images_test.extend(df["image_id"])
    X = []
    Y = []
    for j in range(len(imgpath)):
        if imgpath[j][0][0] not in images_test:
            X.append(imgpath[j][0][:])
            Y.append([labels[j]])

    out = np.hstack((X, Y))
    df_train = pd.DataFrame(np.asarray(out), columns=['image_id', 'MOS'])
    df_train = df_train.astype({'image_id': 'str', 'MOS': 'float32'})
    df_train.drop_duplicates(subset=['image_id'], inplace=True)

    # Randomly split into train and validation
    train_ratio = 0.7
    train_df, val_df = train_test_split(df_train, train_size=train_ratio, random_state=42)

    # Save train and validation CSV files
    train_saver = "train_" + str(iter) + ".csv"
    val_saver = "val_" + str(iter) + ".csv"
    train_df.to_csv(train_saver, index=False)
    val_df.to_csv(val_saver, index=False)
    return df_train

# The main dataloader
class IQADataloader(Dataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = data_dir
        self.csv_file = csv_file
        self.transform = transform
        self.df = pd.read_csv(csv_file)
        self.image_ids = self.df['image_id'].values
        self.labels = self.df['MOS'].values

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_ids[idx])
        image = Image.open(img_path).convert('RGB')
        #image = Image.open(img_path)

        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return image, label


#split_test("D:/IISC/Project/LIVE_Challenge/Data/AllImages_release.mat","D:/IISC/Project/LIVE_Challenge/Data/AllMOS_release.mat",k_fold)
# for i in range(k_fold):
#     split_train("/LIVE_Challenge/Data/AllImages_release.mat","/LIVE_Challenge/Data/AllMOS_release.mat",i)
