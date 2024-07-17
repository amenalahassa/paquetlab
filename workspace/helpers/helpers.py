from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import torch
import torch.nn as nn
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_local_model(model):
    model.load_state_dict(torch.load(f'models/best_{model.name}_model.pth'))

def get_indices(size, train_ratio = 0.8, test_ratio = 0.1, valid_ratio = 0.1):
    # Split the indices of the dataset
    indices = list(range(size))
    train_indices, temp_indices = train_test_split(indices, test_size=(1 - train_ratio), shuffle=True, random_state=42)
    if valid_ratio > 0:
        test_indices, valid_indices = train_test_split(temp_indices, test_size=test_ratio/(test_ratio + valid_ratio), shuffle=True, random_state=42)
    else: 
        valid_indices = []
        test_indices = temp_indices
    return train_indices, test_indices, valid_indices

def load_face_data(data_dir, img_dir, extract_name=True):
    df = pd.read_csv(data_dir, index_col=False)

    labels = df.label.unique()
    label2id = {l:i for i, l in enumerate(labels)}
    id2label = {i:l for i, l in enumerate(labels)}
    df['target'] = df.apply(lambda row: label2id[row['label']], axis=1)
    
    directory_path = img_dir
    files_list = os.listdir(directory_path)
    id_pattern = r'_(\d{4})_'
    
    files = {}
    files_calf = {}
    for f in files_list:
        if f.lower().endswith('.jpg') and os.path.isfile(os.path.join(directory_path, f)):
            if extract_name:
                img_name = f.replace("#", "-")
                img_name = img_name.replace("=", "-")
                img_name = img_name.replace(".", "-")
                img_name = list(img_name)
                img_name[-4] = "_"
                img_name = ''.join(img_name)
            else:
                img_name = f
                
            files[img_name] = f

            match = re.search(id_pattern, img_name)
            if match:
                calf_id = match.group(1)
            else:
                calf_id = None

            files_calf[img_name] = calf_id

    left_df = pd.DataFrame({'image': files.keys(), 'path': files.values(), 'calf': files_calf.values()})
    # print(df.shape, left_df.shape)
    # print(df.head(), left_df.head())
    
    df = pd.merge(df, left_df, on="image", how='inner')
    
    # print(df.shape, left_df.shape)
    
    return df, labels, label2id, id2label


# Function to sample uniformly with specified values
def uniform_sample_with_values(df, sample_size, groupby_cols, filter_values = None, use_remaining = False):
    copy = df.copy(deep=True)
    # Filter dataframe based on specified values for each column
    if filter_values is not None:
        for col, values in filter_values.items():
            copy = copy[copy[col].isin(values)]
    
    # Calculate the number of rows to sample per group
    num_groups = copy.groupby(groupby_cols).ngroups
    sample_per_group = max(sample_size // num_groups, 1)
    
    # Sample uniformly within each group
    sampled_df = copy.groupby(groupby_cols, group_keys=False).apply(lambda x: x.sample(min(len(x), sample_per_group)))
    
    # If we sampled less than the desired sample_size due to group sizes, sample more from the remaining dataframe
    if len(sampled_df) < sample_size and use_remaining:
        remaining_sample_size = sample_size - len(sampled_df)
        remaining_df = copy[~copy.index.isin(sampled_df.index)]
        additional_samples = remaining_df.sample(min(len(remaining_df), remaining_sample_size))
        sampled_df = pd.concat([sampled_df, additional_samples])
    
    return sampled_df, copy[~copy.index.isin(sampled_df.index)]
