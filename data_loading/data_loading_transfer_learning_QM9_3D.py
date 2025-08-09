import os
import torch
from tqdm.auto import tqdm

from data_loading.transforms import *
from data_loading.graphgps_utils import join_dataset_splits
from data_loading.data_loading import apply_scaler, scale_y_for_regression_task, CustomPyGDataset


def load_qm9_gw_hq_chemprop_3D_PyG(dataset_dir, target_name):
    train = torch.load(os.path.join(dataset_dir, "train_with_edge_index.pt"))
    val = torch.load(os.path.join(dataset_dir, "val_with_edge_index.pt"))
    test = torch.load(os.path.join(dataset_dir, "test_with_edge_index.pt"))
   
    print("Selecting target...")
    target_transform = TargetToY(target_name)

    train = [target_transform(data) for data in tqdm(train)]
    val = [target_transform(data) for data in tqdm(val)]
    test = [target_transform(data) for data in tqdm(test)]

    print("\nDataset items look like: ", train[0])

    global_transforms = T.Compose([AddMaxEdgeGlobal(94), AddMaxNodeGlobal(30)]) # cut_off = 2

    print("Applying global node/edge count transforms...")
    train = [global_transforms(data) for data in tqdm(train)]
    val = [global_transforms(data) for data in tqdm(val)]
    test = [global_transforms(data) for data in tqdm(test)]

    y_scaler = scale_y_for_regression_task(train)

    print("Applying label scaler for data splits...")
    train = [apply_scaler(data, scaler=y_scaler, convert_to_numpy=False) for data in tqdm(train)]
    val = [apply_scaler(data, scaler=y_scaler, convert_to_numpy=False) for data in tqdm(val)]
    test = [apply_scaler(data, scaler=y_scaler, convert_to_numpy=False) for data in tqdm(test)]

    print("\nDataset y look like: ", train[0].y)

    print("Finished loading data!")

    train = CustomPyGDataset(train)
    val = CustomPyGDataset(val)
    test = CustomPyGDataset(test)

    num_classes = 1
    task_type = "regression"

    return train, val, test, num_classes, task_type, y_scaler


def load_qm9_dft_lq_chemprop_3D_PyG(dataset_dir, target_name, ind_or_trans):
    if ind_or_trans == "inductive":
        train = torch.load(os.path.join(dataset_dir, "inductive_full_with_edge_index.pt"))
    else:
        train = torch.load(os.path.join(dataset_dir, "transductive_full_with_edge_index.pt"))

    print("Selecting target...")
    target_transform = TargetToY(target_name)

    train = [target_transform(data) for data in tqdm(train)]

    print("\nDataset items look like: ", train[0])

    global_transforms = T.Compose([AddMaxEdgeGlobal(94), AddMaxNodeGlobal(30)]) # cut_off = 2

    print("Applying global node/edge count transforms...")
    train = [global_transforms(data) for data in tqdm(train)]

    y_scaler = scale_y_for_regression_task(train)

    print("Applying label scaler for data splits...")
    train = [apply_scaler(data, scaler=y_scaler, convert_to_numpy=False) for data in tqdm(train)]

    print("Finished loading data!")

    train = CustomPyGDataset(train)

    num_classes = 1
    task_type = "regression"

    return train, None, None, num_classes, task_type, y_scaler


def get_dataset_train_val_test_with_indices_for_graphgps(dataser_dir, target_name, hq_or_lq, ind_or_trans=None):        
    if hq_or_lq == "hq":
        train, val, test, num_classes, task_type, scaler = load_qm9_gw_hq_chemprop_3D_PyG(dataser_dir, target_name)
        joined = join_dataset_splits((train, val, test))
        
    elif hq_or_lq == "lq":
        train, val, test, num_classes, task_type, scaler = load_qm9_dft_lq_chemprop_3D_PyG(dataser_dir, target_name, ind_or_trans=ind_or_trans)
        joined = join_dataset_splits((train, train, train))

    return joined, scaler