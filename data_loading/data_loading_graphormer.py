import os
import torch
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from safetensors.torch import save_file, load_file
from transformers.utils.import_utils import is_cython_available

from data_loading.data_loading import get_dataset_train_val_test
from graphormer_tokengt.graphormer.collating_graphormer_safetensors import preprocess_item as preprocess_item_graphormer


if is_cython_available():
    import pyximport

    pyximport.install(setup_args={"include_dirs": np.get_include()})
    from graphormer_tokengt.graphormer import algos_graphormer  # noqa E402


def check_is_node_level_dataset(dataset_name):
    if dataset_name in ["PPI", "Cora", "CiteSeer"]:
        return True
    elif "infected" in dataset_name:
        return True
    elif "hetero" in dataset_name:
        return True
    
    return False


def run_floyd_warshall(dataset, num_nodes, split):
    print(f"Initial run of Floyd-Warshall on {split}...")

    distances = []
    for item in tqdm(dataset):
        edge_index = item["edge_index"].numpy()
        adj = np.zeros([item["num_nodes"], item["num_nodes"]], dtype=bool)
        adj[edge_index[0], edge_index[1]] = True

        shortest_path_result, _ = algos_graphormer.floyd_warshall(adj)
        max_dist = np.amax(shortest_path_result)
        distances.append(max_dist)

    distances_no_default = [d for d in distances if d != 510]
    if len(distances_no_default) > 0:
        max_dist = max(distances_no_default)
    else:
        max_dist = 0

    return max_dist


def convert_to_hf_format(pyg_data_list, dataset_name):
    hf_data = []
    print("Converting data from PyG to huggingface format...")
    for data in tqdm(pyg_data_list):
        if data is None:
            continue

        edge_attr_flag = data.edge_attr is not None
        if hasattr(data, "max_degree"):
            max_degree_flag = data.max_degree is not None
        else:
            max_degree_flag = False

        data_dict = {
            "edge_index": data.edge_index,
            "num_nodes": torch.tensor(data.num_nodes),
            "y": data.y if data.y.ndim > 1 else data.y.unsqueeze(1),
            "node_feat": data.x.float(),
            "dataset_name": [dataset_name],
        }

        if edge_attr_flag:
            data_dict["edge_attr"] = data.edge_attr.float()

        if max_degree_flag:
            data_dict["max_degree"] = torch.tensor([data.max_degree])

        hf_data.append(data_dict)

    return hf_data


def get_max_dist(ds_processed):
    return max([item["max_dist"] for item in ds_processed])

def edit_max_dist_dataset(ds_processed, max_d):
    for data in ds_processed:
        data["max_dist"] = max_d


def dataset_to_safetensors(ds, preprocess_fn, safetensors_dir, split, max_nodes, overall_max_dist, is_node_task=False):
    dataset_processed = []

    all_files = os.listdir(safetensors_dir)
    all_files = set([os.path.join(safetensors_dir, fpath) for fpath in all_files])

    print(f"Pre-processing {split} files for Graphormer...")
    for i in tqdm(range(len(ds))):
        file_out_path = os.path.join(safetensors_dir, f"{split}_safetensors_file_{i}_of_{len(ds)}.sft")

        if file_out_path not in all_files:
            prep_item = preprocess_fn(ds[i], max_nodes, overall_max_dist, is_node_task=is_node_task)
            
            save_file(prep_item, file_out_path)
        else:
            prep_item = load_file(file_out_path)

        dataset_processed.append(prep_item)

    return dataset_processed


def get_dataset_train_val_test_graphormer(dataset, dataset_dir, **kwargs):
    train_mask, val_mask, test_mask = None, None, None
    if not check_is_node_level_dataset(dataset):
        train, val, test, num_classes, task_type, y_scaler = get_dataset_train_val_test(dataset, dataset_dir, **kwargs)
        is_node_task = False
    else:
        train, val, test, num_classes, task_type, y_scaler, train_mask, val_mask, test_mask = get_dataset_train_val_test(dataset, dataset_dir, **kwargs)
        is_node_task = True

    model_arch = "graphormer"
    target_name = kwargs["target_name"]

    safetensors_dir = os.path.join(dataset_dir, f"{dataset}_{target_name}_safetensors_cache")
    Path(safetensors_dir).mkdir(exist_ok=True, parents=True)

    max_nodes = train[0].max_node_global

    train_hf = convert_to_hf_format(train, dataset)
    val_hf = convert_to_hf_format(val, dataset)
    test_hf = convert_to_hf_format(test, dataset)

    if model_arch == "graphormer":
        train_result_path = os.path.join(dataset_dir, f"train_{dataset}_{target_name}_Floyd_Warshall_max_dist.npy")
        if os.path.isfile(train_result_path):
            train_max_dist = np.load(train_result_path)
        else:
            train_max_dist = run_floyd_warshall(train_hf, max_nodes, "train")
            np.save(train_result_path, train_max_dist)

        val_result_path = os.path.join(dataset_dir, f"val_{dataset}_{target_name}_Floyd_Warshall_max_dist.npy")
        test_result_path = os.path.join(dataset_dir, f"test_{dataset}_{target_name}_Floyd_Warshall_max_dist.npy")

        if os.path.isfile(val_result_path):
            val_max_dist = np.load(val_result_path)
        else:
            val_max_dist = run_floyd_warshall(val_hf, max_nodes, "val")
            np.save(val_result_path, val_max_dist)

        if os.path.isfile(test_result_path):
            test_max_dist = np.load(test_result_path)
        else:
            test_max_dist = run_floyd_warshall(test_hf, max_nodes, "test")
            np.save(test_result_path, test_max_dist)

        overall_max_dist = max(train_max_dist, val_max_dist, test_max_dist)
    else:
        overall_max_dist = None

    print("Max distance found = ", overall_max_dist)

    preprocess_fn = preprocess_item_graphormer
    train_dataset_proc = dataset_to_safetensors(train_hf, preprocess_fn, safetensors_dir, "train", max_nodes, overall_max_dist, is_node_task=is_node_task)
    val_dataset_proc = dataset_to_safetensors(val_hf, preprocess_fn, safetensors_dir, "val", max_nodes, overall_max_dist, is_node_task=is_node_task)
    test_dataset_proc = dataset_to_safetensors(test_hf, preprocess_fn, safetensors_dir, "test", max_nodes, overall_max_dist, is_node_task=is_node_task)

    return train_dataset_proc, val_dataset_proc, test_dataset_proc, num_classes, task_type, y_scaler, train_mask, val_mask, test_mask