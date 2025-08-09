import os
import torch
import torch_geometric
from tqdm.auto import tqdm
from datasets import Dataset

from transfer_learning.tokengt_3d.collating_tokengt import preprocess_item as preprocess_item_tokengt
from data_loading.data_loading_transfer_learning_QM9_3D import load_qm9_gw_hq_chemprop_3D_PyG, load_qm9_dft_lq_chemprop_3D_PyG


def convert_to_hf_format(pyg_data_list):
    hf_data = []
    print("Converting data from PyG to huggingface format...")
    for data in tqdm(pyg_data_list):
        if data is None:
            continue

        num_nodes = 0
        if not hasattr(data, "num_nodes"):
            num_nodes = data.z.flatten().shape[0]
        else:
            num_nodes = data.num_nodes

        y = data.y
        y = y.tolist() if y.ndim > 1 else y.unsqueeze(1).tolist()

        edge_index = data.edge_index
        edge_index = edge_index[[1, 0], :]
        edge_index = torch_geometric.utils.sort_edge_index(edge_index)

        data_dict = {
            "edge_index": edge_index.tolist(),
            "num_nodes": num_nodes,
            "num_edges": edge_index.shape[-1],
            "y": y,
            "pos": data.pos.to(torch.float32).tolist(),
            "z": data.z.to(torch.float32).tolist(),
        }

        hf_data.append(data_dict)

    return hf_data


def get_dataset_train_val_test_tokengt_3d(dataset_dir, target_name, ind_or_trans=None):
    hq_or_lq = None
    if "gw" in target_name:
        train, val, test, num_classes, task_type, y_scaler = load_qm9_gw_hq_chemprop_3D_PyG(dataset_dir, target_name)
        hq_or_lq = "hq"
    else:
        train, _, _, num_classes, task_type, y_scaler = load_qm9_dft_lq_chemprop_3D_PyG(dataset_dir, target_name, ind_or_trans)
        hq_or_lq = "lq"

    preprocess_fn = preprocess_item_tokengt
    map_dict = dict(
        batched=True, batch_size=16, num_proc=4, keep_in_memory=True, load_from_cache_file=False, writer_batch_size=64
    )

    def preprocess_batch(batch):
        processed_items = []
        for i in range(len(batch["z"])):
            single_item = {
                "edge_index": batch["edge_index"][i],
                "num_nodes": batch["num_nodes"][i],
                "num_edges": batch["num_edges"][i],
                "y": batch["y"][i],
                "z": batch["z"][i],
                "pos": batch["pos"][i],
            }
            processed_item = preprocess_fn(single_item)
            processed_items.append(processed_item)

        # Convert list of processed items back into batch format
        keys_to_include = ["edge_index", "num_nodes", "num_edges", "y", "z", "pos", "in_degree", "out_degree", "lap_eigvec", "lap_eigval", "labels"]
        batch = {key: [processed_item[key] for processed_item in processed_items] for key in keys_to_include}

        return batch

    if hq_or_lq == "hq":
        train_hf = convert_to_hf_format(train)
        val_hf = convert_to_hf_format(val)
        test_hf = convert_to_hf_format(test)

        train_dataset = Dataset.from_dict({k: [dic[k] for dic in train_hf] for k in train_hf[0]})
        val_dataset = Dataset.from_dict({k: [dic[k] for dic in val_hf] for k in val_hf[0]})
        test_dataset = Dataset.from_dict({k: [dic[k] for dic in test_hf] for k in test_hf[0]})

        train_dataset_processed = train_dataset.map(preprocess_batch, **map_dict,)
        val_dataset_processed = val_dataset.map(preprocess_batch, **map_dict,)
        test_dataset_processed = test_dataset.map(preprocess_batch, **map_dict,)

        return train_dataset_processed, val_dataset_processed, test_dataset_processed, num_classes, task_type, y_scaler

    elif hq_or_lq == "lq":
        train_hf = convert_to_hf_format(train)
        train_dataset = Dataset.from_dict({k: [dic[k] for dic in train_hf] for k in train_hf[0]})

        train_dataset_processed = train_dataset.map(preprocess_batch, **map_dict,)

        return train_dataset_processed, None, None, num_classes, task_type, y_scaler