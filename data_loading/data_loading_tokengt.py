import os
import torch
from tqdm.auto import tqdm
from datasets import Dataset

from graphormer_tokengt.tokengt.collating_tokengt import preprocess_item as preprocess_item_tokengt
from data_loading.data_loading import get_dataset_train_val_test


def check_is_node_level_dataset(dataset_name):
    if dataset_name in ["PPI", "Cora", "CiteSeer"]:
        return True
    elif "infected" in dataset_name:
        return True
    elif "hetero" in dataset_name:
        return True
    
    return False


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
            "edge_index": data.edge_index.tolist(),
            "num_nodes": data.num_nodes,
            "y": data.y.tolist() if data.y.ndim > 1 else data.y.unsqueeze(1).tolist(),
            "node_feat": data.x.to(torch.float32).tolist(),
            "dataset_name": [dataset_name],
        }

        if edge_attr_flag:
            data_dict["edge_attr"] = data.edge_attr.to(torch.float32).tolist()

        if max_degree_flag:
            data_dict["max_degree"] = [data.max_degree]

        hf_data.append(data_dict)

    return hf_data


def get_dataset_train_val_test_tokengt(dataset, dataset_dir, **kwargs):
    train_mask, val_mask, test_mask = None, None, None
    if not check_is_node_level_dataset(dataset):
        train, val, test, num_classes, task_type, y_scaler = get_dataset_train_val_test(dataset, dataset_dir, **kwargs)
    else:
        train, val, test, num_classes, task_type, y_scaler, train_mask, val_mask, test_mask = get_dataset_train_val_test(dataset, dataset_dir, **kwargs)

    target_name = kwargs["target_name"]
    model_arch = kwargs["model"]

    train_hf = convert_to_hf_format(train, dataset)
    val_hf = convert_to_hf_format(val, dataset)
    test_hf = convert_to_hf_format(test, dataset)

    train_dataset = Dataset.from_dict({k: [dic[k] for dic in train_hf] for k in train_hf[0]})
    val_dataset = Dataset.from_dict({k: [dic[k] for dic in val_hf] for k in val_hf[0]})
    test_dataset = Dataset.from_dict({k: [dic[k] for dic in test_hf] for k in test_hf[0]})

    del train_hf
    del val_hf
    del test_hf

    preprocess_fn = preprocess_item_tokengt

    def preprocess_batch(batch):
        processed_items = []
        for i in range(len(batch["node_feat"])):
            single_item = {
                "node_feat": batch["node_feat"][i],
                "edge_index": batch["edge_index"][i],
                "num_nodes": batch["num_nodes"][i],
                "y": batch["y"][i],
                "dataset_name": batch["dataset_name"][i],
            }

            if "edge_attr" in batch:
                single_item["edge_attr"] = batch["edge_attr"][i]

            if "max_degree" in batch:
                single_item["max_degree"] = batch["max_degree"][i]

            processed_item = preprocess_fn(single_item)
            processed_items.append(processed_item)

        # Convert list of processed items back into batch format
        batch = {key: [processed_item[key] for processed_item in processed_items] for key in processed_items[0]}

        return batch

    # Use small values for batch_size and num_proc to help fitting in memory
    map_dict = dict(
        batched=True, batch_size=16, num_proc=4, keep_in_memory=False, load_from_cache_file=True, writer_batch_size=64
    )

    train_dataset_processed = train_dataset.map(
        preprocess_batch,
        **map_dict,
        cache_file_name=os.path.join(dataset_dir, f"train_{dataset}_{target_name}_{model_arch}.cache"),
    )
    val_dataset_processed = val_dataset.map(
        preprocess_batch,
        **map_dict,
        cache_file_name=os.path.join(dataset_dir, f"val_{dataset}_{target_name}_{model_arch}.cache"),
    )
    test_dataset_processed = test_dataset.map(
        preprocess_batch,
        **map_dict,
        cache_file_name=os.path.join(dataset_dir, f"test_{dataset}_{target_name}_{model_arch}.cache"),
    )

    return train_dataset_processed, val_dataset_processed, test_dataset_processed, num_classes, task_type, y_scaler, train_mask, val_mask, test_mask