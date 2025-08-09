from torch_geometric.data.collate import collate


def collate_data_list(data_list):
    if len(data_list) == 1:
        return data_list[0], None

    print("Collating...")
    data, slices, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=False,
        add_batch=False,
    )
    print("Collating finished!")

    return data, slices


def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = [datasets[0][i] for i in range(n1)] + \
                [datasets[1][i] for i in range(n2)] + \
                [datasets[2][i] for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = collate_data_list(data_list)

    split_idxs = [
        list(range(n1)),
        list(range(n1, n1 + n2)),
        list(range(n1 + n2, n1 + n2 + n3))
    ]
    
    datasets[0].split_idxs = split_idxs

    return datasets[0]
