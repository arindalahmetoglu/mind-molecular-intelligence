import torch
import lmdb
import pickle
from pathlib import Path
from typing import Sequence, Union
from functools import lru_cache
from torch import Tensor
from sklearn.preprocessing import StandardScaler
import numpy as np

from fairseq.data import (
    FairseqDataset,
    NestedDictionaryDataset,
)


class LMDBDataset:
    def __init__(self, db_path):
        super().__init__()
        assert Path(db_path).exists(), f"{db_path}: No such file or directory"
        self.env = lmdb.Environment(
            db_path,
            map_size=(1024 ** 3) * 256,
            subdir=False,
            readonly=True,
            readahead=True,
            meminit=False,
            lock=False
        )
        self.len: int = self.env.stat()["entries"]

    def __len__(self):
        return self.len

    @lru_cache(maxsize=16)
    def __getitem__(self, idx: int) -> dict[str, Union[Tensor, float]]:
        if idx < 0 or idx >= self.len:
            raise IndexError
        data = pickle.loads(self.env.begin().get(f"{idx}".encode()))
        data = data.__dict__
        return dict(
            pos=torch.as_tensor(data["pos"]).float(),
            pos_relaxed=torch.as_tensor(data["pos_relaxed"]).float(),
            cell=torch.as_tensor(data["cell"]).float().view(3, 3),
            atoms=torch.as_tensor(data["atomic_numbers"]).long(),
            tags=torch.as_tensor(data["tags"]).long(),
            relaxed_energy=data["y_relaxed"],  # python float
            edge_index=data["edge_index"]
        )
    
class AtomDataset(FairseqDataset):
    def __init__(self, dataset, keyword):
        super().__init__()
        self.dataset = dataset
        self.keyword = keyword
        self.atom_list = [
            1,
            5,
            6,
            7,
            8,
            11,
            13,
            14,
            15,
            16,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            55,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
        ]
        # fill others as unk
        unk_idx = len(self.atom_list) + 1
        self.atom_mapper = torch.full((128,), unk_idx)
        for idx, atom in enumerate(self.atom_list):
            self.atom_mapper[atom] = idx + 1  # reserve 0 for paddin

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        atoms: Tensor = self.dataset[index][self.keyword]
        return self.atom_mapper[atoms]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return pad_1d(samples)


class KeywordDataset(FairseqDataset):
    def __init__(self, dataset, keyword, is_scalar=False, pad_fill=0, scaler=None):
        super().__init__()
        self.dataset = dataset
        self.keyword = keyword
        self.is_scalar = is_scalar
        self.pad_fill = pad_fill
        self.scaler = scaler

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index][self.keyword]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if self.is_scalar:
            return torch.tensor(self.scaler.transform(np.array(samples).reshape(-1, 1)).flatten())
        return pad_1d(samples, fill=self.pad_fill)
    

def pad_1d(samples: Sequence[Tensor], fill=0, multiplier=8):
    max_len = max(x.size(0) for x in samples)
    max_len = (max_len + multiplier - 1) // multiplier * multiplier
    n_samples = len(samples)
    out = torch.full(
        (n_samples, max_len, *samples[0].shape[1:]), fill, dtype=samples[0].dtype
    )
    for i in range(n_samples):
        x_len = samples[i].size(0)
        out[i][:x_len] = samples[i]
    return out


class PBCDataset:
    def __init__(self, dataset: LMDBDataset):
        self.dataset = dataset
        self.cell_offsets = torch.tensor(
            [
                [-1, -1, 0],
                [-1, 0, 0],
                [-1, 1, 0],
                [0, -1, 0],
                [0, 1, 0],
                [1, -1, 0],
                [1, 0, 0],
                [1, 1, 0],
            ],
        ).float()
        self.n_cells = self.cell_offsets.size(0)
        self.cutoff = 8
        self.filter_by_tag = True

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.dataset[idx]

        pos = data["pos"]
        pos_relaxed = data["pos_relaxed"]
        cell = data["cell"]
        atoms = data["atoms"]
        tags = data["tags"]

        offsets = torch.matmul(self.cell_offsets, cell).view(self.n_cells, 1, 3)
        expand_pos = (pos.unsqueeze(0).expand(self.n_cells, -1, -1) + offsets).view(
            -1, 3
        )
        expand_pos_relaxed = (
            pos.unsqueeze(0).expand(self.n_cells, -1, -1) + offsets
        ).view(-1, 3)
        src_pos = pos[tags > 1] if self.filter_by_tag else pos

        dist: Tensor = (src_pos.unsqueeze(1) - expand_pos.unsqueeze(0)).norm(dim=-1)
        used_mask = (dist < self.cutoff).any(dim=0) & tags.ne(2).repeat(
            self.n_cells
        )  # not copy ads
        used_expand_pos = expand_pos[used_mask]
        used_expand_pos_relaxed = expand_pos_relaxed[used_mask]

        used_expand_tags = tags.repeat(self.n_cells)[
            used_mask
        ]  # original implementation use zeros, need to test
        return dict(
            pos=torch.cat([pos, used_expand_pos], dim=0),
            atoms=torch.cat([atoms, atoms.repeat(self.n_cells)[used_mask]]),
            tags=torch.cat([tags, used_expand_tags]),
            real_mask=torch.cat(
                [
                    torch.ones_like(tags, dtype=torch.bool),
                    torch.zeros_like(used_expand_tags, dtype=torch.bool),
                ]
            ),
            deltapos=torch.cat(
                [pos_relaxed - pos, used_expand_pos_relaxed - used_expand_pos], dim=0
            ),
            relaxed_energy=data["relaxed_energy"],
            edge_index=data["edge_index"]
        )


class EdgeIndexDataset(FairseqDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index]["edge_index"]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        concatenated_edge_index = torch.cat(samples, dim=1)

        # Calculate the offsets for each graph
        offsets = torch.tensor([0] + [edge_index.size(1) for edge_index in samples[:-1]])
        offsets = offsets.cumsum(dim=0)

        for i in range(len(offsets)):
            samples[i] += offsets[i]

        concatenated_edge_index = torch.cat(samples, dim=1)

        return concatenated_edge_index
    

class BatchMappingDataset(FairseqDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index]["edge_index"]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        concatenated_edge_index = torch.cat(samples, dim=1)

        # Calculate the offsets for each graph
        offsets = torch.tensor([0] + [edge_index.size(1) for edge_index in samples[:-1]])
        offsets = offsets.cumsum(dim=0)

        zeros = torch.zeros(concatenated_edge_index.shape[-1], dtype=torch.long)
        zeros[offsets[1:]] = 1

        return torch.cumsum(zeros, 0)
    

def load_ocp(ldmb_datapath, is_train, scaler=None):
    lmdb_dataset = LMDBDataset(ldmb_datapath)
    pbc_dataset = PBCDataset(lmdb_dataset)
    atoms = AtomDataset(pbc_dataset, "atoms")
    tags = KeywordDataset(pbc_dataset, "tags")
    real_mask = KeywordDataset(pbc_dataset, "real_mask")
    edge_index = EdgeIndexDataset(pbc_dataset)
    bs_ds = BatchMappingDataset(pbc_dataset)

    pos = KeywordDataset(pbc_dataset, "pos")

    relaxed_energy = KeywordDataset(pbc_dataset, "relaxed_energy", is_scalar=True)

    if is_train:
        y_train = np.array([relaxed_energy[i] for i in range(len(relaxed_energy))]).reshape(-1, 1)

        scaler = StandardScaler()
        scaler = scaler.fit(y_train)

    relaxed_energy = KeywordDataset(pbc_dataset, "relaxed_energy", is_scalar=True, scaler=scaler)
    deltapos = KeywordDataset(pbc_dataset, "deltapos")

    dataset = NestedDictionaryDataset(
        {
            "net_input": {
                "pos": pos,
                "atoms": atoms,
                "tags": tags,
                "real_mask": real_mask,
                "edge_index": edge_index,
                "batch_mapping": bs_ds
            },
            "targets": {
                "relaxed_energy": relaxed_energy,
                "deltapos": deltapos,
            },
        },
        sizes=[np.zeros(len(atoms))],
    )

    return dataset, scaler