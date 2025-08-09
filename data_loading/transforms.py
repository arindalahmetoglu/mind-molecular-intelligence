import torch
import torch_geometric.transforms as T
from torch_geometric.utils import degree, to_undirected
from .chemprop_featurisation import (
    atom_features,
    atom_features_int,
    bond_features,
    bond_features_int,
    get_atom_constants,
)
from rdkit import Chem
import numpy as np
from data_loading.posenc import compute_posenc_stats

from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

from typing import Optional


class LocalDegreeProfile(BaseTransform):
    r"""Appends the Local Degree Profile (LDP) from the `"A Simple yet
    Effective Baseline for Non-attribute Graph Classification"
    <https://arxiv.org/abs/1811.03508>`_ paper
    (functional name: :obj:`local_degree_profile`)

    .. math::
        \mathbf{x}_i = \mathbf{x}_i \, \Vert \, (\deg(i), \min(DN(i)),
        \max(DN(i)), \textrm{mean}(DN(i)), \textrm{std}(DN(i)))

    to the node features, where :math:`DN(i) = \{ \deg(j) \mid j \in
    \mathcal{N}(i) \}`.
    """

    def __init__(self):
        from torch_geometric.nn.aggr.fused import FusedAggregation

        self.aggr = FusedAggregation(["min", "max", "mean", "std"])

    def forward(self, data: Data) -> Data:
        if data is not None and data.edge_index is not None:
            row, col = data.edge_index
            N = data.num_nodes

            deg = degree(row, N, dtype=torch.float).view(-1, 1)
            xs = [deg] + self.aggr(deg[col], row, dim_size=N)

            if data.x is not None:
                data.x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
                data.x = torch.cat([data.x] + xs, dim=-1)
            else:
                data.x = torch.cat(xs, dim=-1)

        return data


class TargetIndegree(BaseTransform):
    r"""Saves the globally normalized degree of target nodes
    (functional name: :obj:`target_indegree`)

    .. math::

        \mathbf{u}(i,j) = \frac{\deg(j)}{\max_{v \in \mathcal{V}} \deg(v)}

    in its edge attributes.

    Args:
        cat (bool, optional): Concat pseudo-coordinates to edge attributes
            instead of replacing them. (default: :obj:`True`)
    """

    def __init__(
        self,
        norm: bool = True,
        max_value: Optional[float] = None,
        cat: bool = True,
    ):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def forward(self, data: Data) -> Data:
        if data is not None and data.edge_index is not None:
            col, pseudo = data.edge_index[1], data.edge_attr

            deg = degree(col, data.num_nodes)

            if self.norm:
                deg = deg / (deg.max() if self.max is None else self.max)

            deg = deg[col]
            deg = deg.view(-1, 1)

            if pseudo is not None and self.cat:
                pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
                data.edge_attr = torch.cat([pseudo, deg.type_as(pseudo)], dim=-1)
            else:
                data.edge_attr = deg

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(norm={self.norm}, " f"max_value={self.max})"


def add_chemprop_features(data, one_hot, max_atomic_number):
    atom_constants = get_atom_constants(max_atomic_number)
    mol = Chem.MolFromSmiles(data.smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    # Ensure torch receives a proper NumPy ndarray buffer
    adj = np.asarray(Chem.rdmolops.GetAdjacencyMatrix(mol))
    ei = torch.nonzero(torch.from_numpy(adj)).T
    if one_hot:
        atom_feat = torch.tensor(
            [atom_features(atom, atom_constants) for atom in mol.GetAtoms()],
        )

        bond_feat = torch.tensor(
            [bond_features(mol.GetBondBetweenAtoms(ei[0][i].item(), ei[1][i].item())) for i in range(ei.shape[1])],
        )
    else:
        atom_feat = torch.tensor(
            [atom_features_int(atom, atom_constants) for atom in mol.GetAtoms()],
        )

        bond_feat = torch.tensor(
            [bond_features_int(mol.GetBondBetweenAtoms(ei[0][i].item(), ei[1][i].item())) for i in range(ei.shape[1])],
        )

    # ei, bond_feat = to_undirected(ei, edge_attr=bond_feat)

    data.x = atom_feat
    data.edge_index = ei
    data.edge_attr = bond_feat

    return data



class ChempropFeatures(T.BaseTransform):
    def __init__(self, one_hot, max_atomic_number):
        self.one_hot = one_hot
        self.max_atomic_number = max_atomic_number

    def forward(self, data):
        data = add_chemprop_features(data, self.one_hot, self.max_atomic_number)

        return data


class Add3DOrPosAsNodeFeatures(T.BaseTransform):
    def forward(self, data):
        if hasattr(data, "pos") and data.pos is not None:
            if data.pos.shape[0] != data.x.shape[0]:
                return None
            data.x = torch.cat((data.pos, data.x), dim=-1)
        return data


class SelectTarget(T.BaseTransform):
    def __init__(self, target_id):
        self.target_id = target_id

    def forward(self, data):
        if data.y.ndim == 2:
            data.y = data.y[:, self.target_id].squeeze()
        elif data.y.ndim == 1:
            data.y = data.y[self.target_id].squeeze()
        return data


class OneHotInt(T.BaseTransform):
    def __init__(self, max_degree: int, in_degree: bool = False, cat: bool = True):
        self.max_degree = max_degree
        self.in_degree = in_degree
        self.cat = cat

    def forward(self, data):
        idx, x = data.edge_index[1 if self.in_degree else 0], data.x
        deg = degree(idx, data.num_nodes, dtype=torch.long)

        if x is not None and self.cat:
            x = x.view(-1, 1) if x.dim() == 1 else x
            if x.dim() > 1:
                deg = deg.unsqueeze(1)
            data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
        else:
            data.x = deg.reshape(-1, 1)

        return data


class FeaturesToInt(T.BaseTransform):
    def forward(self, data):
        data.x = data.x.to(torch.int32)
        data.edge_attr = data.edge_attr.to(torch.int32)

        return data


class AddNumNodes(T.BaseTransform):
    def forward(self, data):
        if data is not None:
            data.num_nodes = data.x.shape[0]
        return data


class AddMaxEdge(T.BaseTransform):
    def forward(self, data):
        if data is not None:
            if data.edge_index.numel() > 0:
                data.max_edge = torch.tensor(data.edge_index.shape[-1]).unsqueeze(0)
            else:
                return None

        return data


class AddMaxNode(T.BaseTransform):
    def forward(self, data):
        if data is not None:
            data.max_node = torch.tensor(data.num_nodes).unsqueeze(0)

        return data
    

class AddMaxEdgeGlobal(T.BaseTransform):
    def __init__(self, max_edge: int):
        self.max_edge = max_edge

    def forward(self, data):
        data.max_edge_global = self.max_edge

        return data


class AddMaxNodeGlobal(T.BaseTransform):
    def __init__(self, max_node: int):
        self.max_node = max_node

    def forward(self, data):
        data.max_node_global = self.max_node

        return data


class AddMaxDegree(T.BaseTransform):
    def __init__(self, max_degree: int):
        self.max_degree = max_degree

    def forward(self, data):
        if data is not None:
            data.max_degree = self.max_degree

        return data


class FormatSingleLabel(T.BaseTransform):
    def forward(self, data):
        if data is None:
            return data

        if data.y.ndim == 0:
            data.y = data.y.unsqueeze(0)
        elif data.y.ndim == 2:
            data.y = data.y.squeeze(1)

        return data


class LabelNanToZero(T.BaseTransform):
    def forward(self, data):
        if data is None:
            return data

        data.y = torch.nan_to_num(data.y, nan=0.0)

        return data


class EdgeFeaturesUnsqueeze(T.BaseTransform):
    def forward(self, data):
        if data is None:
            return data

        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            if data.edge_attr.ndim == 1:
                data.edge_attr = data.edge_attr.unsqueeze(1)

        return data


class OneHotYToSingle(T.BaseTransform):
    def forward(self, data):
        data.y = torch.argmax(data.y)

        return data
    
    
class SubtractOneY(T.BaseTransform):
    def forward(self, data):
        if data is not None:
            data.y = data.y - 1

        return data
    
class AddPosEnc(T.BaseTransform):
    def __init__(self, pe_types):
        self.pe_types = pe_types

    def forward(self, data):
        return compute_posenc_stats(data, pe_types=self.pe_types, is_undirected=True)
    

class TargetToY(T.BaseTransform):
    def __init__(self, target_name):
        self.target_name = target_name

    def forward(self, data):
        data.y = getattr(data, f"y_{self.target_name}")
        return data
    
    
class AddMasks(T.BaseTransform):
    def __init__(self, train_mask, val_mask, test_mask):
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

    def forward(self, data):
        data.train_mask = self.train_mask
        data.val_mask = self.val_mask
        data.test_mask = self.test_mask

        return data