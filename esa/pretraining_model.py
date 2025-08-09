import torch
import math
import random
import numpy as np
import pytorch_lightning as pl
import bitsandbytes as bnb
import torch_geometric

from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import radius_graph
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from utils.norm_layers import BN, LN
from esa.masked_layers import ESA
from esa.mlp_utils import SmallMLP, GatedMLPMulti
from data_loading.gaussian import GaussianLayer

from utils.reporting import (
    get_cls_metrics_binary_pt,
    get_cls_metrics_multilabel_pt,
    get_cls_metrics_multiclass_pt,
    get_regr_metrics_pt,
)

from utils.posenc_encoders.laplace_pos_encoder import LapPENodeEncoder
from utils.posenc_encoders.kernel_pos_encoder import KernelPENodeEncoder


@dataclass
class PretrainingConfig:
    """Configuration class for pretraining tasks"""
    # General model config
    num_features: int = 128
    graph_dim: int = 256
    edge_dim: int = 64
    batch_size: int = 32
    lr: float = 0.001
    monitor_loss_name: str = "val_loss"
    xformers_or_torch_attn: str = "xformers"
    hidden_dims: List[int] = None
    num_heads: List[int] = None
    num_sabs: int = 4
    sab_dropout: float = 0.0
    mab_dropout: float = 0.0
    pma_dropout: float = 0.0
    apply_attention_on: str = "edge"
    layer_types: List[str] = None
    use_mlps: bool = True
    set_max_items: int = 0
    early_stopping_patience: int = 30
    optimiser_weight_decay: float = 1e-3
    num_workers: int = 4
    mlp_hidden_size: int = 64
    mlp_type: str = "standard"
    attn_residual_dropout: float = 0.0
    norm_type: str = "LN"
    triu_attn_mask: bool = False
    output_save_dir: str = None
    use_bfloat16: bool = True
    is_node_task: bool = False
    posenc: str = None
    num_mlp_layers: int = 3
    pre_or_post: str = "pre"
    pma_residual_dropout: float = 0
    use_mlp_ln: bool = False
    mlp_dropout: float = 0
    
    # Run-level / project config
    seed: int = 42
    dataset: str = "QM9"
    dataset_download_dir: str = "./data"
    out_path: str = "./outputs/qm9_pretrain"
    wandb_project_name: str = "esa-pretraining"
    wandb_run_name: str = ""
    gradient_clip_val: float = 0.5
    max_epochs: int = 100
    
    # Data/loader
    num_workers: int = 4
    
    # Debug controls (optional, can be set via YAML/JSON)
    debug_subset_n: Optional[int] = None
    debug_verbose: bool = False
    
    # Domain-specific configs
    # Small molecules
    atom_types: int = 119  # Maximum atom types for molecules
    bond_types: int = 4    # Single, double, triple, aromatic
    molecule_max_atoms: int = 50
    
    # Proteins
    amino_acid_types: int = 20
    protein_max_residues: int = 500
    
    # RNA
    nucleotide_types: int = 4  # A, U, G, C
    rna_max_nucleotides: int = 200
    
    # Metabolites
    metabolite_atom_types: int = 50
    metabolite_max_atoms: int = 100
    
    # 3D geometric configs
    use_3d_coordinates: bool = True
    coordinate_dim: int = 3
    gaussian_kernels: int = 128
    cutoff_distance: float = 5.0
    max_neighbors: int = 32
    
    # Pretraining task configs
    pretraining_tasks: List[str] = None  # ["long_range_distance", "short_range_distance", "mlm", "bond_prediction"]
    task_weights: Dict[str, float] = None
    

    
    # Distance prediction
    distance_prediction_weight: float = 1.0
    distance_bins: int = 64
    max_distance: float = 10.0
    

    
    # Masked language modeling
    mlm_weight: float = 1.0
    mlm_mask_ratio: float = 0.15
    
    # Temperature for softmax (used in distance prediction)
    temperature: float = 0.1
    
    # Dataset caching control
    use_dataset_cache: bool = True  # Whether to use cached processed datasets
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 256, 256]
        if self.num_heads is None:
            self.num_heads = [8, 8, 8, 8]
        if self.layer_types is None:
            self.layer_types = ["S", "S", "S", "P"]
        if self.pretraining_tasks is None:
            self.pretraining_tasks = ["long_range_distance", "short_range_distance", "mlm"]
        if self.task_weights is None:
            self.task_weights = {
                "long_range_distance": 1.5,
                "short_range_distance": 1.0,
                "mlm": 0.3
            }



def nearest_multiple_of_8(n):
    return math.ceil(n / 8) * 8


class MultiDomainEncoder(nn.Module):
    """Multi-domain encoder that handles different molecular domains"""
    
    def __init__(self, config: PretrainingConfig):
        super().__init__()
        self.config = config
        
        # Domain-specific encoders
        self.molecule_encoder = self._create_molecule_encoder()
        self.protein_encoder = self._create_protein_encoder()
        self.rna_encoder = self._create_rna_encoder()
        self.metabolite_encoder = self._create_metabolite_encoder()
        
        # 3D geometric features
        if config.use_3d_coordinates:
            self.gaussian_layer = GaussianLayer(
                K=config.gaussian_kernels,
                edge_types=config.atom_types * config.atom_types
            )
            # 5 invariant features + gaussian_kernels edge features
            self.coordinate_projection = nn.Linear(5 + config.gaussian_kernels, config.hidden_dims[0])
        
        # Position encodings
        self.rwse_encoder = None
        self.lap_encoder = None
        if config.posenc and "RWSE" in config.posenc:
            self.rwse_encoder = KernelPENodeEncoder()
        if config.posenc and "LapPE" in config.posenc:
            self.lap_encoder = LapPENodeEncoder()
    
    def _create_molecule_encoder(self):
        """Create encoder for small molecules with reliable atomic features"""
        # Just atomic number embedding + period (most reliable features)
        self.atom_embedding = nn.Embedding(self.config.atom_types, self.config.hidden_dims[0] - 8)
        self.period_embedding = nn.Embedding(8, 8)  # Period is very reliable (1-7)
        
        return nn.Sequential(
            nn.Linear(self.config.hidden_dims[0], self.config.hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dims[0], self.config.hidden_dims[0])
        )
    
    def _create_protein_encoder(self):
        """Create simple encoder for proteins (fallback only)"""
        return nn.Sequential(
            nn.Embedding(self.config.amino_acid_types, self.config.hidden_dims[0]),
            nn.Linear(self.config.hidden_dims[0], self.config.hidden_dims[0])
        )
    
    def _create_rna_encoder(self):
        """Create encoder for RNA"""
        return nn.Sequential(
            nn.Embedding(self.config.nucleotide_types, self.config.hidden_dims[0]),
            nn.Linear(self.config.hidden_dims[0], self.config.hidden_dims[0])
        )
    
    def _create_metabolite_encoder(self):
        """Create encoder for metabolites"""
        return nn.Sequential(
            nn.Embedding(self.config.metabolite_atom_types, self.config.hidden_dims[0]),
            nn.Linear(self.config.hidden_dims[0], self.config.hidden_dims[0])
        )
    
    def forward(self, x, domain_type, pos=None, batch=None):
        """
        Forward pass for multi-domain encoding
        
        Args:
            x: Node features (atom types, amino acids, nucleotides, etc.)
            domain_type: Domain type ("molecule", "protein", "rna", "metabolite")
            pos: 3D coordinates (optional)
            batch: Batch indices
        """
        # Handle different input formats
        if x is None:
            raise ValueError("Input features cannot be None")
        
        # For QM9, x might be atomic numbers (z) instead of features
        if domain_type == "molecule" and x.dtype == torch.long:
            # Use atomic numbers with rich chemical features
            encoded = self._encode_molecule_features(x)
        elif domain_type == "molecule":
            # Use features if available, but enhance with chemical properties
            if x.dtype == torch.long:
                encoded = self._encode_molecule_features(x)
            else:
                # If x is already feature-based, use molecule encoder
                encoded = self.molecule_encoder(x)
        elif domain_type == "protein":
            # Proteins actually use atomic elements, same as molecules
            if x.dtype == torch.long:
                encoded = self._encode_molecule_features(x)  # Use molecular encoding for atoms
            else:
                encoded = self.protein_encoder(x)
        elif domain_type == "rna":
            encoded = self.rna_encoder(x)
        elif domain_type == "metabolite":
            encoded = self.metabolite_encoder(x)
        else:
            raise ValueError(f"Unknown domain type: {domain_type}")
        
        # Add 3D geometric features if available
        if pos is not None and self.config.use_3d_coordinates:
            try:
                # Expect a 1D batch index vector for to_dense_batch
                batch_vec = getattr(batch, 'batch', batch)
                geometric_features = self._compute_geometric_features(x, pos, batch_vec)
                encoded = encoded + geometric_features
            except Exception as e:
                print(f"Warning: Could not compute geometric features: {e}")
                # Continue without geometric features
                pass
        
        # Add position encodings
        if self.lap_encoder is not None and hasattr(batch, 'EigVals'):
            lap_pos_enc = self.lap_encoder(batch.EigVals, batch.EigVecs)
            encoded = torch.cat((encoded, lap_pos_enc), 1)
        
        if self.rwse_encoder is not None and hasattr(batch, 'pestat_RWSE'):
            rwse_pos_enc = self.rwse_encoder(batch.pestat_RWSE)
            encoded = torch.cat((encoded, rwse_pos_enc), 1)
        
        return encoded
    
    def _encode_molecule_features(self, atomic_numbers):
        """Encode atomic numbers with reliable chemical features"""
        # Simple, reliable period mapping for common atoms
        period_map = {
            1: 1,   # H
            6: 2, 7: 2, 8: 2, 9: 2,    # C, N, O, F (period 2)
            15: 3, 16: 3, 17: 3,       # P, S, Cl (period 3)
        }
        
        device = atomic_numbers.device
        
        # Main atomic embeddings
        atom_emb = self.atom_embedding(atomic_numbers)
        
        # Simple period feature (very reliable)
        periods = torch.tensor([period_map.get(z.item(), 0) for z in atomic_numbers], device=device)
        period_emb = self.period_embedding(periods)
        
        # Concatenate features
        combined_features = torch.cat([
            atom_emb,      # [batch_size, hidden_dim-8]
            period_emb,    # [batch_size, 8]
        ], dim=-1)       # [batch_size, hidden_dim]
        
        # Pass through molecule encoder
        return self.molecule_encoder(combined_features)
    

    
    def _compute_chemical_coordination(self, x_dense, batch):
        """Compute true chemical coordination numbers from molecular graph topology"""
        try:
            # Get dimensions first (always needed)
            batch_size = x_dense.size(0)
            max_nodes = x_dense.size(1)
            device = x_dense.device
            
            # Try to get edge_index from batch data
            if hasattr(batch, 'edge_index') and batch.edge_index is not None:
                from torch_scatter import scatter_add
                
                edge_index = batch.edge_index
                batch_idx = getattr(batch, 'batch', None)
                
                if batch_idx is not None:
                    # ✅ ROBUST: Handle bidirectional edges correctly
                    # Only count unique edges by keeping edges where src < dst
                    src, dst = edge_index[0], edge_index[1]
                    
                    # Create unique edge mask (avoid double counting)
                    unique_edge_mask = src < dst  # Only count edge (i,j) if i < j
                    if unique_edge_mask.sum() == 0:
                        # If no edges satisfy src < dst, edges might be unidirectional already
                        unique_edge_mask = torch.ones_like(src, dtype=torch.bool)
                    
                    unique_src = src[unique_edge_mask]
                    unique_dst = dst[unique_edge_mask]
                    
                    # Count coordination for each node
                    num_nodes = len(batch_idx)
                    coordination_sparse = torch.zeros(num_nodes, device=device, dtype=torch.float)
                    
                    # Each unique edge contributes 1 to both src and dst coordination
                    coordination_sparse.scatter_add_(0, unique_src, torch.ones_like(unique_src, dtype=torch.float))
                    coordination_sparse.scatter_add_(0, unique_dst, torch.ones_like(unique_dst, dtype=torch.float))
                    
                    # Convert to dense format [batch_size, max_nodes]
                    coordination_dense = torch.zeros(batch_size, max_nodes, device=device)
                    
                    # Fill dense tensor using batch indices
                    node_idx = 0
                    for graph_idx in range(batch_size):
                        nodes_in_graph = (batch_idx == graph_idx).sum().item()
                        if nodes_in_graph > 0:
                            end_idx = node_idx + nodes_in_graph
                            actual_nodes = min(nodes_in_graph, max_nodes)
                            coordination_dense[graph_idx, :actual_nodes] = coordination_sparse[node_idx:node_idx + actual_nodes]
                        node_idx += nodes_in_graph
                    
                    return coordination_dense
                else:
                    # Single graph case - simpler
                    src, dst = edge_index[0], edge_index[1]
                    unique_edge_mask = src < dst
                    if unique_edge_mask.sum() == 0:
                        unique_edge_mask = torch.ones_like(src, dtype=torch.bool)
                    
                    unique_src = src[unique_edge_mask]
                    unique_dst = dst[unique_edge_mask]
                    
                    coordination = torch.zeros(max_nodes, device=device, dtype=torch.float)
                    coordination.scatter_add_(0, unique_src, torch.ones_like(unique_src, dtype=torch.float))
                    coordination.scatter_add_(0, unique_dst, torch.ones_like(unique_dst, dtype=torch.float))
                    
                    return coordination.unsqueeze(0).expand(batch_size, -1)
            else:
                # No edge information available, return zeros
                return torch.zeros(batch_size, max_nodes, device=x_dense.device)
                
        except Exception as e:
            # Fallback: return zeros if anything goes wrong
            print(f"Warning: Could not compute chemical coordination: {e}")
            return torch.zeros(x_dense.size(0), x_dense.size(1), device=x_dense.device)
    
    def _compute_geometric_features(self, x, pos, batch):
        """Compute SE(3) invariant geometric features"""
        # Convert to dense format (let to_dense_batch pick the correct max per-batch)
        x_dense, batch_mask = to_dense_batch(x, batch, fill_value=0)
        pos_dense, _ = to_dense_batch(pos, batch, fill_value=0)
        
        n_graph, n_node = x_dense.size()
        
        # SE(3) INVARIANT: Compute pairwise distances only
        delta_pos = pos_dense.unsqueeze(1) - pos_dense.unsqueeze(2)
        dist = delta_pos.norm(dim=-1)  # INVARIANT to rotation/translation
    
        # SE(3) INVARIANT: Node-level coordination features
        # 1. Distance-based neighbors (local environment density)
        # Use tighter cutoffs that are more chemically meaningful
        close_cutoff = 3.0   # First coordination shell (~typical bond + van der Waals)
        medium_cutoff = self.config.cutoff_distance if hasattr(self.config, 'cutoff_distance') else 6.0
        
        # Count neighbors in different shells
        close_mask = (dist < close_cutoff) & (dist > 0.5)     # Close neighbors (likely bonded)
        medium_mask = (dist < medium_cutoff) & (dist > 0.5)   # Medium range (environment)
        
        close_coordination = close_mask.sum(dim=-1).float()    # [n_graph, n_node]
        distance_based_coordination = medium_mask.sum(dim=-1).float()  # [n_graph, n_node]
        
        # 2. Chemical bond-based coordination (true molecular graph neighbors)
        # We'll compute this from the actual molecular graph structure if available
        chemical_coordination = self._compute_chemical_coordination(x_dense, batch)
        
        # SE(3) INVARIANT: Distance statistics per node
        # Use close_mask for meaningful distance statistics
        masked_dist = dist.masked_fill(~close_mask, float('inf'))
        min_distances = masked_dist.min(dim=-1)[0].masked_fill(close_mask.sum(dim=-1) == 0, 0.0)
        
        masked_dist_finite = dist.masked_fill(~close_mask, 0.0)
        mean_distances = masked_dist_finite.sum(dim=-1) / (close_mask.sum(dim=-1) + 1e-8)
        
        # SE(3) INVARIANT: Combine features
        # Shape: [n_graph, n_node, feature_dim]
        invariant_features = torch.stack([
            chemical_coordination,        # True chemical bonds (sp3≈4, sp2≈3, sp≈2)
            close_coordination,          # Close spatial neighbors (~3Å, likely bonded)
            distance_based_coordination,  # Medium range environment (~6Å)
            min_distances,               # Closest neighbor distance
            mean_distances,              # Average close neighbor distance
        ], dim=-1)  # [n_graph, n_node, 5]
        
        # Apply Gaussian basis functions to distances for richer representation
        edge_type = x_dense.view(n_graph, n_node, 1) * self.config.atom_types + x_dense.view(n_graph, 1, n_node)
        gbf_feature = self.gaussian_layer(dist, edge_type)
        
        # Mask padding
        padding_mask = x_dense.eq(0)
        edge_features = gbf_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1), 0.0
        )
        
        # Aggregate edge features (sum over neighbors is permutation invariant)
        aggregated_edge_features = edge_features.sum(dim=-2)  # [n_graph, n_node, gaussian_kernels]
        
        # Combine invariant node features with aggregated edge features
        combined_features = torch.cat([
            invariant_features,  # [n_graph, n_node, 5]
            aggregated_edge_features  # [n_graph, n_node, gaussian_kernels]
        ], dim=-1)  # [n_graph, n_node, 5 + gaussian_kernels]
        
        # Project to hidden dimension
        geometric_features = self.coordinate_projection(combined_features)
        
        # Convert back to sparse format
        geometric_features = geometric_features[batch_mask]
        
        return geometric_features


class PretrainingTasks(nn.Module):
    """Module containing all pretraining tasks"""
    
    def __init__(self, config: PretrainingConfig):
        super().__init__()
        self.config = config
        
        # Task-specific heads
        self.long_range_distance_head = self._create_long_range_distance_head()
        self.distance_head = self._create_distance_head()  # Used by short_range_distance
        self.mlm_head = self._create_mlm_head()
    
    def _create_long_range_distance_head(self):
        """Head for long-range distance prediction (global 3D structure learning)"""
        # Use classification like the working distance_prediction_loss
        return nn.Sequential(
            nn.Linear(self.config.graph_dim * 2, self.config.graph_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.graph_dim, self.config.graph_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.graph_dim // 2, self.config.distance_bins)  # Classify into distance bins
        )    


    def _create_distance_head(self):
        """Head for distance prediction"""
        return nn.Sequential(
            nn.Linear(self.config.graph_dim * 2, self.config.graph_dim),
            nn.ReLU(),
            nn.Linear(self.config.graph_dim, self.config.distance_bins)
        )
    

    def _create_mlm_head(self):
        """Head for masked language modeling"""
        max_types = max(
            self.config.atom_types,
            self.config.amino_acid_types,
            self.config.nucleotide_types,
            self.config.metabolite_atom_types
        )
        return nn.Sequential(
            nn.Linear(self.config.graph_dim, self.config.graph_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.graph_dim // 2, max_types)
        )
    


    def long_range_distance_loss(self, node_embeddings, data):
        """
        Long-range distance prediction loss for global 3D structure learning.
        
        Predicts distances between all atom pairs (or sampled pairs for large proteins)
        to learn overall molecular geometry. This is completely SE(3) invariant.
        """
        if not hasattr(data, 'pos') or data.pos is None:
            return torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)
        
        pos = data.pos
        num_atoms = pos.size(0)
        
        # For efficiency with large proteins, sample pairs
        max_pairs = min(2000, num_atoms * (num_atoms - 1) // 2)
        
        if num_atoms <= 50:
            # Small molecules: use all pairs
            i_indices, j_indices = torch.triu_indices(
                num_atoms, num_atoms, offset=1, device=node_embeddings.device
            )
        else:
            # Large proteins: intelligent sampling strategy
            pairs = []
            
            # 1. Sample local pairs (distance < 8.0 Å) - most important for structure
            with torch.no_grad():
                dist_matrix = torch.cdist(pos, pos)
                local_mask = (dist_matrix < 8.0) & (dist_matrix > 0)
                local_i, local_j = torch.where(local_mask)
                # Keep upper triangular
                valid_local = local_i < local_j
                local_pairs = list(zip(local_i[valid_local].tolist(), local_j[valid_local].tolist()))
                pairs.extend(local_pairs[:max_pairs//2])  # Half for local pairs
            
            # 2. Sample global pairs for long-range structure
            remaining = max_pairs - len(pairs)
            if remaining > 0:
                for _ in range(remaining):
                    i, j = torch.randint(0, num_atoms, (2,)).tolist()
                    if i != j:
                        pairs.append((min(i, j), max(i, j)))
            
            # Remove duplicates and convert to tensors
            pairs = list(set(pairs))[:max_pairs]
            if pairs:
                i_indices, j_indices = zip(*pairs)
                i_indices = torch.tensor(i_indices, device=node_embeddings.device)
                j_indices = torch.tensor(j_indices, device=node_embeddings.device)
            else:
                return torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)
        
        if len(i_indices) == 0:
            return torch.tensor(0.0, device=node_embeddings.device, requires_grad=True)
        
        # Get node embeddings for pairs
        emb_i = node_embeddings[i_indices]
        emb_j = node_embeddings[j_indices]
        pair_embeddings = torch.cat([emb_i, emb_j], dim=-1)
        
        # Predict distance bins using long-range distance head
        distance_logits = self.long_range_distance_head(pair_embeddings)
        
        # True distances (SE(3) invariant ground truth)
        pos_i = pos[i_indices]
        pos_j = pos[j_indices]
        true_distances = torch.norm(pos_i - pos_j, dim=1)
        
        # Convert distances to bins (same as distance_prediction_loss)
        distance_bins = torch.clamp(
            (true_distances / self.config.max_distance * self.config.distance_bins).long(),
            0, self.config.distance_bins - 1
        )
        
        # Debug: Check predictions occasionally
        with torch.no_grad():
            pred_bins = distance_logits.argmax(dim=-1)
            true_mean = true_distances.mean().item()
            pred_dist_mean = (pred_bins.float() / self.config.distance_bins * self.config.max_distance).mean().item()
            
            # Log debug info occasionally
            if torch.rand(1).item() < 0.01:  # 1% chance to log
                print(f"SE3 Debug - True dist: {true_mean:.3f}Å, Pred dist: {pred_dist_mean:.3f}Å")
        
        # Use cross-entropy loss (same as distance_prediction_loss)
        loss = F.cross_entropy(distance_logits, distance_bins, reduction='mean')
        
        return loss
    

    

    def short_range_distance_loss(self, node_embeddings, edge_index, distances, mask):
        """Compute short-range distance loss (local chemical bonds)"""
        source_emb = node_embeddings[edge_index[0]]
        target_emb = node_embeddings[edge_index[1]]
        edge_emb = torch.cat([source_emb, target_emb], dim=-1)
        
        logits = self.distance_head(edge_emb)
        
        # Convert distances to bins
        distance_bins = torch.clamp(
            (distances / self.config.max_distance * self.config.distance_bins).long(),
            0, self.config.distance_bins - 1
        )
        
        loss = F.cross_entropy(logits[mask], distance_bins[mask], reduction='mean')
        return loss
    

    def mlm_loss(self, node_embeddings, data):
        """
        Compute masked language modeling loss
        """
        if not (hasattr(data, 'mlm_mask') and hasattr(data, 'original_types') and hasattr(data, 'masked_types')):
            return torch.tensor(0.0, device=node_embeddings.device)

        mask = data.mlm_mask
        original_types = data.original_types
        masked_types = getattr(data, 'masked_types', original_types)

        # Debug logging (remove after confirming)
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 1
            
        if self._step_count % 100 == 1:  # Log every 100 steps
            mask_ratio = mask.sum().item() / mask.size(0)
            print(f"MLM Debug - Step {self._step_count}: Mask ratio = {mask_ratio:.3f}, Masked nodes = {mask.sum().item()}/{mask.size(0)}")

        logits = self.mlm_head(node_embeddings)
        loss = F.cross_entropy(logits[mask], original_types[mask], reduction='mean')
        return loss
    



class PretrainingESAModel(pl.LightningModule):
    """
    Comprehensive pretraining ESA model for multi-domain geometric deep learning
    """
    
    def __init__(self, config: PretrainingConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(vars(config))
        
        # Multi-domain encoder
        self.encoder = MultiDomainEncoder(config)
        
        # ESA backbone
        st_args = dict(
            num_outputs=32,
            dim_output=config.graph_dim,
            xformers_or_torch_attn=config.xformers_or_torch_attn,
            dim_hidden=config.hidden_dims,
            num_heads=config.num_heads,
            sab_dropout=config.sab_dropout,
            mab_dropout=config.mab_dropout,
            pma_dropout=config.pma_dropout,
            use_mlps=config.use_mlps,
            mlp_hidden_size=config.mlp_hidden_size,
            mlp_type=config.mlp_type,
            norm_type=config.norm_type,
            node_or_edge=config.apply_attention_on,
            residual_dropout=config.attn_residual_dropout,
            set_max_items=nearest_multiple_of_8(config.set_max_items + 1),
            use_bfloat16=config.use_bfloat16,
            layer_types=config.layer_types,
            num_mlp_layers=config.num_mlp_layers,
            pre_or_post=config.pre_or_post,
            pma_residual_dropout=config.pma_residual_dropout,
            use_mlp_ln=config.use_mlp_ln,
            mlp_dropout=config.mlp_dropout,
        )
        
        self.esa_backbone = ESA(**st_args)
        
        # Pretraining tasks
        self.pretraining_tasks = PretrainingTasks(config)
        
        # Output projection for node-level tasks
        if config.apply_attention_on == "edge":
            # For edge attention, input is concatenated source and target embeddings
            # We'll create the MLP dynamically in forward pass to handle variable edge dimensions
            self.node_edge_mlp = None  # Will be created dynamically
        else:
            if config.mlp_type in ["standard", "gated_mlp"]:
                # In node attention mode, the encoder outputs node embeddings of size graph_dim
                self.node_mlp = SmallMLP(
                    in_dim=config.graph_dim,
                    inter_dim=128,
                    out_dim=config.hidden_dims[0],
                    use_ln=False,
                    dropout_p=0,
                    num_layers=config.num_mlp_layers if config.num_mlp_layers > 1 else config.num_mlp_layers + 1,
                )
        
        # Normalization
        if config.norm_type == "BN":
            norm_fn = BN
        elif config.norm_type == "LN":
            norm_fn = LN
        
        self.mlp_norm = norm_fn(config.hidden_dims[0])
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.test_metrics = defaultdict(list)

        # If we are in node-attention mode and need per-node outputs for a node-level task
        # (e.g., coordinate denoising), ensure we align ESA outputs back to node space.
        if self.config.apply_attention_on == "node":
            self.config.is_node_task = True
    
    def forward(self, batch, domain_type="molecule"):
        """
        Forward pass for pretraining
        
        Args:
            batch: PyTorch Geometric batch
            domain_type: Type of molecular domain
        """
        edge_index, batch_mapping = batch.edge_index, batch.batch
        pos = getattr(batch, 'pos', None)
        
        # Handle different input formats for QM9
        if hasattr(batch, 'z') and batch.z is not None:
            # QM9 uses atomic numbers (z) instead of features (x)
            x = batch.z
        elif hasattr(batch, 'x') and batch.x is not None:
            x = batch.x
        else:
            raise ValueError("Batch must have either 'x' or 'z' attribute")
        
        # Encode nodes based on domain
        node_embeddings = self.encoder(x, domain_type, pos, batch)
        
        # Debug check
        if node_embeddings is None:
            raise ValueError("Encoder returned None embeddings")
        
        # Apply attention mechanism
        if self.config.apply_attention_on == "edge":
            source = node_embeddings[edge_index[0, :], :]
            target = node_embeddings[edge_index[1, :], :]
            h = torch.cat((source, target), dim=1)
            
            edge_attr = getattr(batch, 'edge_attr', None)
            if edge_attr is not None:
                h = torch.cat((h, edge_attr.float()), dim=1)
            
            # Create MLP dynamically if needed
            if self.node_edge_mlp is None:
                in_dim = h.shape[1]
                self.node_edge_mlp = SmallMLP(
                    in_dim=in_dim,
                    inter_dim=128,
                    out_dim=self.config.hidden_dims[0],
                    use_ln=False,
                    dropout_p=0,
                    num_layers=self.config.num_mlp_layers if self.config.num_mlp_layers > 1 else self.config.num_mlp_layers + 1,
                ).to(h.device)
            
            # Ensure tensors are on the same device
            device = node_embeddings.device
            h = self.node_edge_mlp(h.to(device))
            edge_index = edge_index.to(device)
            batch_mapping = batch_mapping.to(device)
            edge_batch_index = batch_mapping.index_select(0, edge_index[0, :]).to(device)

            # Determine max set size dynamically if not provided
            if self.config.set_max_items and self.config.set_max_items > 0:
                num_max_items = nearest_multiple_of_8(self.config.set_max_items + 1)
            else:
                counts = torch.bincount(edge_batch_index)
                num_max_items = int(counts.max().item()) if counts.numel() > 0 else 1
                num_max_items = max(1, num_max_items)
                num_max_items = nearest_multiple_of_8(num_max_items + 1)

            if getattr(self.config, 'debug_verbose', False):
                uniq, cnt = torch.unique(edge_batch_index, return_counts=True)
                print(f"[DEBUG] Edge mode - h:{tuple(h.shape)} edge_index:{tuple(edge_index.shape)} batches:{int(uniq.numel())} max_edges_per_graph:{int(cnt.max().item() if cnt.numel()>0 else 0)} num_max_items:{num_max_items}")
            
            h, _ = to_dense_batch(h, edge_batch_index, fill_value=0, max_num_nodes=num_max_items)
            h = self.esa_backbone(h, edge_index, batch_mapping, num_max_items=num_max_items)
        else:
            # Ensure tensors are on the same device
            device = node_embeddings.device
            h = self.mlp_norm(self.node_mlp(node_embeddings))
            batch_mapping = batch_mapping.to(device)
            if edge_index is not None:
                edge_index = edge_index.to(device)
            else:
                # Node attention path does not require edges; pass an empty tensor
                edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

            # Determine max set size for nodes
            if self.config.set_max_items and self.config.set_max_items > 0:
                num_max_items = nearest_multiple_of_8(self.config.set_max_items + 1)
            else:
                counts = torch.bincount(batch_mapping)
                num_max_items = int(counts.max().item()) if counts.numel() > 0 else 1
                num_max_items = max(1, num_max_items)
                num_max_items = nearest_multiple_of_8(num_max_items + 1)

            if getattr(self.config, 'debug_verbose', False):
                uniq, cnt = torch.unique(batch_mapping, return_counts=True)
                print(f"[DEBUG] Node mode - h:{tuple(h.shape)} edge_index:{tuple(edge_index.shape)} batches:{int(uniq.numel())} max_nodes_per_graph:{int(cnt.max().item() if cnt.numel()>0 else 0)} num_max_items:{num_max_items}")

            h, dense_batch_index = to_dense_batch(h, batch_mapping, fill_value=0, max_num_nodes=num_max_items)
            h = self.esa_backbone(h, edge_index, batch_mapping, num_max_items=num_max_items)
            
            if self.config.is_node_task and h.dim() == 3:
                # Map dense ESA outputs back to the original node ordering only if ESA kept per-node outputs
                h = h[dense_batch_index]
        
        return h, node_embeddings
    
    def _compute_pretraining_losses(self, batch, graph_embeddings, node_embeddings):
        """Compute all pretraining task losses"""
        losses = {}
        total_loss = 0.0               

        # Long-range distance learning (global 3D structure)
        if "long_range_distance" in self.config.pretraining_tasks:
            losses['long_range_distance'] = self.pretraining_tasks.long_range_distance_loss(node_embeddings, batch)
            total_loss += self.config.task_weights['long_range_distance'] * losses['long_range_distance']
        
        # Short-range distance learning (local chemical bonds)
        if "short_range_distance" in self.config.pretraining_tasks:
            if hasattr(batch, 'edge_index') and hasattr(batch, 'pos'):
                # Compute distances from coordinates on-the-fly
                edge_index = batch.edge_index
                pos = batch.pos
                distances = torch.norm(pos[edge_index[1]] - pos[edge_index[0]], dim=1)
                
                mask = torch.rand(edge_index.size(1)) < 0.15
                dist_loss = self.pretraining_tasks.short_range_distance_loss(
                    node_embeddings, edge_index, distances, mask
                )
                losses['short_range_distance'] = dist_loss
                total_loss += self.config.task_weights['short_range_distance'] * dist_loss
        

        # Masked language modeling
        if "mlm" in self.config.pretraining_tasks:
            losses['mlm'] = self.pretraining_tasks.mlm_loss(node_embeddings, batch)
            total_loss += self.config.task_weights['mlm'] * losses['mlm']
        

        return losses, total_loss
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        graph_embeddings, node_embeddings = self.forward(batch)
        
        losses, total_loss = self._compute_pretraining_losses(batch, graph_embeddings, node_embeddings)
        
        # Log individual losses with more detailed info
        for task_name, loss_value in losses.items():
            self.log(f"train_{task_name}_loss", loss_value, prog_bar=True)
            # Also log to console occasionally for debugging
            if batch_idx % 50 == 0:
                print(f"  {task_name}: {loss_value:.4f}")
        
        # Print detailed loss breakdown occasionally
        if batch_idx % 50 == 0:
            loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
            print(f"Step {batch_idx} - {loss_str} | Total: {total_loss:.4f}")
        
        self.log("train_total_loss", total_loss, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        graph_embeddings, node_embeddings = self.forward(batch)
        
        losses, total_loss = self._compute_pretraining_losses(batch, graph_embeddings, node_embeddings)
        
        # Log individual losses
        for task_name, loss_value in losses.items():
            self.log(f"val_{task_name}_loss", loss_value)
        
        self.log("val_total_loss", total_loss)
        
        return total_loss
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        graph_embeddings, node_embeddings = self.forward(batch)
        
        losses, total_loss = self._compute_pretraining_losses(batch, graph_embeddings, node_embeddings)
        
        # Log individual losses
        for task_name, loss_value in losses.items():
            self.log(f"test_{task_name}_loss", loss_value)
        
        self.log("test_total_loss", total_loss)
        
        return total_loss
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        # Simple optimizer setup for all parameters
        optimizer = bnb.optim.AdamW8bit(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.optimiser_weight_decay,
        )

        # Cosine with warmup
        total_epochs = getattr(self.config, 'max_epochs', 100)
        warmup_frac = 0.1
        warmup_epochs = max(1, int(total_epochs * warmup_frac))

        def lr_lambda(current_epoch: int):
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(warmup_epochs)
            # Cosine decay from 1.0 to 0.1 over the remaining epochs
            progress = (current_epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return 0.1 + 0.9 * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": self.config.monitor_loss_name,
        }
        ]
    
    def get_embeddings(self, batch, domain_type="molecule"):
        """Get embeddings for downstream tasks"""
        with torch.no_grad():
            graph_embeddings, node_embeddings = self.forward(batch, domain_type)
            return graph_embeddings, node_embeddings


def create_pretraining_config(**kwargs) -> PretrainingConfig:
    """Helper function to create pretraining configuration"""
    config = PretrainingConfig()
    
    # Update with provided kwargs
    for key, value in kwargs.items():
        # Accept all keys to avoid noisy warnings; attach as attributes if unknown
        try:
            setattr(config, key, value)
        except Exception:
            pass
    
    return config


# Example usage and configuration
if __name__ == "__main__":
    # Example configuration for multi-domain pretraining
    config = create_pretraining_config(
        # Model architecture
        num_features=128,
        graph_dim=256,
        hidden_dims=[256, 256, 256, 256],
        num_heads=[8, 8, 8, 8],
        layer_types=["S", "S", "S", "P"],
        
        # Pretraining tasks
        pretraining_tasks=[
            "long_range_distance",
            "short_range_distance",
            "mlm"
        ],
        
        # Task weights
        task_weights={
            "long_range_distance": 1.5,
            "short_range_distance": 1.0,
            "mlm": 0.3
        },
        
        # Training parameters
        batch_size=32,
        lr=0.001,
        early_stopping_patience=30
    )
    
    # Create model
    model = PretrainingESAModel(config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Pretraining tasks: {config.pretraining_tasks}")
    print(f"Task weights: {config.task_weights}")
