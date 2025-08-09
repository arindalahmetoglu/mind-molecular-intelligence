import torch
import math
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
    pretraining_tasks: List[str] = None  # ["coordinate_denoising", "atom_type_prediction", "bond_prediction", "distance_prediction", "angle_prediction"]
    task_weights: Dict[str, float] = None
    
    # Coordinate denoising
    coordinate_noise_std: float = 0.1
    coordinate_denoising_weight: float = 1.0
    
    # Atom type prediction
    atom_type_prediction_weight: float = 1.0
    atom_type_mask_ratio: float = 0.15
    
    # Bond prediction
    bond_prediction_weight: float = 1.0
    bond_mask_ratio: float = 0.15
    
    # Distance prediction
    distance_prediction_weight: float = 1.0
    distance_bins: int = 64
    max_distance: float = 10.0
    
    # Angle prediction
    angle_prediction_weight: float = 1.0
    angle_bins: int = 36
    
    # Masked language modeling
    mlm_weight: float = 1.0
    mlm_mask_ratio: float = 0.15
    
    # Contrastive learning
    contrastive_weight: float = 1.0
    temperature: float = 0.1
    
    # Graph-level tasks
    graph_property_prediction_weight: float = 1.0
    graph_property_types: List[str] = None  # ["molecular_weight", "logp", "tpsa", etc.]
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 256, 256]
        if self.num_heads is None:
            self.num_heads = [8, 8, 8, 8]
        if self.layer_types is None:
            self.layer_types = ["S", "S", "S", "P"]
        if self.pretraining_tasks is None:
            self.pretraining_tasks = ["coordinate_denoising", "atom_type_prediction", "bond_prediction"]
        if self.task_weights is None:
            self.task_weights = {
                "coordinate_denoising": 1.0,
                "atom_type_prediction": 1.0,
                "bond_prediction": 1.0,
                "distance_prediction": 1.0,
                "angle_prediction": 1.0,
                "mlm": 1.0,
                "contrastive": 1.0,
                "graph_property": 1.0
            }
        if self.graph_property_types is None:
            self.graph_property_types = ["molecular_weight", "logp", "tpsa"]


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
            self.coordinate_projection = nn.Linear(config.gaussian_kernels, config.hidden_dims[0])
        
        # Position encodings
        self.rwse_encoder = None
        self.lap_encoder = None
        if config.posenc and "RWSE" in config.posenc:
            self.rwse_encoder = KernelPENodeEncoder()
        if config.posenc and "LapPE" in config.posenc:
            self.lap_encoder = LapPENodeEncoder()
    
    def _create_molecule_encoder(self):
        """Create encoder for small molecules"""
        return nn.Sequential(
            nn.Embedding(self.config.atom_types, self.config.hidden_dims[0]),
            nn.Linear(self.config.hidden_dims[0], self.config.hidden_dims[0])
        )
    
    def _create_protein_encoder(self):
        """Create encoder for proteins"""
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
            # Use atomic numbers directly for embedding
            encoded = self.molecule_encoder(x)
        elif domain_type == "molecule":
            # Use features if available
            encoded = self.molecule_encoder(x)
        elif domain_type == "protein":
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
    
    def _compute_geometric_features(self, x, pos, batch):
        """Compute 3D geometric features using Gaussian basis functions"""
        # Convert to dense format (let to_dense_batch pick the correct max per-batch)
        x_dense, batch_mask = to_dense_batch(x, batch, fill_value=0)
        pos_dense, _ = to_dense_batch(pos, batch, fill_value=0)
        
        n_graph, n_node = x_dense.size()
        
        # Compute pairwise distances
        delta_pos = pos_dense.unsqueeze(1) - pos_dense.unsqueeze(2)
        dist = delta_pos.norm(dim=-1)
        
        # Compute edge types
        edge_type = x_dense.view(n_graph, n_node, 1) * self.config.atom_types + x_dense.view(n_graph, 1, n_node)
        
        # Apply Gaussian basis functions
        gbf_feature = self.gaussian_layer(dist, edge_type)
        
        # Mask padding
        padding_mask = x_dense.eq(0)
        edge_features = gbf_feature.masked_fill(
            padding_mask.unsqueeze(1).unsqueeze(-1), 0.0
        )
        
        # Project to hidden dimension
        geometric_features = self.coordinate_projection(edge_features.sum(dim=-2))
        
        # Convert back to sparse format
        geometric_features = geometric_features[batch_mask]
        
        return geometric_features


class PretrainingTasks(nn.Module):
    """Module containing all pretraining tasks"""
    
    def __init__(self, config: PretrainingConfig):
        super().__init__()
        self.config = config
        
        # Task-specific heads
        self.coordinate_denoising_head = self._create_coordinate_denoising_head()
        self.atom_type_head = self._create_atom_type_head()
        self.bond_head = self._create_bond_head()
        self.distance_head = self._create_distance_head()
        self.angle_head = self._create_angle_head()
        self.mlm_head = self._create_mlm_head()
        self.graph_property_head = self._create_graph_property_head()
        
        # Contrastive learning
        self.contrastive_projection = nn.Sequential(
            nn.Linear(config.graph_dim, config.graph_dim),
            nn.ReLU(),
            nn.Linear(config.graph_dim, 128)
        )
    
    def _create_coordinate_denoising_head(self):
        """Head for coordinate denoising task"""
        return nn.Sequential(
            nn.Linear(self.config.graph_dim, self.config.graph_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.graph_dim // 2, self.config.coordinate_dim)
        )
    
    def _create_atom_type_head(self):
        """Head for atom type prediction"""
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
    
    def _create_bond_head(self):
        """Head for bond prediction"""
        return nn.Sequential(
            nn.Linear(self.config.graph_dim * 2, self.config.graph_dim),
            nn.ReLU(),
            nn.Linear(self.config.graph_dim, self.config.bond_types)
        )
    
    def _create_distance_head(self):
        """Head for distance prediction"""
        return nn.Sequential(
            nn.Linear(self.config.graph_dim * 2, self.config.graph_dim),
            nn.ReLU(),
            nn.Linear(self.config.graph_dim, self.config.distance_bins)
        )
    
    def _create_angle_head(self):
        """Head for angle prediction"""
        return nn.Sequential(
            nn.Linear(self.config.graph_dim * 3, self.config.graph_dim),
            nn.ReLU(),
            nn.Linear(self.config.graph_dim, self.config.angle_bins)
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
    
    def _create_graph_property_head(self):
        """Head for graph property prediction"""
        num_properties = len(self.config.graph_property_types)
        return nn.Sequential(
            nn.Linear(self.config.graph_dim, self.config.graph_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.graph_dim // 2, num_properties)
        )
    
    def coordinate_denoising_loss(self, node_embeddings, noisy_pos, clean_pos, mask):
        """Compute coordinate denoising loss"""
        predicted_delta = self.coordinate_denoising_head(node_embeddings)
        target_delta = clean_pos - noisy_pos
        
        loss = F.mse_loss(predicted_delta[mask], target_delta[mask], reduction='mean')
        return loss
    
    def atom_type_prediction_loss(self, node_embeddings, atom_types, mask):
        """Compute atom type prediction loss"""
        logits = self.atom_type_head(node_embeddings)
        loss = F.cross_entropy(logits[mask], atom_types[mask], reduction='mean')
        return loss
    
    def bond_prediction_loss(self, node_embeddings, edge_index, bond_types, mask):
        """Compute bond prediction loss"""
        source_emb = node_embeddings[edge_index[0]]
        target_emb = node_embeddings[edge_index[1]]
        edge_emb = torch.cat([source_emb, target_emb], dim=-1)
        
        logits = self.bond_head(edge_emb)
        loss = F.cross_entropy(logits[mask], bond_types[mask], reduction='mean')
        return loss
    
    def distance_prediction_loss(self, node_embeddings, edge_index, distances, mask):
        """Compute distance prediction loss"""
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
    
    def angle_prediction_loss(self, node_embeddings, angle_indices, angles, mask):
        """Compute angle prediction loss"""
        node1_emb = node_embeddings[angle_indices[:, 0]]
        node2_emb = node_embeddings[angle_indices[:, 1]]
        node3_emb = node_embeddings[angle_indices[:, 2]]
        angle_emb = torch.cat([node1_emb, node2_emb, node3_emb], dim=-1)
        
        logits = self.angle_head(angle_emb)
        
        # Convert angles to bins
        angle_bins = torch.clamp(
            (angles / (2 * np.pi) * self.config.angle_bins).long(),
            0, self.config.angle_bins - 1
        )
        
        loss = F.cross_entropy(logits[mask], angle_bins[mask], reduction='mean')
        return loss
    
    def mlm_loss(self, node_embeddings, original_types, masked_types, mask):
        """Compute masked language modeling loss"""
        logits = self.mlm_head(node_embeddings)
        loss = F.cross_entropy(logits[mask], original_types[mask], reduction='mean')
        return loss
    
    def contrastive_loss(self, graph_embeddings, temperature=0.1):
        """Compute contrastive learning loss"""
        # Project to contrastive space
        projections = self.contrastive_projection(graph_embeddings)
        projections = F.normalize(projections, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T) / temperature
        
        # Create positive pairs (same graph, different augmentations)
        batch_size = graph_embeddings.size(0)
        labels = torch.arange(batch_size, device=graph_embeddings.device)
        
        loss = F.cross_entropy(similarity_matrix, labels, reduction='mean')
        return loss
    
    def graph_property_loss(self, graph_embeddings, properties):
        """Compute graph property prediction loss"""
        logits = self.graph_property_head(graph_embeddings)
        loss = F.mse_loss(logits, properties, reduction='mean')
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
            
            if self.config.is_node_task:
                h = h[dense_batch_index]
        
        return h, node_embeddings
    
    def _compute_pretraining_losses(self, batch, graph_embeddings, node_embeddings):
        """Compute all pretraining task losses"""
        losses = {}
        total_loss = 0.0
        
        # Coordinate denoising
        if "coordinate_denoising" in self.config.pretraining_tasks:
            if hasattr(batch, 'pos') and hasattr(batch, 'clean_pos'):
                mask = torch.rand(batch.pos.size(0)) < 0.15  # 15% masking ratio
                coord_loss = self.pretraining_tasks.coordinate_denoising_loss(
                    node_embeddings, batch.pos, batch.clean_pos, mask
                )
                losses['coordinate_denoising'] = coord_loss
                total_loss += self.config.task_weights['coordinate_denoising'] * coord_loss
        
        # Atom type prediction
        if "atom_type_prediction" in self.config.pretraining_tasks:
            # Use atomic numbers (z) for atom type prediction
            if hasattr(batch, 'z') and batch.z is not None:
                atom_types = batch.z
                mask = torch.rand(atom_types.size(0)) < self.config.atom_type_mask_ratio
                atom_loss = self.pretraining_tasks.atom_type_prediction_loss(
                    node_embeddings, atom_types, mask
                )
                losses['atom_type_prediction'] = atom_loss
                total_loss += self.config.task_weights['atom_type_prediction'] * atom_loss
        
        # Bond prediction
        if "bond_prediction" in self.config.pretraining_tasks:
            if hasattr(batch, 'bond_types') and hasattr(batch, 'edge_index'):
                mask = torch.rand(batch.edge_index.size(1)) < self.config.bond_mask_ratio
                bond_loss = self.pretraining_tasks.bond_prediction_loss(
                    node_embeddings, batch.edge_index, batch.bond_types, mask
                )
                losses['bond_prediction'] = bond_loss
                total_loss += self.config.task_weights['bond_prediction'] * bond_loss
        
        # Distance prediction
        if "distance_prediction" in self.config.pretraining_tasks:
            if hasattr(batch, 'distances') and hasattr(batch, 'edge_index'):
                mask = torch.rand(batch.edge_index.size(1)) < 0.15
                dist_loss = self.pretraining_tasks.distance_prediction_loss(
                    node_embeddings, batch.edge_index, batch.distances, mask
                )
                losses['distance_prediction'] = dist_loss
                total_loss += self.config.task_weights['distance_prediction'] * dist_loss
        
        # Angle prediction
        if "angle_prediction" in self.config.pretraining_tasks:
            if hasattr(batch, 'angles') and hasattr(batch, 'angle_indices'):
                mask = torch.rand(batch.angle_indices.size(0)) < 0.15
                angle_loss = self.pretraining_tasks.angle_prediction_loss(
                    node_embeddings, batch.angle_indices, batch.angles, mask
                )
                losses['angle_prediction'] = angle_loss
                total_loss += self.config.task_weights['angle_prediction'] * angle_loss
        
        # Masked language modeling
        if "mlm" in self.config.pretraining_tasks:
            if hasattr(batch, 'original_types') and (hasattr(batch, 'mlm_mask') or hasattr(batch, 'masked_types')):
                # Prefer the exact mask used in the transform if available
                if hasattr(batch, 'mlm_mask') and batch.mlm_mask is not None:
                    mask = batch.mlm_mask
                else:
                    if hasattr(batch, 'z') and batch.z is not None:
                        num_nodes = batch.z.size(0)
                    else:
                        num_nodes = batch.x.size(0)
                    mask = torch.rand(num_nodes) < self.config.mlm_mask_ratio
                mlm_loss = self.pretraining_tasks.mlm_loss(
                    node_embeddings, batch.original_types, getattr(batch, 'masked_types', batch.original_types), mask
                )
                losses['mlm'] = mlm_loss
                total_loss += self.config.task_weights['mlm'] * mlm_loss
        
        # Contrastive learning
        if "contrastive" in self.config.pretraining_tasks:
            contrastive_loss = self.pretraining_tasks.contrastive_loss(
                graph_embeddings, self.config.temperature
            )
            losses['contrastive'] = contrastive_loss
            total_loss += self.config.task_weights['contrastive'] * contrastive_loss
        
        # Graph property prediction
        if "graph_property" in self.config.pretraining_tasks:
            if hasattr(batch, 'graph_properties'):
                graph_prop_loss = self.pretraining_tasks.graph_property_loss(
                    graph_embeddings, batch.graph_properties
                )
                losses['graph_property'] = graph_prop_loss
                total_loss += self.config.task_weights['graph_property'] * graph_prop_loss
        
        return losses, total_loss
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        graph_embeddings, node_embeddings = self.forward(batch)
        
        losses, total_loss = self._compute_pretraining_losses(batch, graph_embeddings, node_embeddings)
        
        # Log individual losses
        for task_name, loss_value in losses.items():
            self.log(f"train_{task_name}_loss", loss_value, prog_bar=True)
        
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
        optimizer = bnb.optim.AdamW8bit(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.optimiser_weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=self.config.early_stopping_patience // 2,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.config.monitor_loss_name,
            },
        }
    
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
            "coordinate_denoising",
            "atom_type_prediction", 
            "bond_prediction",
            "distance_prediction",
            "contrastive"
        ],
        
        # Task weights
        task_weights={
            "coordinate_denoising": 1.0,
            "atom_type_prediction": 1.0,
            "bond_prediction": 1.0,
            "distance_prediction": 0.5,
            "contrastive": 0.1
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
