import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from typing import Optional, List, Dict, Any
import random
import math


class AddRandomNoise(BaseTransform):
    """Add random noise to 3D coordinates for coordinate denoising task.

    Supports per-graph sigma sampling and emits a coordinate mask for loss.
    """
    
    def __init__(
        self,
        noise_std: float = 0.1,
        coordinate_dim: int = 3,
        mask_ratio: float = 0.15,
        sample_sigma: bool = False,
        sigma_min: float = None,
        sigma_max: float = None,
    ):
        self.noise_std = float(noise_std)
        self.coordinate_dim = int(coordinate_dim)
        self.mask_ratio = float(mask_ratio)
        self.sample_sigma = bool(sample_sigma)
        self.sigma_min = float(sigma_min) if sigma_min is not None else None
        self.sigma_max = float(sigma_max) if sigma_max is not None else None
    
    def _sample_sigma(self) -> float:
        if self.sample_sigma and self.sigma_min is not None and self.sigma_max is not None:
            lo = max(1e-6, min(self.sigma_min, self.sigma_max))
            hi = max(self.sigma_min, self.sigma_max)
            u = torch.rand(())
            sigma = float(torch.exp(torch.log(torch.tensor(lo)) + u * (torch.log(torch.tensor(hi)) - torch.log(torch.tensor(lo)))))
            return sigma
        return self.noise_std
    
    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'pos') and data.pos is not None:
            # Store the original, clean positions
            data.clean_pos = data.pos.clone()

            # Determine the noise standard deviation for this specific graph/batch
            current_noise_std = self.noise_std
            if self.sigma_min is not None and self.sigma_max is not None and self.sigma_min < self.sigma_max:
                # Sample noise_std from a log-uniform distribution for better perceptual range
                log_min = math.log(self.sigma_min)
                log_max = math.log(self.sigma_max)
                current_noise_std = math.exp(random.uniform(log_min, log_max))
            
            # Generate and apply simple Gaussian noise
            noise = torch.randn_like(data.pos) * current_noise_std
            data.pos = data.pos + noise # This is now the noisy position
            
            # Store a coordinate mask; allow partial masking for denoising objective
            num_nodes = data.pos.size(0)
            if self.mask_ratio >= 1.0:
                data.coord_mask = torch.ones(num_nodes, dtype=torch.bool)
            else:
                num_mask = max(1, int(num_nodes * self.mask_ratio))
                idx = torch.randperm(num_nodes)[:num_mask]
                mask = torch.zeros(num_nodes, dtype=torch.bool)
                mask[idx] = True
                data.coord_mask = mask
            
            # Attach the actual noise level used to the data object for the loss function
            data.noise_std_used = torch.tensor(current_noise_std, dtype=torch.float32)

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(noise_std={self.noise_std}, coordinate_dim={self.coordinate_dim}, mask_ratio={self.mask_ratio})'


class MaskAtomTypes(BaseTransform):
    """Mask atom types for masked language modeling task.

    Prefer masking `data.z` (atomic numbers, as in QM9). If `z` is absent, fall back to `x`.
    This transform also stores `original_types` and `masked_types` for use by the MLM loss.
    """
    
    def __init__(self, mask_ratio: float = 0.15, mask_token: int = 0):
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token
    
    def __call__(self, data: Data) -> Data:
        # Prefer atomic numbers `z` when available (QM9)
        if hasattr(data, 'z') and data.z is not None:
            # Ensure 1D long tensor of labels
            z = data.z.long()
            data.original_types = z.clone()
            num_nodes = z.size(0)
            if num_nodes > 0:
                num_mask = max(1, int(num_nodes * self.mask_ratio))
                mask_indices = torch.randperm(num_nodes)[:num_mask]
                mlm_mask = torch.zeros(num_nodes, dtype=torch.bool)
                mlm_mask[mask_indices] = True
                masked = z.clone()
                masked[mask_indices] = int(self.mask_token)
                data.masked_types = masked
                data.mlm_mask = mlm_mask
                # Write masked labels back to z so the encoder sees masked inputs
                data.z = masked
            return data
        
        # Fallback: mask first feature channel of x if z is not present
        if hasattr(data, 'x') and data.x is not None:
            # Treat first channel of x as categorical type ids
            x0 = data.x[:, 0].long()
            data.original_types = x0.clone()
            num_nodes = x0.size(0)
            if num_nodes > 0:
                num_mask = max(1, int(num_nodes * self.mask_ratio))
                mask_indices = torch.randperm(num_nodes)[:num_mask]
                mlm_mask = torch.zeros(num_nodes, dtype=torch.bool)
                mlm_mask[mask_indices] = True
                masked = x0.clone()
                masked[mask_indices] = int(self.mask_token)
                data.masked_types = masked
                data.mlm_mask = mlm_mask
                # Also write masked ids back into x[:,0]
                data.x = data.x.clone()
                data.x[mask_indices, 0] = float(self.mask_token)
        
        return data


class MaskBondTypes(BaseTransform):
    """Mask bond types for bond prediction task"""
    
    def __init__(self, mask_ratio: float = 0.15, mask_token: int = 0):
        self.mask_ratio = mask_ratio
        self.mask_token = mask_token
    
    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            # Store original bond types
            data.original_bond_types = data.edge_attr.clone()
            
            # Create mask for bond types
            num_edges = data.edge_attr.size(0)
            mask_indices = torch.randperm(num_edges)[:int(num_edges * self.mask_ratio)]
            
            # Create masked version
            data.masked_bond_types = data.edge_attr.clone()
            data.masked_bond_types[mask_indices] = self.mask_token
            
            # Replace original with masked version
            data.edge_attr = data.masked_bond_types
        
        return data


class AddDistanceFeatures(BaseTransform):
    """Add distance features for distance prediction task"""
    
    def __init__(self, max_distance: float = 10.0, distance_bins: int = 64):
        self.max_distance = max_distance
        self.distance_bins = distance_bins
    
    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'pos') and data.pos is not None and hasattr(data, 'edge_index'):
            # Compute distances between connected nodes
            source_pos = data.pos[data.edge_index[0]]
            target_pos = data.pos[data.edge_index[1]]
            distances = torch.norm(source_pos - target_pos, dim=-1)
            
            # Store distances
            data.distances = distances
            
            # Create distance bins
            distance_bins = torch.clamp(
                (distances / self.max_distance * self.distance_bins).long(),
                0, self.distance_bins - 1
            )
            data.distance_bins = distance_bins
        
        return data


class AddAngleFeatures(BaseTransform):
    """Add angle features for angle prediction task"""
    
    def __init__(self, angle_bins: int = 36):
        self.angle_bins = angle_bins
    
    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'pos') and data.pos is not None and hasattr(data, 'edge_index'):
            # Find triplets of connected nodes
            angle_indices, angles = self._compute_angles(data)
            
            if angle_indices is not None:
                data.angle_indices = angle_indices
                data.angles = angles
                
                # Create angle bins
                angle_bins = torch.clamp(
                    (angles / (2 * np.pi) * self.angle_bins).long(),
                    0, self.angle_bins - 1
                )
                data.angle_bins = angle_bins
        
        return data
    
    def _compute_angles(self, data: Data):
        """Compute angles between triplets of connected nodes"""
        edge_index = data.edge_index
        pos = data.pos
        
        # Create adjacency list
        adj_list = {}
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src not in adj_list:
                adj_list[src] = []
            if dst not in adj_list:
                adj_list[dst] = []
            adj_list[src].append(dst)
            adj_list[dst].append(src)
        
        # Find triplets
        triplets = []
        angles = []
        
        for center_node in adj_list:
            neighbors = adj_list[center_node]
            if len(neighbors) >= 2:
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        node1, node2 = neighbors[i], neighbors[j]
                        
                        # Compute angle
                        vec1 = pos[node1] - pos[center_node]
                        vec2 = pos[node2] - pos[center_node]
                        
                        # Normalize vectors
                        vec1_norm = vec1 / (torch.norm(vec1) + 1e-8)
                        vec2_norm = vec2 / (torch.norm(vec2) + 1e-8)
                        
                        # Compute angle
                        cos_angle = torch.clamp(torch.dot(vec1_norm, vec2_norm), -1.0, 1.0)
                        angle = torch.acos(cos_angle)
                        
                        triplets.append([node1, center_node, node2])
                        angles.append(angle)
        
        if triplets:
            return torch.tensor(triplets, dtype=torch.long), torch.tensor(angles, dtype=torch.float)
        else:
            return None, None


class AddGraphProperties(BaseTransform):
    """Add graph properties for graph property prediction task"""
    
    def __init__(self, property_types: List[str] = None):
        self.property_types = property_types or ["molecular_weight", "logp", "tpsa"]
    
    def __call__(self, data: Data) -> Data:
        # Compute graph properties based on available features
        properties = []
        
        for prop_type in self.property_types:
            if prop_type == "molecular_weight":
                # Simple molecular weight approximation based on atom types
                if hasattr(data, 'x') and data.x is not None:
                    # Assuming x contains atom types, compute approximate molecular weight
                    atom_weights = torch.ones_like(data.x[:, 0]) * 12.0  # Default to carbon weight
                    mol_weight = torch.sum(atom_weights)
                    properties.append(mol_weight)
                else:
                    properties.append(0.0)
            
            elif prop_type == "logp":
                # Simple logP approximation
                if hasattr(data, 'x') and data.x is not None:
                    # Very simplified logP calculation
                    logp = torch.sum(data.x[:, 0]) * 0.1  # Simplified
                    properties.append(logp)
                else:
                    properties.append(0.0)
            
            elif prop_type == "tpsa":
                # Topological polar surface area approximation
                if hasattr(data, 'x') and data.x is not None:
                    # Simplified TPSA calculation
                    tpsa = torch.sum(data.x[:, 0]) * 0.5  # Simplified
                    properties.append(tpsa)
                else:
                    properties.append(0.0)
            
            else:
                # Default property
                properties.append(0.0)
        
        data.graph_properties = torch.tensor(properties, dtype=torch.float)
        return data


class CreateMaskedPairs(BaseTransform):
    """Create masked pairs for contrastive learning"""
    
    def __init__(self, mask_ratio: float = 0.15):
        self.mask_ratio = mask_ratio
    
    def __call__(self, data: Data) -> Data:
        # Create two different masked versions of the same graph
        if hasattr(data, 'x') and data.x is not None:
            # First masked version
            data.x_masked_1 = self._create_masked_version(data.x)
            
            # Second masked version
            data.x_masked_2 = self._create_masked_version(data.x)
        
        return data
    
    def _create_masked_version(self, x: torch.Tensor) -> torch.Tensor:
        """Create a masked version of node features"""
        masked_x = x.clone()
        num_nodes = x.size(0)
        mask_indices = torch.randperm(num_nodes)[:int(num_nodes * self.mask_ratio)]
        masked_x[mask_indices] = 0  # Mask token
        return masked_x


class AddAtomTypeLabels(BaseTransform):
    """Add atom type labels for atom type prediction task"""
    
    def __init__(self):
        pass
    
    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'x') and data.x is not None:
            # Store atom types as labels
            data.atom_types = data.x[:, 0].long()  # Assuming first column contains atom types
        
        return data


class AddBondTypeLabels(BaseTransform):
    """Add bond type labels for bond prediction task"""
    
    def __init__(self):
        pass
    
    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            # Store bond types as labels
            data.bond_types = data.edge_attr[:, 0].long()  # Assuming first column contains bond types
        
        return data


class AddCoordinateLabels(BaseTransform):
    """Add coordinate labels for coordinate denoising task"""
    
    def __init__(self):
        pass
    
    def __call__(self, data: Data) -> Data:
        if hasattr(data, 'pos') and data.pos is not None:
            # Store clean coordinates as labels
            data.coordinate_labels = data.pos.clone()
        
        return data


class MultiTaskTransform(BaseTransform):
    """Combine multiple transforms for multi-task pretraining"""
    
    def __init__(self, transforms: List[BaseTransform]):
        self.transforms = transforms
    
    def __call__(self, data: Data) -> Data:
        for transform in self.transforms:
            data = transform(data)
        return data


class RandomAugmentation(BaseTransform):
    """Random augmentation for contrastive learning"""
    
    def __init__(self, augmentation_prob: float = 0.5):
        self.augmentation_prob = augmentation_prob
    
    def __call__(self, data: Data) -> Data:
        if random.random() < self.augmentation_prob:
            # Random node feature masking
            if hasattr(data, 'x') and data.x is not None:
                mask_ratio = random.uniform(0.1, 0.3)
                num_nodes = data.x.size(0)
                mask_indices = torch.randperm(num_nodes)[:int(num_nodes * mask_ratio)]
                data.x[mask_indices] = 0
            
            # Random edge dropping
            if hasattr(data, 'edge_index') and data.edge_index is not None:
                drop_ratio = random.uniform(0.1, 0.3)
                num_edges = data.edge_index.size(1)
                keep_indices = torch.randperm(num_edges)[:int(num_edges * (1 - drop_ratio))]
                data.edge_index = data.edge_index[:, keep_indices]
                
                if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                    data.edge_attr = data.edge_attr[keep_indices]
        
        return data


class DomainSpecificTransform(BaseTransform):
    """Domain-specific transforms for different molecular domains"""
    
    def __init__(self, domain_type: str):
        self.domain_type = domain_type
    
    def __call__(self, data: Data) -> Data:
        if self.domain_type == "molecule":
            # Molecule-specific transforms
            if hasattr(data, 'x') and data.x is not None:
                # Add molecule-specific features
                data.molecule_features = data.x.clone()
        
        elif self.domain_type == "protein":
            # Protein-specific transforms
            if hasattr(data, 'x') and data.x is not None:
                # Add protein-specific features
                data.protein_features = data.x.clone()
        
        elif self.domain_type == "rna":
            # RNA-specific transforms
            if hasattr(data, 'x') and data.x is not None:
                # Add RNA-specific features
                data.rna_features = data.x.clone()
        
        elif self.domain_type == "metabolite":
            # Metabolite-specific transforms
            if hasattr(data, 'x') and data.x is not None:
                # Add metabolite-specific features
                data.metabolite_features = data.x.clone()
        
        return data


def create_pretraining_transforms(config: Dict[str, Any]) -> List[BaseTransform]:
    """Create a list of transforms based on configuration"""
    transforms = []
    
    # Add domain-specific transform
    domain_type = config.get('domain_type', 'molecule')
    transforms.append(DomainSpecificTransform(domain_type))
    
    # Add coordinate denoising transform
    if config.get('use_coordinate_denoising', False):
        transforms.append(AddRandomNoise(
            noise_std=config.get('coordinate_noise_std', 0.1),
            coordinate_dim=config.get('coordinate_dim', 3)
        ))
        transforms.append(AddCoordinateLabels())
    
    # Add atom type prediction transform
    if config.get('use_atom_type_prediction', False):
        transforms.append(MaskAtomTypes(
            mask_ratio=config.get('atom_type_mask_ratio', 0.15),
            mask_token=config.get('mask_token', 0)
        ))
        transforms.append(AddAtomTypeLabels())
    
    # Add bond prediction transform
    if config.get('use_bond_prediction', False):
        transforms.append(MaskBondTypes(
            mask_ratio=config.get('bond_mask_ratio', 0.15),
            mask_token=config.get('mask_token', 0)
        ))
        transforms.append(AddBondTypeLabels())
    
    # Add distance prediction transform
    if config.get('use_distance_prediction', False):
        transforms.append(AddDistanceFeatures(
            max_distance=config.get('max_distance', 10.0),
            distance_bins=config.get('distance_bins', 64)
        ))
    
    # Add angle prediction transform
    if config.get('use_angle_prediction', False):
        transforms.append(AddAngleFeatures(
            angle_bins=config.get('angle_bins', 36)
        ))
    
    # Add graph property prediction transform
    if config.get('use_graph_property_prediction', False):
        transforms.append(AddGraphProperties(
            property_types=config.get('graph_property_types', ["molecular_weight", "logp", "tpsa"])
        ))
    
    # Add contrastive learning transforms
    if config.get('use_contrastive_learning', False):
        transforms.append(CreateMaskedPairs(
            mask_ratio=config.get('mlm_mask_ratio', 0.15)
        ))
        transforms.append(RandomAugmentation(
            augmentation_prob=config.get('augmentation_prob', 0.5)
        ))
    
    return transforms
