import sys
import os
import warnings
import argparse
import copy
import torch
import wandb
import numpy as np
import pytorch_lightning as pl
import json
import contextlib
from pathlib import Path

from torch_geometric.seed import seed_everything
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# Imports from this project
sys.path.append(os.path.realpath("."))

from esa.pretraining_model import PretrainingESAModel, create_pretraining_config, PretrainingConfig
from data.efficient_molecular_datasets import ESA3DQM9Dataset
from data_loading.data_loading import get_dataset_train_val_test
from esa.config import (
    save_arguments_to_json,
    load_arguments_from_json,
    validate_argparse_arguments,
    get_wandb_name,
)

warnings.filterwarnings("ignore")

os.environ["WANDB__SERVICE_WAIT"] = "500"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Ensure a viable SDP backend even if flash/mem-efficient are unavailable
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass


def create_pretraining_data_transforms(config: PretrainingConfig):
    """Create data transforms for pretraining tasks"""
    # Use the dedicated pretraining transforms module
    from data_loading.pretraining_transforms import (
        MaskAtomTypes,
    )
    
    transforms = []
    

    
    # Atom type masking for MLM
    if "mlm" in config.pretraining_tasks:
        transforms.append(MaskAtomTypes(
            mask_ratio=config.mlm_mask_ratio,
            mask_token=0  # Assuming 0 is the mask token
        ))
    
    return transforms


def load_multi_domain_dataset(config: PretrainingConfig, dataset_name: str, dataset_dir: str):
    """Load dataset for multi-domain pretraining"""
    
    # Create transforms for pretraining tasks
    transforms = create_pretraining_data_transforms(config)
    
    # Load dataset based on domain type
    if dataset_name in ["QM9", "DOCKSTRING", "ESOL", "FreeSolv", "Lipo", "HIV", "BACE", "BBBP"]:
        # Small molecule datasets
        domain_type = "molecule"
        if dataset_name == "QM9":
            # Prefer ESA3DQM9Dataset with 3D coordinates for pretraining
            dataset_root = os.path.join(dataset_dir, 'qm9_dataset')
            
            # Handle cache control for QM9
            if not getattr(config, 'use_dataset_cache', True):
                # Clear cached QM9 processed files to force re-processing
                processed_dir = os.path.join(dataset_root, 'processed')
                if os.path.exists(processed_dir):
                    import glob
                    cache_files = glob.glob(os.path.join(processed_dir, '*.pt'))
                    for cache_file in cache_files:
                        try:
                            os.remove(cache_file)
                            print(f"Removed cached QM9 file: {os.path.basename(cache_file)}")
                        except Exception as e:
                            print(f"Could not remove {cache_file}: {e}")
            
            # Apply pretraining transforms so required fields (clean_pos, masked types, distances, etc.) exist
            # ESA3DQM9Dataset caches processed tensors; if present, it will be reused
            ds = ESA3DQM9Dataset(root=dataset_root, transform=Compose(transforms) if len(transforms) > 0 else None)
            # 80/10/10 split
            total = len(ds)
            idx_train = int(0.8 * total)
            idx_val = int(0.9 * total)
            train = ds[:idx_train]
            val = ds[idx_train:idx_val]
            test = ds[idx_val:]
            num_classes, task_type, scaler = 0, 'pretraining', None
        else:
            # Fallback to generic loader for other molecule datasets
            loader_kwargs = {"transforms": transforms}
            train, val, test, num_classes, task_type, scaler = get_dataset_train_val_test(
                dataset=dataset_name,
                dataset_dir=dataset_dir,
                **loader_kwargs,
            )
        
    elif dataset_name in ["PDB", "AFDB", "protein_dataset"]:
        # Protein datasets via ESA3DPDBDataset (mmCIF/PDB files)
        domain_type = "protein"
        from data.efficient_molecular_datasets import ESA3DPDBDataset
        dataset_root = os.path.join(dataset_dir, 'pdb_dataset' if dataset_name == "PDB" else 'afdb_dataset')
        
        # Handle cache control
        if not getattr(config, 'use_dataset_cache', True):
            # Clear cached processed files to force re-processing
            processed_dir = os.path.join(dataset_root, 'processed')
            if os.path.exists(processed_dir):
                import glob
                cache_files = glob.glob(os.path.join(processed_dir, 'esa3d_pdb_max_atoms_*.pt'))
                for cache_file in cache_files:
                    try:
                        os.remove(cache_file)
                        print(f"🗑️  Removed cached file: {os.path.basename(cache_file)}")
                    except Exception as e:
                        print(f"⚠️  Could not remove {cache_file}: {e}")
        
        # Allow limiting atom count via config if provided
        max_atoms = getattr(config, 'protein_max_residues', 2000)
        ds = ESA3DPDBDataset(
            root=dataset_root,
            max_atoms=getattr(config, 'protein_max_residues', 2000),
            cutoff_distance=getattr(config, 'cutoff_distance', 6.0),
            max_neighbors=getattr(config, 'max_neighbors', 64),
            transform=Compose(transforms) if len(transforms) > 0 else None,
            pre_transform=None,
            pre_filter=None,
        )
        total = len(ds)
        idx_train = int(0.8 * total)
        idx_val = int(0.9 * total)
        train = ds[:idx_train]
        val = ds[idx_train:idx_val]
        test = ds[idx_val:]
        num_classes, task_type, scaler = 0, 'pretraining', None
        
    elif dataset_name in ["rna_dataset"]:  # Replace with actual RNA dataset names
        # RNA datasets
        domain_type = "rna"
        train, val, test, num_classes, task_type, scaler = get_dataset_train_val_test(
            dataset=dataset_name,
            dataset_dir=dataset_dir,
            transforms=transforms
        )
        
    elif dataset_name in ["metabolite_dataset"]:  # Replace with actual metabolite dataset names
        # Metabolite datasets
        domain_type = "metabolite"
        train, val, test, num_classes, task_type, scaler = get_dataset_train_val_test(
            dataset=dataset_name,
            dataset_dir=dataset_dir,
            transforms=transforms
        )
        
    else:
        # Default to molecule domain
        domain_type = "molecule"
        train, val, test, num_classes, task_type, scaler = get_dataset_train_val_test(
            dataset=dataset_name,
            dataset_dir=dataset_dir,
            transforms=transforms
        )
    
    return train, val, test, domain_type, scaler


def create_data_loaders(train, val, test, config: PretrainingConfig):
    """Create data loaders for pretraining"""
    
    def collate_fn(batch):
        """Custom collate function for pretraining data"""
        # This will be handled by PyTorch Geometric's default collate
        return batch
    
    train_loader = GeometricDataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=getattr(config, 'num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = GeometricDataLoader(
        val,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=getattr(config, 'num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = GeometricDataLoader(
        test,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=getattr(config, 'num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def main():
    torch.set_num_threads(1)
    
    parser = argparse.ArgumentParser(description="Pretraining ESA Model for Multi-Domain Geometric Deep Learning")
    
    # Basic arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset", type=str, required=False, default=None, help="Dataset name")
    parser.add_argument("--dataset-download-dir", type=str, required=False, default=None, help="Dataset download directory")
    parser.add_argument("--out-path", type=str, required=False, default=None, help="Output directory")
    parser.add_argument("--config-json-path", type=str, help="Path to JSON config file")
    parser.add_argument("--config-yaml-path", type=str, help="Path to YAML config file")
    parser.add_argument("--wandb-project-name", type=str, default="esa-pretraining", help="WandB project name")
    parser.add_argument("--use-dataset-cache", action="store_true", default=True, help="Use cached processed datasets")
    parser.add_argument("--no-dataset-cache", dest="use_dataset_cache", action="store_false", help="Force re-processing, ignore cached datasets")
    
    # Model architecture arguments
    parser.add_argument("--num-features", type=int, default=128, help="Number of input features")
    parser.add_argument("--graph-dim", type=int, default=256, help="Graph embedding dimension")
    parser.add_argument("--edge-dim", type=int, default=64, help="Edge feature dimension")
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256, 256, 256], help="Hidden dimensions")
    parser.add_argument("--num-heads", type=int, nargs="+", default=[8, 8, 8, 8], help="Number of attention heads")
    parser.add_argument("--layer-types", type=str, nargs="+", default=["S", "S", "S", "P"], help="Layer types")
    parser.add_argument("--apply-attention-on", type=str, default="edge", choices=["node", "edge"], help="Attention type")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--early-stopping-patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--gradient-clip-val", type=float, default=0.5, help="Gradient clipping value")
    parser.add_argument("--optimiser-weight-decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum number of epochs")
    
    # Pretraining task arguments
    parser.add_argument("--pretraining-tasks", type=str, nargs="+", 
                       default=["long_range_distance", "short_range_distance", "mlm"],
                       help="Pretraining tasks to use")
    parser.add_argument("--task-weights", type=float, nargs="+", 
                       default=[1.0, 1.0, 1.0], help="Weights for pretraining tasks")
    
    # Domain-specific arguments
    parser.add_argument("--atom-types", type=int, default=119, help="Number of atom types")
    parser.add_argument("--amino-acid-types", type=int, default=20, help="Number of amino acid types")
    parser.add_argument("--nucleotide-types", type=int, default=4, help="Number of nucleotide types")
    parser.add_argument("--metabolite-atom-types", type=int, default=50, help="Number of metabolite atom types")
    
    # 3D geometric arguments
    parser.add_argument("--use-3d-coordinates", action="store_true", help="Use 3D coordinates")
    parser.add_argument("--gaussian-kernels", type=int, default=128, help="Number of Gaussian kernels")
    parser.add_argument("--cutoff-distance", type=float, default=5.0, help="Cutoff distance for neighbors")
    parser.add_argument("--max-neighbors", type=int, default=32, help="Maximum number of neighbors")
    
    # Task-specific arguments
    parser.add_argument("--distance-bins", type=int, default=64, help="Number of distance bins")
    parser.add_argument("--max-distance", type=float, default=10.0, help="Maximum distance for binning")
    parser.add_argument("--mlm-mask-ratio", type=float, default=0.15, help="MLM masking ratio")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for softmax")
    
    # ESA-specific arguments
    parser.add_argument("--xformers-or-torch-attn", type=str, default="xformers", choices=["xformers", "torch"], help="Attention implementation")
    parser.add_argument("--sab-dropout", type=float, default=0.0, help="SAB dropout")
    parser.add_argument("--mab-dropout", type=float, default=0.0, help="MAB dropout")
    parser.add_argument("--pma-dropout", type=float, default=0.0, help="PMA dropout")
    parser.add_argument("--attn-residual-dropout", type=float, default=0.0, help="Attention residual dropout")
    parser.add_argument("--pma-residual-dropout", type=float, default=0.0, help="PMA residual dropout")
    parser.add_argument("--norm-type", type=str, default="LN", choices=["BN", "LN"], help="Normalization type")
    parser.add_argument("--use-mlps", action="store_true", help="Use MLPs")
    parser.add_argument("--mlp-hidden-size", type=int, default=64, help="MLP hidden size")
    parser.add_argument("--mlp-layers", type=int, default=3, help="Number of MLP layers")
    parser.add_argument("--mlp-type", type=str, default="standard", choices=["standard", "gated_mlp"], help="MLP type")
    parser.add_argument("--mlp-dropout", type=float, default=0.0, help="MLP dropout")
    parser.add_argument("--use-mlp-ln", action="store_true", help="Use layer normalization in MLPs")
    parser.add_argument("--pre-or-post", type=str, default="pre", choices=["pre", "post"], help="Pre or post normalization")
    parser.add_argument("--use-bfloat16", action="store_true", help="Use bfloat16")
    parser.add_argument("--pos-enc", type=str, help="Position encoding type")
    
    args = parser.parse_args()
    
    # Load config from YAML or JSON if provided
    if args.config_yaml_path:
        import yaml
        with open(args.config_yaml_path, 'r') as f:
            y = yaml.safe_load(f)
        if not isinstance(y, dict):
            y = {}
        # Preserve nested dicts like task_weights instead of flattening
        config = create_pretraining_config(**y)
        # Reflect only non-dict top-level entries into argparse args
        for k, v in y.items():
            if not isinstance(v, dict):
                try:
                    setattr(args, k, v)
                except Exception:
                    pass
    elif args.config_json_path:
        with open(args.config_json_path, 'r') as f:
            config_dict = json.load(f)
        config = create_pretraining_config(**config_dict)
        # Populate missing CLI args from config file
        for k, v in config_dict.items():
            try:
                setattr(args, k, v)
            except Exception:
                pass
    else:
        # Create config from command line arguments
        task_weights_dict = {}
        for i, task in enumerate(args.pretraining_tasks):
            if i < len(args.task_weights):
                task_weights_dict[task] = args.task_weights[i]
            else:
                task_weights_dict[task] = 1.0
        
        config = create_pretraining_config(
            # Basic model config
            num_features=args.num_features,
            graph_dim=args.graph_dim,
            edge_dim=args.edge_dim,
            batch_size=args.batch_size,
            lr=args.lr,
            early_stopping_patience=args.early_stopping_patience,
            optimiser_weight_decay=args.optimiser_weight_decay,
            
            # Architecture
            hidden_dims=args.hidden_dims,
            num_heads=args.num_heads,
            layer_types=args.layer_types,
            apply_attention_on=args.apply_attention_on,
            
            # ESA specific
            xformers_or_torch_attn=args.xformers_or_torch_attn,
            sab_dropout=args.sab_dropout,
            mab_dropout=args.mab_dropout,
            pma_dropout=args.pma_dropout,
            attn_residual_dropout=args.attn_residual_dropout,
            pma_residual_dropout=args.pma_residual_dropout,
            norm_type=args.norm_type,
            use_mlps=args.use_mlps,
            mlp_hidden_size=args.mlp_hidden_size,
            num_mlp_layers=args.mlp_layers,
            mlp_type=args.mlp_type,
            mlp_dropout=args.mlp_dropout,
            use_mlp_ln=args.use_mlp_ln,
            pre_or_post=args.pre_or_post,
            use_bfloat16=args.use_bfloat16,
            posenc=args.pos_enc,
            
            # Domain specific
            atom_types=args.atom_types,
            amino_acid_types=args.amino_acid_types,
            nucleotide_types=args.nucleotide_types,
            metabolite_atom_types=args.metabolite_atom_types,
            
            # 3D geometric
            use_3d_coordinates=args.use_3d_coordinates,
            gaussian_kernels=args.gaussian_kernels,
            cutoff_distance=args.cutoff_distance,
            max_neighbors=args.max_neighbors,
            
            # Pretraining tasks
            pretraining_tasks=args.pretraining_tasks,
            task_weights=task_weights_dict,
            
            # Task specific
            distance_bins=args.distance_bins,
            max_distance=args.max_distance,
            mlm_mask_ratio=args.mlm_mask_ratio,
            temperature=args.temperature,
            
            # Dataset control
            use_dataset_cache=args.use_dataset_cache,
        )
    
    # Set seed
    seed_everything(args.seed)
    
    # Create output directory
    # Prefer YAML/JSON fields if present in config
    out_path = Path(getattr(args, 'out_path', getattr(config, 'out_path', './outputs')))
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_dict = vars(config)
    try:
        merged = {**config_dict, **vars(args)}
    except Exception:
        merged = config_dict
    save_arguments_to_json(merged, str(out_path))
    
    # Load dataset
    dataset_name = getattr(args, 'dataset', getattr(config, 'dataset', 'QM9'))
    print(f"Loading dataset: {dataset_name}")
    train, val, test, domain_type, scaler = load_multi_domain_dataset(
        config, dataset_name, getattr(args, 'dataset_download_dir', getattr(config, 'dataset_download_dir', './data'))
    )
    
    print(f"Dataset loaded. Domain type: {domain_type}")
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    has_validation = len(val) > 0
    # Ensure LR scheduler inside the model monitors the correct metric
    if not has_validation:
        try:
            config.monitor_loss_name = "train_total_loss"
        except Exception:
            pass
    
    # Compute a safe global max set size depending on attention type
    if config.apply_attention_on == "edge" and dataset_name == "QM9":
        try:
            # Fast path: compute edge counts from cached slices without triggering transforms
            def max_edges_from_slices(ds):
                if hasattr(ds, 'slices') and isinstance(ds.slices, dict) and 'edge_index' in ds.slices:
                    ptr = ds.slices['edge_index']
                    # ptr is a 1D LongTensor of length num_graphs+1 over concatenated dim-1
                    # Number of edges per graph = ptr[i+1] - ptr[i]
                    # edge_index shape is [2, E], so slices are along dim=1; counts computed on that axis
                    counts = (ptr[1:] - ptr[:-1]).tolist()
                    return max(counts) if counts else 1
                # Fallback to slow iteration only if needed
                m = 1
                for d in ds:
                    if hasattr(d, 'edge_index') and d.edge_index is not None:
                        m = max(m, d.edge_index.size(1))
                return m

            max_e = max(max_edges_from_slices(train), max_edges_from_slices(val), max_edges_from_slices(test))
            from esa.pretraining_model import nearest_multiple_of_8
            config.set_max_items = int(nearest_multiple_of_8(max_e))
            print(f"Using global set_max_items={config.set_max_items} (max edges per graph={max_e})")
        except Exception as e:
            print(f"Warning: could not compute max edges, keeping set_max_items={config.set_max_items}: {e}")

    # For node attention (e.g., proteins), set_max_items must cap the number of NODES per graph
    if config.apply_attention_on == "node":
        try:
            def max_nodes_from_slices(ds):
                if hasattr(ds, 'slices') and isinstance(ds.slices, dict):
                    for key in ['pos', 'x', 'z']:
                        if key in ds.slices:
                            ptr = ds.slices[key]
                            counts = (ptr[1:] - ptr[:-1]).tolist()
                            if counts:
                                return max(counts)
                # Fallback slow path
                m = 1
                for d in ds:
                    if hasattr(d, 'num_nodes') and d.num_nodes is not None:
                        m = max(m, int(d.num_nodes))
                    elif hasattr(d, 'pos') and d.pos is not None:
                        m = max(m, d.pos.size(0))
                return m

            max_n = max(max_nodes_from_slices(train), max_nodes_from_slices(val), max_nodes_from_slices(test))
            from esa.pretraining_model import nearest_multiple_of_8
            config.set_max_items = int(nearest_multiple_of_8(max_n))
            print(f"Using global set_max_items={config.set_max_items} (max nodes per graph={max_n})")
        except Exception as e:
            print(f"Warning: could not compute max nodes, keeping set_max_items={config.set_max_items}: {e}")

    # Subset for quick debug if requested via JSON, else default to 2000 for QM9
    subset_n = getattr(config, 'debug_subset_n', None) if dataset_name == "QM9" else None
    if subset_n is not None:
        print(f"Using debug subset: first {subset_n} graphs")
        train = train[:subset_n]
        val = val[:max(1, subset_n//10)]
        test = test[:max(1, subset_n//10)]

    # Respect config.debug_verbose; do not force enable

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(train, val, test, config)
    
    # Create model
    print("Creating pretraining model...")
    model = PretrainingESAModel(config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Pretraining tasks: {config.pretraining_tasks}")
    print(f"Task weights: {config.task_weights}")
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpointing
    monitor_metric = "val_total_loss" if has_validation else "train_total_loss"
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(out_path / "checkpoints"),
        filename=f"pretraining-{{epoch:02d}}-{{{monitor_metric}:.4f}}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric,
        mode="min",
        patience=args.early_stopping_patience,
        verbose=True,
    )
    callbacks.append(early_stopping_callback)
    
    # Setup logger
    logger = None
    if getattr(args, 'wandb_project_name', getattr(config, 'wandb_project_name', None)):
        logger = WandbLogger(
            project=getattr(args, 'wandb_project_name', getattr(config, 'wandb_project_name', 'esa-pretraining')),
            name=getattr(args, 'wandb_run_name', getattr(config, 'wandb_run_name', f"pretraining-{dataset_name}-{domain_type}")),
            log_model=True,
        )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=getattr(args, 'max_epochs', getattr(config, 'max_epochs', 100)),
        accelerator="auto",
        devices="auto",
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=getattr(args, 'gradient_clip_val', getattr(config, 'gradient_clip_val', 0.5)),
        precision="16-mixed" if getattr(args, 'use_bfloat16', getattr(config, 'use_bfloat16', False)) else "32",
        deterministic=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
        num_sanity_val_steps=0 if not has_validation else 2,
    )
    
    # Train model
    print("Starting training...")
    if has_validation:
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
    else:
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
        )
    
    # Test model
    print("Testing model...")
    trainer.test(
        model=model,
        dataloaders=test_loader,
        ckpt_path="best",
    )
    
    # Save final model
    final_model_path = out_path / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save embeddings for downstream tasks
    print("Computing embeddings for downstream tasks...")
    model.eval()
    
    # Get embeddings for a sample batch (ensure same device and viable SDPA backend)
    sample_batch = next(iter(test_loader))
    model_device = next(model.parameters()).device
    if model_device.type == "cpu" and torch.cuda.is_available():
        model = model.to("cuda")
        model_device = torch.device("cuda")
    # PyG batch has .to
    try:
        sample_batch = sample_batch.to(model_device)
    except Exception:
        pass
    sdp_ctx = (
        torch.backends.cuda.sdp_kernel(enable_math=True, enable_mem_efficient=False, enable_flash=False)
        if (hasattr(torch.backends, "cuda") and model_device.type == "cuda") else contextlib.nullcontext()
    )
    with sdp_ctx:
        with torch.no_grad():
            graph_embeddings, node_embeddings = model.get_embeddings(sample_batch, domain_type)
    
    print(f"Graph embeddings shape: {graph_embeddings.shape}")
    print(f"Node embeddings shape: {node_embeddings.shape}")
    
    # Save embedding info
    embedding_info = {
        "graph_embedding_dim": graph_embeddings.shape[-1],
        "node_embedding_dim": node_embeddings.shape[-1],
        "domain_type": domain_type,
        "pretraining_tasks": config.pretraining_tasks,
        "model_config": config_dict,
    }
    
    with open(out_path / "embedding_info.json", 'w') as f:
        json.dump(embedding_info, f, indent=2)
    
    print("Training completed successfully!")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
