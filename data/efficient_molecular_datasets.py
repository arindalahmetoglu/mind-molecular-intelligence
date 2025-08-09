#!/usr/bin/env python3
"""
Efficient Molecular Datasets for ESA3D using InMemoryDataset

This module provides efficient dataset classes for ESA3D that use PyTorch Geometric's
InMemoryDataset to avoid memory bloat and provide fast data loading.

Key Features:
- Uses InMemoryDataset for efficient storage and loading
- Processes raw files only once
- Saves data in optimized format
- Dramatically reduces file sizes compared to naive approaches
"""

import os
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
 
# Neighborhood graph constructor (radius graph)
try:
    from torch_cluster import radius_graph  # preferred
except Exception:
    try:
        from torch_geometric.nn.pool import radius_graph  # fallback location in some versions
    except Exception:
        radius_graph = None

# For PDB processing
try:
    from Bio.PDB import PDBParser, MMCIFParser
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("Warning: BioPython not available. PDB processing will not work.")

# For QM9 processing
try:
    from torch_geometric.datasets import QM9
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("Warning: PyTorch Geometric not available. QM9 processing will not work.")


class ESA3DQM9Dataset(InMemoryDataset):
    """
    Efficient QM9 dataset for ESA3D using InMemoryDataset.
    
    This class downloads QM9 via PyTorch Geometric and converts it to ESA3D format
    in a memory-efficient way.
    """
    
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.qm9_properties = [
            'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 
            'u0', 'u298', 'h298', 'g298', 'cv'
        ]
        super().__init__(root, transform, pre_transform, pre_filter)
        # Robust load that works with PyTorch >=2.6 (weights_only default) and older versions
        processed_path = self.processed_paths[0]
        try:
            self.data, self.slices = torch.load(processed_path, map_location='cpu', weights_only=False)
        except TypeError:
            # Older torch without weights_only
            self.data, self.slices = torch.load(processed_path, map_location='cpu')

    @property
    def raw_file_names(self):
        # Match the filename we save in download()
        return ['qm9_pyg_data.pt']

    @property
    def processed_file_names(self):
        return ['esa3d_qm9.pt']

    def download(self):
        """Download QM9 dataset using PyTorch Geometric."""
        if not PYG_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for QM9 download")

        raw_pt = os.path.join(self.raw_dir, 'qm9_pyg_data.pt')
        if os.path.exists(raw_pt):
            print("✅ Using cached QM9 raw data")
            return

        print("📥 Downloading QM9 dataset via PyTorch Geometric...")
        qm9_dataset = QM9(root=os.path.join(self.raw_dir, 'qm9_pyg'))
        print(f"✅ QM9 downloaded: {len(qm9_dataset)} molecules")

        # Save raw data for processing
        os.makedirs(self.raw_dir, exist_ok=True)
        torch.save(qm9_dataset, raw_pt)

    def process(self):
        """Process QM9 data into ESA3D format efficiently."""
        # If processed exists, reuse cache
        if os.path.exists(self.processed_paths[0]):
            print(f"✅ Using cached processed dataset at: {self.processed_paths[0]}")
            return

        print("🔧 Processing QM9 data for ESA3D...")
        
        # Load raw QM9 data
        qm9_data = torch.load(os.path.join(self.raw_dir, 'qm9_pyg_data.pt'))
        
        data_list = []
        
        for i, mol_data in enumerate(tqdm(qm9_data, desc="Converting QM9")):
            try:
                # Extract 3D coordinates and atomic numbers
                positions = mol_data.pos  # Already torch.tensor
                atomic_numbers = mol_data.z  # Already torch.tensor
                
                # Extract edge_index if available
                edge_index = mol_data.edge_index if hasattr(mol_data, 'edge_index') else None
                edge_attr = mol_data.edge_attr if hasattr(mol_data, 'edge_attr') else None
                
                # Extract properties
                properties = {}
                for prop_name in self.qm9_properties:
                    if hasattr(mol_data, prop_name):
                        prop_value = getattr(mol_data, prop_name)
                        properties[prop_name] = float(prop_value) if prop_value is not None else 0.0
                    else:
                        properties[prop_name] = 0.0
                
                # Create ESA3D-compatible Data object
                data = Data(
                    pos=positions,
                    z=atomic_numbers,
                    properties=properties,
                    mol_id=f'qm9_{i:06d}',
                    num_atoms=len(atomic_numbers)
                )
                
                # Add edge information if available
                if edge_index is not None:
                    data.edge_index = edge_index
                if edge_attr is not None:
                    data.edge_attr = edge_attr
                
                data_list.append(data)
                
            except Exception as e:
                print(f"Error processing molecule {i}: {e}")
                continue
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        # Efficiently collate and save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        print(f"✅ Processed {len(data_list)} molecules")
        print(f"💾 Saved to: {self.processed_paths[0]}")


class ESA3DPDBDataset(InMemoryDataset):
    """
    Efficient PDB protein dataset for ESA3D using InMemoryDataset.
    
    This class processes PDB/mmCIF files and saves them in an optimized format.
    """
    
    def __init__(self, root, max_atoms=2000, cutoff_distance: float = 6.0, max_neighbors: int = 64,
                 transform=None, pre_transform=None, pre_filter=None):
        self.max_atoms = max_atoms
        self.cutoff_distance = float(cutoff_distance)
        self.max_neighbors = int(max_neighbors)
        self.atom_mapping = {
            'C': 1, 'N': 2, 'O': 3, 'S': 4, 'H': 5, 'P': 6, 
            'F': 7, 'CL': 8, 'BR': 9, 'I': 10, 'FE': 11, 'ZN': 12, 
            'CA': 13, 'MG': 14, 'MN': 15, 'CU': 16, 'CO': 17, 
            'NI': 18, 'SE': 19
        }
        super().__init__(root, transform, pre_transform, pre_filter)
        # Robust load that works across torch versions
        processed_path = self.processed_paths[0]
        try:
            self.data, self.slices = torch.load(processed_path, map_location='cpu', weights_only=False)
        except TypeError:
            self.data, self.slices = torch.load(processed_path, map_location='cpu')
        # After initial load, ensure cache matches current raw and config; if not, rebuild
        try:
            if not self._cache_is_fresh():
                print("⚠️  Cache metadata mismatch or raw changed. Rebuilding PDB processed cache...")
                self.process()
                try:
                    self.data, self.slices = torch.load(processed_path, map_location='cpu', weights_only=False)
                except TypeError:
                    self.data, self.slices = torch.load(processed_path, map_location='cpu')
        except Exception:
            # If anything goes wrong, proceed with whatever is loaded
            pass

    @property
    def raw_file_names(self):
        """Return list of raw file names."""
        if not os.path.exists(self.raw_dir):
            return []
        return [f for f in os.listdir(self.raw_dir) if f.endswith(('.cif', '.pdb'))]

    @property
    def processed_file_names(self):
        return [f'esa3d_pdb_max_atoms_{self.max_atoms}.pt']

    def download(self):
        """Attempt to download a small curated set of PDB/mmCIF files if raw_dir is empty.

        If there is no internet or the server is blocked, instruct the user to place files manually.
        """
        # If raw has any pdb/cif files, nothing to do
        if os.path.exists(self.raw_dir):
            have_files = any(f.endswith((".cif", ".pdb")) for f in os.listdir(self.raw_dir))
            if have_files:
                return
        else:
            os.makedirs(self.raw_dir, exist_ok=True)

        # Try to fetch a small sample set
        sample_ids = ["1CRN", "1UBQ", "4HHB"]
        try:
            import urllib.request
            for pdb_id in sample_ids:
                url = f"https://files.rcsb.org/download/{pdb_id}.cif"
                out_path = os.path.join(self.raw_dir, f"{pdb_id}.cif")
                if not os.path.exists(out_path):
                    print(f"📥 Downloading {pdb_id}.cif from RCSB...")
                    urllib.request.urlretrieve(url, out_path)
            print("✅ Downloaded sample PDB files.")
        except Exception as e:
            print(f"⚠️ Could not download sample PDB files automatically: {e}")
            print(f"📁 Please place PDB/mmCIF files in: {self.raw_dir}")
            print("   Supported formats: .cif, .pdb")
            # Do not raise to allow manual placement before next run

    def process(self):
        """Process PDB files into ESA3D format efficiently."""
        if not BIOPYTHON_AVAILABLE:
            raise ImportError("BioPython is required for PDB processing")
        
        print(f"🔧 Processing PDB files from {self.raw_dir}...")
        
        data_list = []
        pdb_parser = PDBParser(QUIET=True)
        cif_parser = MMCIFParser(QUIET=True)
        
        for filename in tqdm(self.raw_paths, desc="Processing PDB files"):
            try:
                # Parse structure
                if filename.endswith('.cif'):
                    structure = cif_parser.get_structure('protein', filename)
                else:
                    structure = pdb_parser.get_structure('protein', filename)
                
                # Extract coordinates and atom types
                positions, atom_types = [], []
                for atom in structure.get_atoms():
                    element = atom.element.strip().upper()
                    positions.append(atom.get_coord())
                    atom_types.append(self.atom_mapping.get(element, 0))
                
                # Filter by size
                if not positions or len(positions) > self.max_atoms:
                    continue
                
                # Convert to tensors
                pos = torch.tensor(np.array(positions), dtype=torch.float32)
                z = torch.tensor(atom_types, dtype=torch.long)
                
                # Create Data object
                data = Data(
                    pos=pos,
                    z=z,
                    protein_id=os.path.basename(filename),
                    num_atoms=len(positions)
                )
                
                # Build edges using a radius graph for node or edge attention
                if radius_graph is not None and len(positions) > 1:
                    try:
                        # Use configured geometric neighborhood parameters
                        cutoff = float(self.cutoff_distance)
                        max_neighbors = int(self.max_neighbors)
                        edge_index = radius_graph(
                            pos, r=cutoff, batch=None, loop=False, max_num_neighbors=max_neighbors
                        )
                        data.edge_index = edge_index
                    except Exception as e:
                        print(f"Warning: could not build radius graph for {filename}: {e}")
                
                data_list.append(data)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        # Efficiently collate and save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        print(f"✅ Processed {len(data_list)} proteins")
        print(f"💾 Saved to: {self.processed_paths[0]}")
        # Save cache metadata fingerprint for fast reuse
        try:
            meta = {
                'raw_count': self._raw_count(),
                'raw_hash': self._raw_fingerprint(),
                'config_sig': self._config_signature(),
            }
            os.makedirs(self.processed_dir, exist_ok=True)
            meta_path = os.path.join(self.processed_dir, f'esa3d_pdb_meta_{self.max_atoms}.json')
            import json
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
        except Exception:
            pass

    def _raw_count(self) -> int:
        try:
            files = [f for f in os.listdir(self.raw_dir) if f.endswith(('.cif', '.pdb'))]
            return len(files)
        except Exception:
            return 0

    def _raw_fingerprint(self) -> str:
        """Compute a lightweight fingerprint of the raw set: name|size|mtime hash."""
        import hashlib
        h = hashlib.sha256()
        try:
            entries = []
            for name in os.listdir(self.raw_dir):
                if not name.endswith(('.cif', '.pdb')):
                    continue
                p = os.path.join(self.raw_dir, name)
                try:
                    st = os.stat(p)
                    entries.append((name, st.st_size, int(st.st_mtime)))
                except Exception:
                    continue
            entries.sort()
            for name, size, mtime in entries:
                h.update(f"{name}|{size}|{mtime}\n".encode('utf-8'))
            return h.hexdigest()
        except Exception:
            return ""

    def _config_signature(self) -> str:
        return f"max_atoms={self.max_atoms};cutoff={self.cutoff_distance};max_neighbors={self.max_neighbors}"

    def _cache_is_fresh(self) -> bool:
        """Compare stored metadata with current raw fingerprint and config."""
        import json
        meta_path = os.path.join(self.processed_dir, f'esa3d_pdb_meta_{self.max_atoms}.json')
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if meta.get('config_sig') != self._config_signature():
                return False
            if meta.get('raw_count') != self._raw_count():
                return False
            if meta.get('raw_hash') != self._raw_fingerprint():
                return False
            print("✅ Using cached processed PDB dataset (fingerprint matched)")
            return True
        except Exception:
            return False


def create_qm9_dataset(root_dir="./data/esa3d_qm9"):
    """Create ESA3D QM9 dataset."""
    print("🚀 Creating ESA3D QM9 Dataset")
    print("=" * 50)
    
    dataset = ESA3DQM9Dataset(root=root_dir)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total molecules: {len(dataset)}")
    print(f"   File size: {os.path.getsize(dataset.processed_paths[0]) / (1024**3):.2f} GB")
    
    # Test DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        print(f"   Sample batch: {batch.num_graphs} molecules, {batch.num_nodes} atoms")
        break
    
    return dataset


def create_pdb_dataset(root_dir="./data/esa3d_pdb", pdb_source_dir=None):
    """Create ESA3D PDB dataset."""
    print("🚀 Creating ESA3D PDB Dataset")
    print("=" * 50)
    
    if pdb_source_dir and not os.path.exists(os.path.join(root_dir, 'raw')):
        # Copy PDB files to raw directory
        import shutil
        raw_dir = os.path.join(root_dir, 'raw')
        os.makedirs(raw_dir, exist_ok=True)
        
        print(f"📁 Copying PDB files from {pdb_source_dir} to {raw_dir}")
        for file in os.listdir(pdb_source_dir):
            if file.endswith(('.cif', '.pdb')):
                shutil.copy2(
                    os.path.join(pdb_source_dir, file),
                    os.path.join(raw_dir, file)
                )
    
    dataset = ESA3DPDBDataset(root=root_dir)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total proteins: {len(dataset)}")
    print(f"   File size: {os.path.getsize(dataset.processed_paths[0]) / (1024**3):.2f} GB")
    
    # Test DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch in dataloader:
        print(f"   Sample batch: {batch.num_graphs} proteins, {batch.num_nodes} atoms")
        break
    
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create ESA3D datasets")
    parser.add_argument("--dataset", choices=["qm9", "pdb"], required=True, help="Dataset type")
    parser.add_argument("--root", default=None, help="Root directory for dataset")
    parser.add_argument("--pdb-source", default=None, help="Source directory for PDB files")
    
    args = parser.parse_args()
    
    if args.dataset == "qm9":
        root_dir = args.root or "./data/esa3d_qm9"
        create_qm9_dataset(root_dir)
    elif args.dataset == "pdb":
        root_dir = args.root or "./data/esa3d_pdb"
        create_pdb_dataset(root_dir, args.pdb_source) 