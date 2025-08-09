#!/usr/bin/env python3
"""
QM9 Dataset Downloader for ESA3D using PyTorch Geometric

This script downloads QM9 dataset using PyTorch Geometric and converts it to ESA3D format.
PyTorch Geometric provides a reliable way to access QM9 with proper 3D coordinates.

Key Features:
- Downloads QM9 via PyTorch Geometric (reliable source)
- Preserves 3D coordinates and quantum properties
- Converts to ESA3D-compatible format
- Uses torch.save for optimal PyTorch compatibility
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.datasets import QM9
from torch_geometric.data import Data
import pickle

# Download settings
DOWNLOAD_DIR = "./data/qm9_pyg"

def download_qm9_pyg():
    """Download QM9 dataset using PyTorch Geometric."""
    print("🚀 QM9 Dataset Downloader for ESA3D using PyTorch Geometric")
    print("=" * 70)
    print("📋 This script downloads QM9 dataset with proper 3D coordinates")
    print("🔬 Uses PyTorch Geometric for reliable data access")
    print()
    
    # Create output directory
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    print("📥 Downloading QM9 dataset via PyTorch Geometric...")
    print("⚠️  This may take a while on first run...")
    
    try:
        # Download QM9 dataset
        dataset = QM9(root=DOWNLOAD_DIR)
        print(f"✅ QM9 dataset downloaded successfully!")
        print(f"📊 Dataset size: {len(dataset)} molecules")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Failed to download QM9: {e}")
        return None

def convert_to_esa3d_format(dataset):
    """Convert PyTorch Geometric QM9 to ESA3D format."""
    print(f"\n🔧 Converting QM9 to ESA3D format...")
    
    # QM9 property names
    property_names = [
        'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv'
    ]
    
    processed_data = []
    
    for i, data in enumerate(tqdm(dataset, desc="Converting molecules")):
        try:
            # Extract 3D coordinates and atomic numbers
            positions = data.pos  # 3D coordinates
            atomic_numbers = data.z  # Atomic numbers
            
            # Extract properties
            properties = {}
            for j, prop_name in enumerate(property_names):
                if hasattr(data, prop_name):
                    prop_value = getattr(data, prop_name)
                    if prop_value is not None:
                        properties[prop_name] = float(prop_value)
                    else:
                        properties[prop_name] = 0.0
                else:
                    properties[prop_name] = 0.0
            
            # Create ESA3D-compatible data structure
            molecule_data = {
                'mol_id': f'qm9_{i:06d}',
                'positions': positions,  # Already torch.tensor
                'atomic_numbers': atomic_numbers,  # Already torch.tensor
                'properties': properties,
                'num_atoms': len(atomic_numbers)
            }
            
            processed_data.append(molecule_data)
            
        except Exception as e:
            print(f"Error processing molecule {i}: {e}")
            continue
    
    return processed_data

def save_esa3d_format(processed_data, output_dir):
    """Save processed data in ESA3D format."""
    print(f"\n💾 Saving ESA3D format data...")
    
    # Save as PyTorch file
    output_file = os.path.join(output_dir, 'qm9_esa3d.pt')
    torch.save(processed_data, output_file)
    
    # Save as pickle for compatibility
    pickle_file = os.path.join(output_dir, 'qm9_esa3d.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"✅ Saved {len(processed_data)} molecules")
    print(f"📁 PyTorch format: {output_file}")
    print(f"📁 Pickle format: {pickle_file}")
    
    return output_file, pickle_file

def analyze_dataset(processed_data):
    """Analyze the processed dataset."""
    print(f"\n📊 Dataset Analysis:")
    print(f"   Total molecules: {len(processed_data)}")
    
    if len(processed_data) > 0:
        # Sample statistics
        sample_mol = processed_data[0]
        print(f"   Sample molecule: {sample_mol['mol_id']}")
        print(f"   Atoms: {sample_mol['num_atoms']}")
        print(f"   Coordinates shape: {sample_mol['positions'].shape}")
        print(f"   Properties: {list(sample_mol['properties'].keys())}")
        
        # Property statistics
        property_names = list(sample_mol['properties'].keys())
        print(f"\n📈 Property Statistics:")
        for prop in property_names:
            values = [mol['properties'][prop] for mol in processed_data if mol['properties'][prop] != 0]
            if values:
                print(f"   {prop}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, range=[{np.min(values):.4f}, {np.max(values):.4f}]")
        
        # Size distribution
        sizes = [mol['num_atoms'] for mol in processed_data]
        print(f"\n📏 Size Distribution:")
        print(f"   Min atoms: {min(sizes)}")
        print(f"   Max atoms: {max(sizes)}")
        print(f"   Mean atoms: {np.mean(sizes):.1f}")
        print(f"   Median atoms: {np.median(sizes):.1f}")

def main():
    """Main function."""
    # Download QM9 dataset
    dataset = download_qm9_pyg()
    
    if dataset is None:
        print("❌ Failed to download QM9 dataset")
        return
    
    # Convert to ESA3D format
    processed_data = convert_to_esa3d_format(dataset)
    
    if len(processed_data) == 0:
        print("❌ No data processed")
        return
    
    # Save in ESA3D format
    pt_file, pkl_file = save_esa3d_format(processed_data, DOWNLOAD_DIR)
    
    # Analyze dataset
    analyze_dataset(processed_data)
    
    print("\n🎉 QM9 dataset processing completed!")
    print(f"📁 Dataset location: {pt_file}")
    print("\n📝 Key features:")
    print("   ✅ High-quality 3D coordinates from PyTorch Geometric")
    print("   ✅ Complete quantum chemistry properties")
    print("   ✅ ESA3D-compatible format")
    print("   ✅ Both PyTorch (.pt) and Pickle (.pkl) formats")
    print("\n🔧 Next steps:")
    print("   1. Update ESA3D config.yaml to use this dataset")
    print("   2. Run ESA3D training with QM9 data")
    print("   3. Use either .pt or .pkl file based on your preference")

if __name__ == "__main__":
    main() 