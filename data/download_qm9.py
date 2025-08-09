#!/usr/bin/env python3
"""
QM9 Dataset Downloader for ESA3D 

This script downloads the ORIGINAL QM9 dataset with high-quality DFT-calculated 3D coordinates.
QM9 contains ~134,000 molecules with quantum chemistry (DFT) calculated geometries.

Key Features:
- Downloads original .xyz files with DFT-calculated 3D coordinates
- Preserves the high-quality quantum chemistry data
- Converts to ESA3D-compatible format efficiently
- Uses torch.save for optimal PyTorch compatibility

Dataset Source: https://quantum-machine.org/datasets/
"""

import os
import requests
import tarfile
import numpy as np
import torch
from tqdm import tqdm
import ase.io
from ase import Atoms
import pickle

# QM9 Dataset URLs (original sources)
QM9_URLS = {
    "xyz_files": "https://quantum-machine.org/datasets/qm9.tar.gz",
    "properties": "https://quantum-machine.org/datasets/qm9.csv"
}

# Download settings
DOWNLOAD_DIR = "./data/qm9_original"
CHUNK_SIZE = 8192

def download_file(url, filepath, description=""):
    """Download file with progress bar."""
    print(f"\n📥 Downloading: {description}")
    print(f"🔗 URL: {url}")
    print(f"💾 Save to: {filepath}")
    
    # Create directory
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✅ Download completed: {filepath}")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def extract_tar_gz(filepath, extract_dir):
    """Extract tar.gz file."""
    print(f"\n📦 Extracting: {filepath}")
    print(f"📁 Extract to: {extract_dir}")
    
    try:
        with tarfile.open(filepath, 'r:gz') as tar:
            members = tar.getmembers()
            
            with tqdm(total=len(members), desc="Extracting") as pbar:
                for member in members:
                    tar.extract(member, extract_dir)
                    pbar.update(1)
        
        print(f"✅ Extraction completed: {extract_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def read_xyz_file(filepath):
    """Read .xyz file and return atomic coordinates and types."""
    try:
        # Read with ASE
        atoms = ase.io.read(filepath, format='xyz')
        
        # Get positions and atomic numbers
        positions = atoms.get_positions()
        atomic_numbers = atoms.get_atomic_numbers()
        
        return positions, atomic_numbers
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None

def process_qm9_xyz_files(xyz_dir, properties_file, output_dir):
    """Process QM9 .xyz files and properties for ESA3D format."""
    print(f"\n🔧 Processing QM9 dataset with original DFT coordinates...")
    
    # Read properties CSV
    import pandas as pd
    properties_df = pd.read_csv(properties_file)
    print(f"📊 Loaded properties for {len(properties_df)} molecules")
    
    # Get list of .xyz files
    xyz_files = [f for f in os.listdir(xyz_dir) if f.endswith('.xyz')]
    xyz_files.sort()
    print(f"📁 Found {len(xyz_files)} .xyz files")
    
    # Process each molecule
    processed_data = []
    
    for i, xyz_file in enumerate(tqdm(xyz_files, desc="Processing molecules")):
        try:
            # Get molecule ID from filename (e.g., "dsgdb9nsd_000001.xyz" -> "dsgdb9nsd_000001")
            mol_id = xyz_file.replace('.xyz', '')
            
            # Read .xyz file with DFT coordinates
            xyz_path = os.path.join(xyz_dir, xyz_file)
            positions, atomic_numbers = read_xyz_file(xyz_path)
            
            if positions is None or atomic_numbers is None:
                continue
            
            # Get properties from CSV
            mol_properties = properties_df[properties_df['mol_id'] == mol_id]
            
            if len(mol_properties) == 0:
                continue
            
            # Extract key properties
            props = mol_properties.iloc[0]
            
            # Create ESA3D-compatible data structure
            molecule_data = {
                'mol_id': mol_id,
                'positions': torch.tensor(positions, dtype=torch.float32),  # DFT-calculated 3D coordinates
                'atomic_numbers': torch.tensor(atomic_numbers, dtype=torch.long),
                'properties': {
                    'homo': float(props.get('homo', 0)),
                    'lumo': float(props.get('lumo', 0)),
                    'gap': float(props.get('gap', 0)),
                    'mu': float(props.get('mu', 0)),
                    'alpha': float(props.get('alpha', 0)),
                    'cv': float(props.get('cv', 0)),
                    'u0': float(props.get('u0', 0)),
                    'u': float(props.get('u', 0)),
                    'h': float(props.get('h', 0)),
                    'g': float(props.get('g', 0)),
                    'zpve': float(props.get('zpve', 0))
                }
            }
            
            processed_data.append(molecule_data)
            
        except Exception as e:
            print(f"Error processing {xyz_file}: {e}")
            continue
    
    # Save processed data
    output_file = os.path.join(output_dir, 'qm9_dft_coordinates.pt')
    torch.save(processed_data, output_file)
    
    print(f"✅ Processed {len(processed_data)} molecules with DFT coordinates")
    print(f"💾 Saved to: {output_file}")
    print(f"📊 Dataset size: {len(processed_data)} molecules")
    
    # Print sample statistics
    if len(processed_data) > 0:
        sample_mol = processed_data[0]
        print(f"📏 Sample molecule: {sample_mol['mol_id']}")
        print(f"   Atoms: {len(sample_mol['atomic_numbers'])}")
        print(f"   Coordinates shape: {sample_mol['positions'].shape}")
        print(f"   Properties: {list(sample_mol['properties'].keys())}")
    
    return output_file

def main():
    """Main function to download and process QM9 dataset."""
    print("🚀 QM9 Dataset Downloader for ESA3D - CORRECT VERSION")
    print("=" * 70)
    print("📋 This script downloads the ORIGINAL QM9 dataset with DFT-calculated 3D coordinates")
    print("🔬 Preserves high-quality quantum chemistry data for accurate training")
    print()
    
    # Create output directory
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    # Download QM9 .xyz files
    xyz_tar_path = os.path.join(DOWNLOAD_DIR, "qm9.tar.gz")
    if not os.path.exists(xyz_tar_path):
        success = download_file(
            QM9_URLS["xyz_files"], 
            xyz_tar_path, 
            "QM9 .xyz files with DFT coordinates"
        )
        if not success:
            print("❌ Failed to download QM9 .xyz files")
            return
    else:
        print(f"⏩ QM9 .xyz files already exist: {xyz_tar_path}")
    
    # Extract .xyz files
    xyz_dir = os.path.join(DOWNLOAD_DIR, "xyz_files")
    if not os.path.exists(xyz_dir):
        success = extract_tar_gz(xyz_tar_path, DOWNLOAD_DIR)
        if not success:
            print("❌ Failed to extract QM9 .xyz files")
            return
    else:
        print(f"⏩ QM9 .xyz files already extracted: {xyz_dir}")
    
    # Download properties CSV
    properties_path = os.path.join(DOWNLOAD_DIR, "qm9.csv")
    if not os.path.exists(properties_path):
        success = download_file(
            QM9_URLS["properties"], 
            properties_path, 
            "QM9 properties CSV"
        )
        if not success:
            print("❌ Failed to download QM9 properties")
            return
    else:
        print(f"⏩ QM9 properties already exist: {properties_path}")
    
    # Process dataset
    output_file = process_qm9_xyz_files(xyz_dir, properties_path, DOWNLOAD_DIR)
    
    print("\n🎉 QM9 dataset processing completed!")
    print(f"📁 Dataset location: {output_file}")
    print("\n📝 Key features:")
    print("   ✅ Original DFT-calculated 3D coordinates preserved")
    print("   ✅ High-quality quantum chemistry data")
    print("   ✅ ESA3D-compatible format")
    print("   ✅ PyTorch .pt format for efficient loading")
    print("\n🔧 Next steps:")
    print("   1. Update ESA3D config.yaml to use this dataset")
    print("   2. Run ESA3D training with high-quality QM9 data")

if __name__ == "__main__":
    main() 