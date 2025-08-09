#!/usr/bin/env python3
"""
Small Molecule Dataset Downloader for ESA3D

This script downloads various small molecule datasets suitable for ESA3D training:
- QM9: Standard quantum chemistry dataset (133k molecules)
- DrugBank: Drug molecules (11k+ approved drugs)
- ChEMBL: Bioactive molecules (2M+ compounds)
- ZINC: Commercial compound library (230M+ compounds)

These datasets are compatible with ESA3D's 3D molecular representation learning.
"""

import os
import requests
import gzip
import shutil
import tarfile
import zipfile
from tqdm import tqdm
import time
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle

# Dataset configurations
DATASETS = {
    "qm9": {
        "name": "QM9 Quantum Chemistry Dataset",
        "description": "133,885 molecules with quantum chemical properties",
        "size_mb": 850,
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv",
        "format": "csv",
        "features": ["3D conformations", "quantum properties", "SMILES"]
    },
    "drugbank": {
        "name": "DrugBank Approved Drugs",
        "description": "11,000+ FDA approved drugs with 3D structures",
        "size_mb": 150,
        "url": "https://go.drugbank.com/releases/latest/downloads/all-drugbank-vocabulary",
        "format": "tsv",
        "features": ["drug structures", "targets", "SMILES"]
    },
    "chembl_sample": {
        "name": "ChEMBL Bioactive Molecules (Sample)",
        "description": "100k bioactive molecules from ChEMBL",
        "size_mb": 50,
        "url": "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_30_chemreps.txt.gz",
        "format": "tsv.gz",
        "features": ["bioactive compounds", "targets", "SMILES"]
    },
    "zinc_sample": {
        "name": "ZINC Commercial Compounds (Sample)",
        "description": "1M commercial compounds from ZINC",
        "size_mb": 200,
        "url": "https://files.docking.org/3D/3D.sdf.gz",
        "format": "sdf.gz",
        "features": ["commercial compounds", "3D structures", "SDF format"]
    }
}

# Download settings
DOWNLOAD_DIR = "./data/small_molecules"
MAX_WORKERS = 8

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
                for chunk in response.iter_content(chunk_size=8192):
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

def process_qm9_dataset(filepath, output_dir):
    """Process QM9 dataset for ESA3D format."""
    print(f"\n🔧 Processing QM9 dataset...")
    
    # Read CSV
    df = pd.read_csv(filepath)
    print(f"📊 Loaded {len(df)} molecules from QM9")
    
    # Convert to ESA3D format
    processed_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing molecules"):
        try:
            # Get SMILES
            smiles = row['smiles']
            
            # Generate 3D conformation
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate 3D conformation
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            # Get 3D coordinates
            conf = mol.GetConformer()
            positions = []
            atom_types = []
            
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                positions.append([pos.x, pos.y, pos.z])
                atom_types.append(atom.GetAtomicNum())
            
            if len(positions) > 0:
                processed_data.append({
                    'smiles': smiles,
                    'positions': positions,
                    'atom_types': atom_types,
                    'properties': {
                        'homo': row.get('homo', 0),
                        'lumo': row.get('lumo', 0),
                        'gap': row.get('gap', 0),
                        'mu': row.get('mu', 0),
                        'alpha': row.get('alpha', 0)
                    }
                })
                
        except Exception as e:
            continue
    
    # Save processed data
    output_file = os.path.join(output_dir, 'qm9_processed.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"✅ Processed {len(processed_data)} molecules")
    print(f"💾 Saved to: {output_file}")
    return output_file

def process_drugbank_dataset(filepath, output_dir):
    """Process DrugBank dataset for ESA3D format."""
    print(f"\n🔧 Processing DrugBank dataset...")
    
    # Read TSV
    df = pd.read_csv(filepath, sep='\t')
    print(f"📊 Loaded {len(df)} drugs from DrugBank")
    
    # Convert to ESA3D format
    processed_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing drugs"):
        try:
            # Get SMILES
            smiles = row.get('SMILES', '')
            if pd.isna(smiles) or smiles == '':
                continue
            
            # Generate 3D conformation
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Generate 3D conformation
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            # Get 3D coordinates
            conf = mol.GetConformer()
            positions = []
            atom_types = []
            
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                positions.append([pos.x, pos.y, pos.z])
                atom_types.append(atom.GetAtomicNum())
            
            if len(positions) > 0:
                processed_data.append({
                    'smiles': smiles,
                    'positions': positions,
                    'atom_types': atom_types,
                    'properties': {
                        'drug_name': row.get('Name', ''),
                        'drug_type': row.get('Type', ''),
                        'groups': row.get('Groups', '')
                    }
                })
                
        except Exception as e:
            continue
    
    # Save processed data
    output_file = os.path.join(output_dir, 'drugbank_processed.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"✅ Processed {len(processed_data)} drugs")
    print(f"💾 Saved to: {output_file}")
    return output_file

def main():
    """Main function to download and process small molecule datasets."""
    print("🚀 Small Molecule Dataset Downloader for ESA3D")
    print("=" * 60)
    
    # Show available datasets
    print("\n📋 Available datasets:")
    for i, (key, info) in enumerate(DATASETS.items(), 1):
        print(f"  {i}. {info['name']}")
        print(f"     Size: {info['size_mb']} MB")
        print(f"     Description: {info['description']}")
        print(f"     Features: {', '.join(info['features'])}")
        print()
    
    # Ask user which dataset to download
    print("Which dataset would you like to download?")
    print("Options:")
    print("  1. qm9 (Quantum Chemistry - 133k molecules)")
    print("  2. drugbank (FDA Approved Drugs - 11k+ drugs)")
    print("  3. chembl_sample (Bioactive Molecules - 100k sample)")
    print("  4. zinc_sample (Commercial Compounds - 1M sample)")
    print("  5. all (all datasets)")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    # Map choice to dataset
    choice_map = {
        "1": ["qm9"],
        "2": ["drugbank"],
        "3": ["chembl_sample"],
        "4": ["zinc_sample"],
        "5": list(DATASETS.keys())
    }
    
    if choice not in choice_map:
        print("❌ Invalid choice. Exiting.")
        return
    
    selected_datasets = choice_map[choice]
    
    # Create output directory
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    # Download and process selected datasets
    for dataset_key in selected_datasets:
        dataset_info = DATASETS[dataset_key]
        
        # Create filepath
        filename = os.path.basename(dataset_info["url"])
        filepath = os.path.join(DOWNLOAD_DIR, filename)
        
        # Check if file already exists
        if os.path.exists(filepath):
            print(f"\n⚠️  File already exists: {filepath}")
            overwrite = input("Do you want to overwrite? (y/N): ").strip().lower()
            if overwrite != 'y':
                print(f"⏩ Skipping {dataset_key}")
                continue
        
        # Download file
        success = download_file(
            dataset_info["url"], 
            filepath, 
            dataset_info["description"]
        )
        
        if success:
            # Process dataset based on type
            if dataset_key == "qm9":
                process_qm9_dataset(filepath, DOWNLOAD_DIR)
            elif dataset_key == "drugbank":
                process_drugbank_dataset(filepath, DOWNLOAD_DIR)
            else:
                print(f"📦 Dataset '{dataset_key}' downloaded to: {filepath}")
                print(f"   (Manual processing may be required for {dataset_key})")
    
    print("\n🎉 Download and processing completed!")
    print(f"📁 All datasets are in: {DOWNLOAD_DIR}")
    print("\n📝 Next steps:")
    print("   1. Check the processed .pkl files")
    print("   2. Update ESA3D config.yaml to use these datasets")
    print("   3. Run ESA3D training with the new data")

if __name__ == "__main__":
    main() 