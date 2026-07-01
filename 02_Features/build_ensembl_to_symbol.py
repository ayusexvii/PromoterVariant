"""
Build mapping from Ensembl ID to gene symbol using GENCODE.
"""
import gzip
import pandas as pd
import json
from pathlib import Path

print("=" * 70)
print("🧬 BUILDING ENSEMBL TO SYMBOL MAPPING")
print("=" * 70)

# Load GENCODE GTF
gtf_path = Path("../01_Data/raw/gencode.v46.basic.annotation.gtf.gz")

if not gtf_path.exists():
    print("❌ GENCODE file not found!")
    exit(1)

print("📂 Loading GENCODE annotation...")

# Extract gene symbol and Ensembl ID mapping
mapping = {}
with gzip.open(gtf_path, 'rt') as f:
    for line in f:
        if line.startswith('#'):
            continue
        fields = line.strip().split('\t')
        if len(fields) < 9 or fields[2] != 'gene':
            continue
        
        info = fields[8]
        
        # Extract gene_name and gene_id
        gene_name = None
        gene_id = None
        
        for item in info.split(';'):
            item = item.strip()
            if item.startswith('gene_name "'):
                gene_name = item.split('"')[1]
            elif item.startswith('gene_id "'):
                gene_id = item.split('"')[1].split('.')[0]  # Remove version
        
        if gene_name and gene_id:
            mapping[gene_id] = gene_name

print(f"✅ Mapped {len(mapping):,} Ensembl IDs to gene symbols")

# Save mapping
with open('processed/ensembl_to_symbol.json', 'w') as f:
    json.dump(mapping, f, indent=2)
print("💾 Saved to: processed/ensembl_to_symbol.json")

# Check specific genes
print("\n📊 Target genes:")
for ens in ['ENSG00000186575', 'ENSG00000100342', 'ENSG00000164362', 'ENSG00000244734']:
    symbol = mapping.get(ens, 'NOT FOUND')
    print(f"   {ens} -> {symbol}")
