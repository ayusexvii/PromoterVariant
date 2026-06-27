#!/usr/bin/env python3
import gzip
import re

def build_gene_mapping(gtf_path: str, output_path: str):
    """Robust GENCODE gene_name → Ensembl ID mapping"""
    mapping = {}
    count = 0
    
    print("🔄 Parsing GENCODE GTF (this may take 30-60 seconds)...")
    
    with gzip.open(gtf_path, 'rt', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.startswith('#'):
                continue
            if 'gene_name' not in line or 'gene_id' not in line:
                continue
                
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'gene':
                continue
            
            info = fields[8]
            
            # More robust regex
            gene_name_match = re.search(r'gene_name "([^"]+)"', info)
            gene_id_match = re.search(r'gene_id "([^"]+?)"', info)
            
            if gene_name_match and gene_id_match:
                gene_name = gene_name_match.group(1).strip()
                gene_id = gene_id_match.group(1).split('.')[0]  # Remove version
                mapping[gene_name] = gene_id
                count += 1
                
                if count % 5000 == 0:
                    print(f"  → Mapped {count:,} genes...", end='\r')
    
    # Write mapping
    with open(output_path, 'w', encoding='utf-8') as f:
        for gene_name in sorted(mapping.keys()):
            f.write(f"{gene_name}\t{mapping[gene_name]}\n")
    
    print(f"\n✅ Successfully mapped {len(mapping):,} genes")
    return mapping

if __name__ == "__main__":
    build_gene_mapping(
        "../01_Data/raw/gencode.v46.basic.annotation.gtf.gz",
        "gencode_gene_mapping_full.txt"
    )
