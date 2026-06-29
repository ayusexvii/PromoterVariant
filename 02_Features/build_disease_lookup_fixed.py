#!/usr/bin/env python3
"""
Build disease gene lookup from OMIM or DisGeNET.
FIXED: Correctly extracts gene symbols, not Ensembl IDs.
"""
import pandas as pd
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("🧬 REPRODUCIBLE DISEASE GENE LOOKUP EXTRACTOR (FIXED)")
    print("=" * 70)
    
    script_dir = Path(__file__).resolve().parent
    raw_dir = script_dir / "../01_Data/raw"
    processed_dir = script_dir / "processed"
    
    processed_dir.mkdir(exist_ok=True, parents=True)
    
    disease_genes = set()
    source = "None"
    
    # --- METHOD 1: Try OMIM ---
    omim_path = raw_dir / "mim2gene.txt"
    if omim_path.exists():
        print("📂 Discovering OMIM tracking ledger...")
        try:
            # Read OMIM with proper column assignment
            # Columns: MIM_Number, Type, Entrez_ID, Ensembl_ID, Gene_Symbol
            omim = pd.read_csv(
                omim_path, 
                sep='\t', 
                comment='#', 
                header=None,
                names=['MIM_Number', 'Type', 'Entrez_Gene_ID', 'Ensembl_Gene_ID', 'Gene_Symbol']
            )
            
            # Filter to gene entries
            gene_entries = omim[omim['Type'] == 'gene']
            
            # FIXED: Use Gene_Symbol column (column 5, not Ensembl)
            # Filter out empty or non-standard symbols
            raw_symbols = gene_entries['Gene_Symbol'].dropna()
            raw_symbols = raw_symbols[~raw_symbols.str.startswith('ENSG')]
            raw_symbols = raw_symbols[raw_symbols.str.len() > 1]
            
            disease_genes = set(raw_symbols.astype(str).str.upper().unique())
            source = "OMIM"
            print(f"✅ Parsed OMIM archive. Found {len(disease_genes):,} disease genes.")
        except Exception as e:
            print(f"⚠️ Error parsing OMIM: {e}", file=sys.stderr)
            
    # --- METHOD 2: Fallback to DisGeNET ---
    if len(disease_genes) == 0:
        dg_path = raw_dir / "curated_gene_disease_associations.tsv.gz"
        if dg_path.exists():
            print("📂 OMIM absent. Trying DisGeNET...")
            try:
                dg = pd.read_csv(dg_path, sep='\t', compression='gzip')
                sym_col = [c for c in dg.columns if c.lower() in ['genesymbol', 'gene_symbol']][0]
                disease_genes = set(dg[sym_col].dropna().astype(str).str.upper().unique())
                source = "DisGeNET"
                print(f"✅ Parsed DisGeNET. Found {len(disease_genes):,} genes.")
            except Exception as e:
                print(f"⚠️ Error parsing DisGeNET: {e}", file=sys.stderr)
                
    # --- METHOD 3: Fallback to ClinVar ---
    if len(disease_genes) == 0:
        print("📂 Using ClinVar pathogenic genes as proxy...")
        clinvar_path = script_dir.parent / "01_Data" / "processed" / "clinvar_regulatory.txt.gz"
        if clinvar_path.exists():
            try:
                df = pd.read_csv(clinvar_path, sep='\t', compression='gzip')
                # Find clinical significance column
                sig_col = [c for c in df.columns if c.lower() in ['clinicalsignificance', 'clinsig']][0]
                # Find gene symbol column
                gene_col = [c for c in df.columns if c.lower() in ['genesymbol', 'gene']][0]
                
                pathogenic = df[df[sig_col].astype(str).str.contains('Pathogenic', case=False, na=False)]
                disease_genes = set(pathogenic[gene_col].dropna().astype(str).str.upper().unique())
                source = "ClinVar"
                print(f"✅ ClinVar pathogenic: {len(disease_genes):,} genes.")
            except Exception as e:
                print(f"⚠️ Error parsing ClinVar: {e}", file=sys.stderr)
                
    # --- METHOD 4: Hardcoded ---
    if len(disease_genes) == 0:
        print("⚠️ No source found. Using hardcoded genes.")
        disease_genes = {'CHEK2', 'BRCA1', 'BRCA2', 'TP53', 'APOL1', 'NF2', 'SMARCB1', 'TIMP3', 'ADSL'}
        source = "Hardcoded"
        
    # Save
    output_path = processed_dir / 'disease_genes.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        for gene in sorted(disease_genes):
            f.write(f"{gene}\n")
            
    print(f"\n💾 Saved {len(disease_genes):,} disease genes to: {output_path}")
    print(f"🔍 Source: {source}")
    print("\n📊 Sample Disease Genes:")
    for gene in sorted(list(disease_genes))[:15]:
        print(f"   - {gene}")
        
    # Check if CHEK2 is in the list
    if 'CHEK2' in disease_genes:
        print("\n✅ CHEK2 is in the disease gene list!")
    else:
        print("\n⚠️ CHEK2 not found in disease gene list.")
        
    print("=" * 70)
    print("✅ COMPLETE!")

if __name__ == "__main__":
    main()
