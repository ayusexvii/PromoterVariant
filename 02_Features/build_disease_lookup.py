#!/usr/bin/env python3
"""
Build disease gene lookup from OMIM or DisGeNET.
Fixes skiprows dimensionality constraints and forces uppercase normalization.
"""
import pandas as pd
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("🧬 REPRODUCIBLE DISEASE GENE LOOKUP EXTRACTOR")
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
            # FIXED: Read OMIM by ignoring comment parsing lines and setting columns manually
            # Official columns: MIM Number, Entry Type, Entrez ID, Ensembl ID, Approved Symbol
            omim = pd.read_csv(
                omim_path, 
                sep='\t', 
                comment='#', 
                header=None,
                names=['MIM_Number', 'Type', 'Entrez_Gene_ID', 'Ensembl_Gene_ID', 'Gene_Symbol']
            )
            
            # Filter layout types strictly matching confirmed gene targets
            gene_entries = omim[omim['Type'] == 'gene']
            raw_symbols = gene_entries['Gene_Symbol'].dropna().astype(str).str.upper().unique()
            disease_genes = set(raw_symbols)
            source = "OMIM"
            print(f"✅ Parsed OMIM archive. Found {len(disease_genes):,} disease genes.")
        except Exception as e:
            print(f"⚠️ Error parsing OMIM: {e}", file=sys.stderr)
            
    # --- METHOD 2: Fallback to DisGeNET ---
    if len(disease_genes) == 0:
        dg_path = raw_dir / "curated_gene_disease_associations.tsv.gz"
        if dg_path.exists():
            print("📂 OMIM absent. Trying DisGeNET database core...")
            try:
                dg = pd.read_csv(dg_path, sep='\t', compression='gzip')
                # Find matching structural column targets dynamically
                sym_col = [c for c in dg.columns if c.lower() in ['genesymbol', 'gene_symbol']][0]
                disease_genes = set(dg[sym_col].dropna().astype(str).str.upper().unique())
                source = "DisGeNET"
                print(f"✅ Parsed DisGeNET archive. Found {len(disease_genes):,} target variants.")
            except Exception as e:
                print(f"⚠️ Error parsing DisGeNET: {e}", file=sys.stderr)
                
    # --- METHOD 3: Fallback to ClinVar Pathogenic Proxy ---
    if len(disease_genes) == 0:
        print("📂 Testing local ClinVar pathogenic genomic tracks as proxy...")
        # Path fixed to check upstream raw data directories relative to workspace boundaries
        clinvar_path = script_dir.parent / "01_Data" / "processed" / "clinvar_regulatory.txt.gz"
        if clinvar_path.exists():
            try:
                df = pd.read_csv(clinvar_path, sep='\t', compression='gzip')
                sig_col = [c for c in df.columns if c.lower() in ['clinicalsignificance', 'clnsig']][0]
                gene_col = [c for c in df.columns if c.lower() in ['genesymbol', 'gene']][0]
                
                pathogenic = df[df[sig_col].astype(str).str.contains('Pathogenic', case=False, na=False)]
                disease_genes = set(pathogenic[gene_col].dropna().astype(str).str.upper().unique())
                source = "ClinVar"
                print(f"✅ ClinVar pathogenic baseline parsed: {len(disease_genes):,} unique loci genes discovered.")
            except Exception as e:
                print(f"⚠️ Error parsing ClinVar alignment tracks: {e}", file=sys.stderr)
                
    # --- METHOD 4: Hardcoded Core Portfolio Targets ---
    if len(disease_genes) == 0:
        print("⚠️ All file sources missing from local disk arrays. Initializing candidate portfolio subset.")
        disease_genes = {'CHEK2', 'BRCA1', 'BRCA2', 'TP53', 'APOL1', 'NF2', 'SMARCB1', 'TIMP3', 'ADSL'}
        source = "Hardcoded_Portfolio"
        
    # 4. Commit and Export Uniform Dataset Outputs
    output_path = processed_dir / 'disease_genes.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        for gene in sorted(disease_genes):
            f.write(f"{gene}\n")
            
    print(f"\n💾 Saved {len(disease_genes):,} normalized disease tracking records to: {output_path}")
    print(f"🔍 Extraction Selection Channel: {source}")
    print("\n📊 First 10 Sample Disease Genes:")
    for gene in sorted(list(disease_genes))[:10]:
        print(f"   - {gene}")
    print("=" * 70)
    print("✅ REGULATORY CONTEXT COMPILATION PIPELINE STEP RUN STABLE!")

if __name__ == "__main__":
    main()
