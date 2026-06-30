#!/usr/bin/env python3
"""
Exact-Match Pipeline: Query ClinVar variants against indexed GTEx.
Uses pysam for Tabix queries.
"""
import pysam
import pandas as pd
import gzip
import time
from pathlib import Path

def main():
    print("=" * 70)
    print("🧬 EXACT-MATCH PIPELINE: CLINVAR × GTEX")
    print("=" * 70)
    
    script_dir = Path(__file__).resolve().parent
    
    # Paths
    gtex_tabix = script_dir / "../01_Data/processed/liver_eqtl_sorted.txt.gz"
    clinvar_path = script_dir / "../01_Data/processed/clinvar_regulatory.txt.gz"
    output_path = script_dir / "processed/exact_matches.csv.gz"
    output_dir = script_dir / "processed"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Check files exist
    if not gtex_tabix.exists():
        print(f"❌ ERROR: GTEx Tabix file not found: {gtex_tabix}")
        return
    
    if not clinvar_path.exists():
        print(f"❌ ERROR: ClinVar file not found: {clinvar_path}")
        return
    
    print(f"📂 GTEx Tabix: {gtex_tabix}")
    print(f"📂 ClinVar: {clinvar_path}")
    print(f"📝 Output: {output_path}")
    
    # Open Tabix file
    print("\n🔍 Opening Tabix file...")
    tabix = pysam.TabixFile(str(gtex_tabix))
    
    # Load ClinVar variants (only needed columns)
    print("📂 Loading ClinVar variants...")
    clinvar_df = pd.read_csv(
        clinvar_path,
        sep='\t',
        compression='gzip',
        usecols=['Chromosome', 'Start', 'ReferenceAllele', 'AlternateAllele', 'GeneSymbol', 'ClinicalSignificance']
    )
    print(f"✅ Loaded {len(clinvar_df):,} ClinVar variants")
    
    # Query each variant
    matches = []
    total = 0
    matched = 0
    start_time = time.time()
    
    print("\n🔍 Querying exact positions...")
    
    for idx, row in clinvar_df.iterrows():
        total += 1
        chrom = str(row['Chromosome']).strip()
        pos = int(row['Start'])
        
        try:
            # Query exact position: chrom:pos-pos
            records = tabix.fetch(f"{chrom}:{pos}-{pos}")
            for record in records:
                fields = record.split('\t')
                if len(fields) >= 7:
                    matches.append({
                        'clinvar_chrom': chrom,
                        'clinvar_pos': pos,
                        'clinvar_ref': row['ReferenceAllele'],
                        'clinvar_alt': row['AlternateAllele'],
                        'clinvar_gene': row['GeneSymbol'],
                        'clinvar_sig': row['ClinicalSignificance'],
                        'gtex_ref': fields[2],
                        'gtex_alt': fields[3],
                        'gtex_gene': fields[4],
                        'gtex_slope': float(fields[5]),
                        'gtex_pval': float(fields[6])
                    })
                    matched += 1
        except (ValueError, KeyError):
            # No match at this position
            pass
        
        if total % 10000 == 0:
            elapsed = time.time() - start_time
            match_rate = matched / total * 100 if total > 0 else 0
            print(f"   Processed {total:,} | Matched {matched:,} ({match_rate:.2f}%) | {elapsed:.1f}s")
    
    tabix.close()
    elapsed = time.time() - start_time
    
    print(f"\n" + "=" * 70)
    print("📊 EXACT-MATCH RESULTS")
    print("=" * 70)
    print(f"   Total ClinVar variants: {total:,}")
    print(f"   Exact matches: {matched:,}")
    print(f"   Match rate: {matched/total*100:.2f}%")
    print(f"   Time: {elapsed:.1f}s")
    
    if matches:
        df = pd.DataFrame(matches)
        df.to_csv(output_path, index=False, compression='gzip')
        print(f"   Output: {output_path}")
        print(f"   Size: {output_path.stat().st_size / (1024**2):.1f} MB")
        print(f"   Unique positions: {df[['clinvar_chrom', 'clinvar_pos']].drop_duplicates().shape[0]:,}")
        
        # Show sample matches
        print("\n🔍 Sample matches:")
        print(df[['clinvar_chrom', 'clinvar_pos', 'clinvar_gene', 'gtex_slope']].head(10).to_string())
    else:
        print("⚠️ No matches found!")
    
    print("=" * 70)
    print("✅ EXACT-MATCH PIPELINE COMPLETE!")

if __name__ == "__main__":
    main()
