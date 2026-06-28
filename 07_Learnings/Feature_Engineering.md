# Feature Engineering for Regulatory Variants

## What Works
1. **Distance to TSS** - Most important biological feature
2. **Gene Variant Count** - Genes with more variants are more predictable
3. **Absolute Distance** - Symmetrical effect around TSS

## What Doesn't Work
1. **Distance Squared** - Redundant with distance
2. **Log Distance** - Redundant with distance
3. **Direction** - No directional bias in effect

## Biological Insight
Distance to TSS alone explains ~50% of variance in eQTL effects.
This confirms that promoter-proximal variants are more impactful.
