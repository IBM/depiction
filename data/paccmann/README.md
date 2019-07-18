# GDSC compounds

Molecular structure in `.smi` format and gene expression in `.csv.gz` format for the compounds and cell lines considered in [Yang et al.](https://academic.oup.com/nar/article/41/D1/D955/1059448).

- `gdsc.smi`, publicly available structures for 209 drugs used in the study.
- `gdsc.csv.gz`, gene expression for 970 cell lines in 2128 genes selected via network propagation as described in [Oskooei at al.](https://arxiv.org/abs/1811.06802) and [Manica et al.](https://arxiv.org/abs/1904.11223).
- `gdsc_sensitivity.csv.gz`, drug sensitivity for 212539 pairs expressed as a boolean label: 1 effective (IC50 < 1 um), 0 non effective.
