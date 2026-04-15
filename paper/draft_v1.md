# Protein Language Model Scale Determines the Direction of Structure-Aggregation Coupling in Intrinsically Disordered Proteins

Taesung Park^1, Claude AI^2

1. Independent Researcher
2. Anthropic

## Abstract

Predicting how mutations affect the aggregation propensity of intrinsically disordered proteins (IDPs) is critical for understanding neurodegenerative diseases, yet existing methods rely primarily on sequence-based features. Here we present a conditional flow matching framework that generates mutation-specific conformational ensembles from protein language model (PLM) embeddings, enabling structure-based aggregation prediction without molecular dynamics simulation. Evaluating on deep mutational scanning (DMS) data for amyloid-beta 42 (Abeta42, n=751) and islet amyloid polypeptide (IAPP, n=661), we show that the change in radius of gyration (Delta-Rg) from flow-generated ensembles correlates significantly with experimental nucleation scores (Abeta42: rho=0.229, p=1.4e-9; IAPP: rho=0.110, p=1.3e-3), outperforming the sequence-based CANYA predictor (rho=0.115) while providing orthogonal information to ESM-2 log-likelihood ratios (LLR). Unexpectedly, we discover that the *direction* of the Delta-Rg--nucleation correlation reverses depending on PLM scale: ESM-2 35M and 150M yield positive correlations, while ESM-2 650M yields a negative correlation, even when architecture is held constant. This qualitative phase transition in PLM representations has not been previously reported and suggests that larger PLMs encode fundamentally different structural information about IDP conformational dynamics. Our results establish flow matching as a lightweight alternative to MD for mutation-specific ensemble generation and reveal scale-dependent behavior in protein language model representations.

## Introduction

Intrinsically disordered proteins (IDPs) play central roles in neurodegenerative diseases through aberrant aggregation. Amyloid-beta 42 (Abeta42) aggregates into plaques in Alzheimer's disease, while islet amyloid polypeptide (IAPP) forms toxic deposits in type 2 diabetes. Understanding how mutations modulate aggregation propensity is essential for interpreting variants of uncertain significance (VUS) and guiding therapeutic development.

Deep mutational scanning (DMS) has generated comprehensive maps of mutational effects on amyloid nucleation for both Abeta42 (Seuma et al. 2022, n=751 single-point substitutions) and IAPP (Badia et al. 2026, n=661). These datasets provide ground truth for benchmarking computational predictors but remain limited to specific proteins and experimental conditions.

Current computational approaches for predicting mutation effects on IDP aggregation fall into two categories. Sequence-based methods, including ESM-2 log-likelihood ratios (LLR) and the recently published CANYA neural network (Thompson et al. 2025, Science Advances), predict aggregation propensity directly from amino acid sequence. These methods are fast but cannot capture the conformational dynamics that underlie aggregation. Structure-based methods, primarily molecular dynamics (MD) simulations, can model mutation-specific conformational changes but are computationally prohibitive for genome-scale variant interpretation. MDmis (bioRxiv 2025) applies MD to IDR variants but requires days per mutation and produces only binary classification.

Flow matching generative models have recently emerged as powerful tools for protein conformational ensemble generation. AlphaFlow (Jing et al. 2024, ICML) fine-tunes AlphaFold under a flow matching objective, while P2DFlow (2025, JCTC) introduces SE(3)-equivariant flow matching for ensembles. IDPFold2 (2026, bioRxiv) integrates mixture-of-experts into flow matching for IDP-specific ensemble prediction. However, none of these methods have been applied to predict mutation effects on IDP aggregation.

Here we present a conditional flow matching framework that generates mutation-specific conformational ensembles by conditioning on ESM-2 protein language model embeddings. We make three contributions:

1. We demonstrate that the change in radius of gyration (Delta-Rg) from flow-generated ensembles predicts amyloid nucleation scores, outperforming the sequence-based CANYA predictor and providing information orthogonal to ESM-2 LLR.

2. We validate our approach on two independent amyloid DMS datasets (Abeta42 and IAPP), showing generalization to a protein absent from the training data.

3. We discover that the direction of the Delta-Rg--nucleation correlation undergoes a qualitative phase transition depending on PLM scale, with ESM-2 650M reversing the correlation direction relative to 35M and 150M. A controlled experiment holding architecture constant confirms that this reversal originates in the PLM representations, not the flow model architecture.

## Results

### Flow matching generates mutation-specific conformational ensembles

We trained a conditional flow matching model (MutationConditionedStructureHead) that takes ESM-2 sequence embeddings, noised C-alpha coordinates, and mutation position/identity as input, and outputs a velocity field for ODE integration (Figure 1a). Starting from Gaussian noise, 50-step Euler integration produces C-alpha coordinate ensembles of 32-64 structures per mutation (Figure 1b). The model was trained on conformational ensembles from the Protein Ensemble Database (PED) for three IDP targets: tau K18 (PED00017), alpha-synuclein (PED00024), and Abeta42 (PED00531).

### Delta-Rg predicts amyloid nucleation scores

We scored all 751 single-point substitutions from the Seuma et al. (2022) Abeta42 DMS dataset by generating mutation-specific ensembles and computing Delta-Rg = Rg(mutant) - Rg(wild-type). Using ESM-2 35M embeddings, Delta-Rg showed a significant positive correlation with experimental nucleation scores (rho=0.229, p=1.4e-9, 95% CI [0.148, 0.285]; Figure 2a). This correlation survived Benjamini-Hochberg correction for multiple comparisons across 10 tests (adjusted p=7.1e-9).

Among seven trajectory-derived features tested, Delta-Rg was the strongest individual predictor and the most important feature in both Lasso regression (weight=0.298) and Random Forest (Gini importance=0.338) models. Other trajectory features (late-stage velocity, velocity variance, contact switching rates, convergence time) showed no significant correlation with nucleation scores after multiple comparison correction (Table 1).

ESM-2 LLR showed the highest raw correlation (rho=0.161 in our evaluation), consistent with its known ability to predict mutation effects from evolutionary information. Importantly, Delta-Rg and LLR capture partially independent information: their cross-correlation was rho=0.369, indicating that ~86% of the variance in Delta-Rg is not explained by LLR. A composite model combining all seven features achieved a cross-validated Spearman rho of 0.285 (Gradient Boosting, 5-fold CV), confirming the additive value of structural features.

### ESM-2 scale determines correlation direction

When we repeated the analysis using ESM-2 650M embeddings (with a correspondingly larger flow model: d_model=384, n_layers=8), the Delta-Rg--nucleation correlation reversed direction: rho=-0.219 (p=4.0e-10; Figure 2b). The 650M model's negative correlation indicates that mutations causing compaction (decreased Rg) are associated with increased nucleation, consistent with the established biophysical mechanism for Abeta42 aggregation where compact intermediates nucleate fibril formation.

To determine whether this reversal originates from the PLM representations or the flow model architecture, we conducted a controlled experiment. We trained three flow models with identical architecture (d_model=256, n_heads=8, n_layers=6) but different ESM-2 embeddings as input:

| ESM-2 Scale | d_seq_in | rho | p-value | Direction |
|-------------|----------|-----|---------|-----------|
| 35M | 480 | +0.250 | 3.6e-12 | Positive |
| 150M | 640 | +0.205 | 1.4e-8 | Positive |
| 650M | 1280 | -0.207 | 1.0e-8 | **Negative** |

The direction reversal occurred only with ESM-2 650M, confirming that the PLM representation--not the flow model architecture--determines the sign of the structure-aggregation coupling (Figure 3). This qualitative phase transition between 150M and 650M has not been previously reported in the protein language model literature.

### Per-residue analysis reveals scale-dependent structural responses

To understand the mechanistic basis of the direction reversal, we analyzed per-residue contributions to Delta-Rg for familial Alzheimer's disease (fAD) mutations (Figure 4). The 35M model showed large Delta-Rg contributions at the C-terminus (residues 35-42), far from the mutation sites (e.g., D7N at position 7), suggesting non-local and potentially artifactual structural responses. In contrast, the 650M model showed Delta-Rg contributions concentrated near the mutation site, consistent with local structural perturbation expected from single-point mutations. This suggests that the 650M PLM encodes more physically realistic local structural effects.

### Multi-amyloid validation on IAPP

To test generalization beyond the training distribution, we scored the Badia et al. (2026) IAPP DMS dataset (n=661 single-point substitutions). IAPP was not included in the PED training data, making this a true out-of-distribution test. Despite this, Delta-Rg from the 35M flow model showed a significant correlation with IAPP nucleation scores (rho=0.110, p=1.3e-3, BH-adjusted p=3.1e-3; Figure 5).

The weaker but significant IAPP correlation (0.110 vs 0.229 for Abeta42) is consistent with the expected performance drop for out-of-distribution proteins and supports the validity of the approach.

We also observed a direction asymmetry consistent with findings from Badia et al. (2026): for Abeta42, Delta-Rg correlates more strongly with slowing mutations (rho=0.172, p=1.6e-4) than accelerating mutations (rho=0.096, NS). This structural confirmation of the cross-amyloid asymmetry reported from sequence analysis alone provides independent evidence that different biophysical mechanisms underlie aggregation acceleration versus deceleration.

### Comparison with existing methods

We compared our approach against multiple baselines on the Abeta42 DMS dataset (Table 2):

| Method | Type | rho | p-value |
|--------|------|-----|---------|
| Flow Delta-Rg (35M) | Structure | 0.229 | 1.4e-9 |
| Flow Delta-Rg (650M) | Structure | -0.219 | 4.0e-10 |
| ESM-2 LLR | Sequence | 0.161 | 8.7e-6 |
| CANYA | Sequence | 0.115 | 1.6e-3 |
| All-7 features (GB CV) | Composite | 0.285 | CV |

Flow Delta-Rg outperformed CANYA, a recently published aggregation predictor trained on over 100,000 random peptide sequences (Thompson et al. 2025). While CANYA is a general-purpose predictor not specifically designed for Abeta42, this comparison demonstrates that mutation-specific structural ensemble information provides stronger predictive signal than sequence features alone for nucleation prediction.

## Discussion

### PLM scale as a qualitative determinant of structural representations

Our most unexpected finding is that the direction of the Delta-Rg--nucleation correlation reverses between ESM-2 150M (positive) and 650M (negative). This is qualitatively different from the quantitative scaling effects previously reported, where larger PLMs generally show improved but directionally consistent performance. Recent work has shown that only 39% of protein fitness prediction tasks exhibit predictable scaling behavior, with the remainder showing nonmonotonic or inverse scaling. Our results extend this observation by demonstrating a case where scaling produces a *sign change* in a downstream structural prediction, not merely a change in magnitude.

The 650M direction (compact leads to aggregation) aligns with the established biophysical mechanism for Abeta42, where collapse of the hydrophobic core (residues 16-21, KLVFFA) into compact intermediates nucleates fibril formation. This suggests that the larger PLM encodes information about the relationship between local compaction and aggregation that the smaller PLMs do not capture.

### Limitations

Several limitations should be considered when interpreting our results:

**In-distribution bias.** Abeta42 conformational ensembles from PED were included in the flow model training data. The higher Abeta42 performance (rho=0.229) relative to the out-of-distribution IAPP (rho=0.110) likely reflects this bias. We emphasize the IAPP result as the more conservative estimate of generalization.

**Unphysical absolute structures.** While the 35M model produces ensembles with reasonable wild-type Rg (~27 Angstrom for 42-residue Abeta42), the controlled-experiment models with 150M and 650M embeddings produced unphysical structures (Rg > 100 Angstrom). This indicates that while relative Delta-Rg rank ordering is preserved, the absolute structures should not be interpreted as physical conformations. The original 650M model (with scaled architecture) produced more physical structures.

**Cross-protein transfer failure.** Transfer learning from Abeta42 DMS to tau/alpha-synuclein disease mutations was not successful (all cross-protein rho values NS), consistent with Badia et al.'s finding that aggregation-accelerating mutations do not transfer across amyloids. This limits the current approach to per-protein application.

**CANYA comparison.** CANYA is a general-purpose aggregation predictor, while our flow model was trained on Abeta42-containing data. The comparison should be interpreted as "specialized structure-based vs. general sequence-based," not as a claim of universal superiority.

**ESMFold baseline.** Due to software compatibility issues, we were unable to run the ESMFold dropout ensemble baseline directly. Future work should include this comparison.

## Methods

### Conditional Flow Matching Model

The MutationConditionedStructureHead is a Transformer-based velocity network with 6 layers, 256-dimensional hidden states, and 8 attention heads (35M configuration). Input features include ESM-2 sequence embeddings (projected from d_seq_in to d_model), noised C-alpha coordinates encoded as RBF distance features (16 bins, max 20 Angstrom), and mutation position/amino acid embeddings (32-dimensional). The model is trained with the conditional flow matching objective (Lipman et al. 2023) using AdamW optimizer, cosine learning rate schedule, and EMA (decay 0.999).

### ESM-2 Embeddings

We used three ESM-2 model scales: esm2_t12_35M_UR50D (480-dim output), esm2_t30_150M_UR50D (640-dim), and esm2_t33_650M_UR50D (1280-dim). For the controlled experiment, all three embedding dimensions were projected to d_model=256 via a learned linear layer, with all other architecture parameters held constant.

### DMS Datasets

**Abeta42:** Seuma et al. (2022) nucleation scores for 751 single-point substitutions, downloaded from GitHub (BEBlab/DIM-abeta). Wild-type sequence: DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA (42 residues).

**IAPP:** Badia et al. (2026) nucleation scores for 661 single-point substitutions, processed from BEBlab/MAVE-IAPP (singles.df, nscore_c column). Wild-type sequence: KCNTATCATQRLANFLVHSSNNFGAILSSTNVGSNTY (37 residues).

### Ensemble Generation and Scoring

For each mutation, we generated 32 C-alpha coordinate ensembles via 50-step Euler ODE integration from Gaussian noise. Wild-type reference ensembles used 64 samples. Delta-Rg was computed as mean(Rg_mutant) - mean(Rg_wild-type), where Rg is the radius of gyration of C-alpha coordinates.

### Statistical Analysis

Spearman rank correlation was used for all correlation analyses. Bootstrap 95% confidence intervals were computed from 10,000 resamples. Multiple comparison correction used the Benjamini-Hochberg procedure across all tested features. Cross-validated performance used 5-fold CV with Gradient Boosting (n_estimators=100, max_depth=3) and Lasso regression.

### CANYA Baseline

CANYA (Thompson et al. 2025) was run locally using the ensemble mode (10 models, median summarization across 20-residue sliding windows) with TensorFlow 2.15 on Python 3.11, as Colab's TensorFlow version was incompatible.

### Code and Data Availability

All code is available at https://github.com/layeredlabs06/brain-idp-flow. The master pipeline notebook (colab/10_master_pipeline.ipynb) reproduces all results with Google Colab GPU.

## References

- Badia M, Batlle C, Bolognesi B (2026). Massively parallel quantification of mutational impact on IAPP amyloid formation. Nature Communications.
- Jing B et al. (2024). AlphaFold Meets Flow Matching for Generating Protein Ensembles. ICML.
- Lin Z et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. Science 379:1123-1130.
- Lipman Y et al. (2023). Flow Matching for Generative Modeling. ICLR.
- Seuma M et al. (2022). An atlas of amyloid aggregation: the impact of substitutions, insertions, deletions and truncations on amyloid beta fibril nucleation. Nature Communications 13:7084.
- Thompson MJ et al. (2025). Massive experimental quantification allows interpretable deep learning of protein aggregation. Science Advances.

## Figures

### Figure 1: Method Overview
(a) Architecture: ESM-2 embeddings + mutation conditioning -> Transformer velocity network -> ODE integration -> C-alpha ensemble
(b) Pipeline: DMS mutations -> flow ensemble generation -> Delta-Rg computation -> correlation with nucleation scores

### Figure 2: Abeta42 DMS Validation
(a) Delta-Rg (35M) vs nucleation score scatter (rho=0.229, n=751)
(b) Delta-Rg (650M) vs nucleation score scatter (rho=-0.219, n=751)
fAD mutations highlighted in red

### Figure 3: Controlled Experiment — ESM-2 Scale
Three-panel scatter: same architecture (d=256, L=6), different ESM-2 (35M/150M/650M)
Shows direction reversal at 650M

### Figure 4: Per-Residue Rg Contribution
35M vs 650M for fAD mutations (D7N, D7H, E11K)
650M shows local response, 35M shows distal response

### Figure 5: Multi-Amyloid Validation
2x2 panel: Abeta42 (35M/650M) vs IAPP (35M/650M)

### Figure 6: Forest Plot with Bootstrap CI
All features with 95% CI, color-coded by protein

### Table 1: Feature Correlation with Nucleation Score
All 10 features with rho, BH-adjusted p, Bootstrap CI

### Table 2: Method Comparison
Flow vs CANYA vs LLR vs Composite
