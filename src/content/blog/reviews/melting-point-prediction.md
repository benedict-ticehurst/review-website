---
title: 'Building and Comparing Models for Melting Point Prediction of Organic Compounds'
description: 'A comparative evaluation of four modelling strategies, PCA-based and full-descriptor linear regression, LASSO regression, and descriptor- and Morgan-fingerprint-based neural networks, for predicting the melting points of organic compounds from molecular structure.'
pubDate: 2026-04-29
---

**Benedict Ticehurst**
*Yr 4, Machine Learning and AI, University of Southampton*

## Contents

1. Introduction
2. The Datasets
3. Molecular Descriptors
4. Principal Component Analysis
5. Morgan Fingerprints and Hierarchical Clustering
   1. Hierarchical Clustering
6. Linear Regression
7. LASSO Regression
8. Neural Networks: Descriptor-Based
   1. Architecture and Training
   2. Results
9. Neural Networks: Morgan Fingerprint-Based
   1. Architecture and Training
   2. Results
   3. Comparison with Descriptor Network and Learning Curves
10. Discussion
    1. What Additional Information Would Improve Future Models?
11. Conclusion
12. References

## 1. Introduction

The melting point of an organic compound reflects the thermodynamic balance between the stability of the crystalline solid and the thermal energy required to disrupt intermolecular forces and produce the disordered liquid phase. It is determined by a complex interplay of molecular size, shape, polarity, rigidity, hydrogen-bonding capacity, aromaticity, and crystal packing symmetry. As a consequence, predicting melting point from molecular structure alone is a challenging but scientifically valuable problem. Within the pharmaceutical industry in particular, melting point is directly linked to aqueous solubility through the Yalkowsky general solubility equation, making it an important parameter for early-stage drug candidate screening [1].

Explicit quantum-mechanical or molecular-dynamics modelling of the full crystalline lattice, large enough to capture all relevant intermolecular interactions, is computationally intractable for large compound libraries. Statistical and machine-learning approaches therefore offer a practical alternative. In this project, four modelling strategies were evaluated and compared: ordinary least-squares linear regression on molecular descriptors and principal components; LASSO regularised regression; and two neural network architectures using descriptors and Morgan fingerprints, respectively. This makes it possible to compare, fairly directly, whether the model choice or the molecular representation matters more for predicting melting point.

## 2. The Datasets

Four datasets of experimentally measured melting points were provided as CSV files. The OCHEM [2] and Enamine datasets (21,883 and 22,404 molecules, respectively) were used as the combined training source. The Bradley [3] and Bergstrom [4] datasets (2,886 and 277 molecules, respectively) were reserved throughout as independent test sets. The combined training set contains 44,286 molecules after removing entries with missing or non-numeric melting points; the test set contains 3,163 molecules. All melting points are reported in Kelvin.

The four datasets were originally described in Nigsch *et al.* [1] and span a broad range of drug-like organic compounds. The Bradley and Bergstrom datasets are curated, high-quality collections and represent a more demanding test than using a random split of the same source data. Training on OCHEM and Enamine, which come from higher-throughput measurements and literature aggregations, and testing on these curated collections gives a tougher test than a random split of the same data would.

## 3. Molecular Descriptors

Twenty-four molecular descriptors were calculated from SMILES strings using the RDKit library [5], including the five from the example notebook (TPSA, NumHAcceptors, NumHDonors, RingCount, MolLogP) and nineteen additional descriptors selected for their expected chemical relevance. These capture the key physicochemical features expected to influence melting point: molecular size (MolWt, AtomNum, LabuteASA), polarity and hydrogen bonding (TPSA, NumHDonors, NumHAcceptors), flexibility (NumRotBond, SP3Hydrid), aromatic character (NumAroRings, NumAroCarboC), ring systems (NumAliRing, NumSatRing, NumSpiro, NumBridge), heteroatom and carbon composition (NumHeteroA, NumAliHeteroC, NumAroHeteroC, NumSatHeteroC, NumSatCarboC, NumAliCarboC), and functional group complexity (NumAmide, NumStereocentres). Scatter plots of all 24 descriptors against experimental melting point are provided in Figure H.

## 4. Principal Component Analysis

Principal component analysis was applied to the standardised descriptor matrix for all 47,450 molecules across all four datasets combined, as required. Standardisation (mean-centring and unit-variance scaling) was performed before fitting to prevent descriptors with large numerical ranges from dominating the variance decomposition. This combined fit is strictly exploratory analysis of the descriptor space; the PCA model and scaler used in the supervised regression section were refitted on the training set only, so no test-set information enters the predictive models.

> **Figure 1.** Left: scree plot of variance explained by each principal component. Right: cumulative explained variance with reference lines at 90% and 95%. The first 10 PCs account for 92.2% of total variance; 12 PCs account for 95.4%.

The first principal component (PC1) accounts for 26.4% of the total descriptor variance and is dominated by molecular size and ring-system descriptors: NumAliRing (loading 0.326), AtomNum (0.318), NumSatRing (0.308), LabuteASA (0.305), and NumStereocentres (0.286). This component broadly separates small, simple molecules from large, complex ring systems. The second component (PC2, 18.7% of variance) captures aromaticity, with strong positive loadings on NumAroRings (0.392) and NumAroCarboC (0.321), and a negative loading on SP3Hydrid (−0.285), distinguishing flat, aromatic compounds from saturated, three-dimensional ones.

Together, the first twelve principal components explain 95.4% of the total descriptor variance (Figure 1), and these twelve were used as input to the PC-based linear regression model. This choice balances information retention against the risk of including lower-variance components that may encode noise rather than chemically meaningful structure.

> **Figure 2.** PC1 vs PC2 scores for all four datasets overlaid. The OCHEM and Enamine training data occupy broad, overlapping regions of PC space. The Bradley and Bergstrom test molecules sit largely within the training data distribution, which supports the plausibility of generalisation, though overlap in two dimensions is not a definitive test of full chemical space coverage.

The overlap of all four datasets in the PC1–PC2 plane (Figure 2) shows no obvious separation between the test and training compounds in this projection, which is encouraging for generalisation. The Enamine dataset sits in a noticeably tighter cluster than OCHEM, which likely reflects its more focused pharmaceutical-like origin.

## 5. Morgan Fingerprints and Hierarchical Clustering

Morgan circular fingerprints (ECFP) [8] were calculated for all 47,450 molecules using radius=2 and a bit-vector length of 2048 bits, following the approach demonstrated in the workshop notebook. At radius=2, each bit encodes the presence of a particular substructural environment extending up to two bonds from a central atom. The resulting fingerprint matrix (47,450 × 2048) was saved to disk for reuse in the neural network section.

With an average bit density of 0.014 (1.4% of bits set per molecule), the fingerprints are highly sparse. No bits were found to be universally on or off across the dataset, indicating that all 2048 positions carry at least some discriminating information.

### 5.1. Hierarchical Clustering

Hierarchical clustering with average linkage was performed on a stratified random sample of 200 molecules (50 per dataset) to make pairwise distance computation tractable. Computing pairwise distances for all 47,450 molecules would require approximately 1.12 billion matrix entries, which is intractable for hierarchical clustering. The stratified 200-molecule sample ensures equal representation from all four datasets and captures the broad structural diversity visible in the full PCA scatter; the general conclusions drawn from this sample are representative. Two representations were compared.

Fingerprint clustering used (1 − Tanimoto similarity) as the pairwise distance metric. The distance cutoff of 0.70 was chosen by visual inspection of the dendrogram to identify a threshold giving a biologically meaningful separation without collapsing all molecules into a single cluster or placing every molecule in its own cluster. At a distance cutoff of 0.70, the 200 molecules were distributed across 157 distinct clusters, indicating that most molecules are structurally unique at this level of resolution. The high cluster count reflects the sparse, high-dimensional nature of the fingerprint space: small structural differences map to distinct, non-overlapping bit patterns.

Descriptor clustering used Euclidean distance on the standardised 24-descriptor matrix, with a cutoff of 5.0 chosen by the same visual criterion. At a cutoff of 5.0, the same 200 molecules formed only 24 clusters, with the largest cluster containing over 100 molecules. Descriptors integrate many structural features into continuous values, so structurally diverse compounds can have similar descriptor profiles.

The comparison (Figure M) shows the two representations are related but not equivalent: pairs that fingerprints place in the same cluster often span a wide range of descriptor distances, and vice versa. Fingerprints pick up fine-grained structural differences that continuous descriptors tend to average out.

> **Figure 3.** Fingerprint dendrograms at 512, 1024, and 2048 bits (left to right) at Tanimoto distance cutoff = 0.70. The number of resolved clusters increases with bit length: 141 clusters at 512 bits, 148 at 1024 bits, and 157 at 2048 bits. Longer fingerprints encode more structural detail and suffer fewer bit collisions, resolving more distinct molecular environments.

**Table 1.** Effect of fingerprint bit length on clustering and mean pairwise Tanimoto similarity for the 200-molecule sample (radius=2, cutoff=0.70).

| Bit length | Clusters | Mean Tanimoto sim. | Interpretation |
|---|---|---|---|
| 512 | 141 | 0.1110 | Highest bit collision rate |
| 1024 | 148 | 0.0990 | Intermediate |
| 2048 | 157 | 0.0934 | Lowest bit collision rate |

As shown in Figure 3 and Table 1, the number of clusters increases from 141 at 512 bits to 157 at 2048 bits. This is consistent with physical expectations: shorter fingerprints have more bit collisions (different substructures hashing to the same position), which inflates apparent molecular similarity and reduces the number of resolved clusters. The mean pairwise Tanimoto similarity decreases from 0.111 to 0.093 as bit length increases, confirming that collision rate falls and structural discrimination improves. Given this result, 2048 bits was used as the fingerprint length throughout the neural network section.

## 6. Linear Regression

Three ordinary least-squares linear regression models were fitted using the combined OCHEM and Enamine training data and evaluated on the combined Bradley and Bergstrom test set. Descriptor features were standardised before fitting in all cases, with the scaler fitted on the training data only and then applied to the test data.

Model 1 used the top 12 principal components from the PCA fitted in Section 4, which together explain 95.4% of total descriptor variance. The PC scores for the training and test molecules were computed by applying the saved PCA model and scaler.

Model 2 used a hand-selected set of 12 descriptors, chosen to match the number of PCs used in Model 1. The selected descriptors were MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors, NumAroRings, RingCount, NumRotBond, SP3Hydrid, LabuteASA, NumStereocentres, and NumAmide. This set was chosen to represent the main axes of physicochemical variation identified in the PCA (size, aromaticity, polarity, flexibility, and hydrogen bonding).

Model 3 used all 24 descriptors, providing the maximum information available from this representation.

**Table 2.** Test set performance of the four linear-style models. RMSE and MAE are in Kelvin. Model 4 is the LASSO result from Section 7.

| Model | R² | MAE (K) | RMSE (K) |
|---|---|---|---|
| Model 1: Linear regression, top 12 PCs | 0.485 | 54.83 | 69.03 |
| Model 2: Linear regression, 12 descriptors | 0.530 | 52.83 | 65.94 |
| Model 3: Linear regression, 24 descriptors | 0.566 | 50.24 | 63.39 |
| Model 4: LASSO (α = 0.01, 24 descriptors) | 0.567 | 50.19 | 63.33 |

The results (Table 2) show a clear and consistent improvement from Model 1 to Model 3: using the raw descriptors rather than PCA-compressed scores, and using more descriptors rather than fewer, both improve performance. Model 1 (R² = 0.485, RMSE = 69.0 K) performs worst, suggesting that the PCA transformation loses information that is predictive of melting point even when retaining 95% of descriptor variance. Model 3 (R² = 0.566, RMSE = 63.4 K) outperforms Model 2 (R² = 0.530, RMSE = 65.9 K), indicating that the additional 12 descriptors not included in the curated set contribute useful predictive signal, despite potentially higher multicollinearity; the RMSE comparison across all four models is shown in Figure N.

All three linear models show substantial residual RMSE (>60 K), indicating that the relationship between descriptors and melting point is not well captured by a linear model alone. The systematic scatter in the predicted vs. experimental plots reflects the inability of a linear combination of these descriptors to account for crystal packing effects and higher-order structure–property relationships.

## 7. LASSO Regression

LASSO (Least Absolute Shrinkage and Selection Operator) [9] was chosen over ridge regression because it performs simultaneous variable selection by shrinking some coefficients exactly to zero. This is desirable given that several of the 24 descriptors are likely to be collinear or uninformative, and a sparser model is more interpretable.

The OCHEM and Enamine training data were split 90/10 into a training subset and a validation set using a random split with `random_state=42`. A random split was used rather than a dataset-level split because both OCHEM and Enamine cover similar chemical space, making either a valid source of validation data. Seven values of the regularisation parameter α were evaluated over four orders of magnitude: 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, and 100.0.

> **Figure 4.** Left: validation R² vs log(α). Centre: validation MAE vs log(α). Right: number of non-zero LASSO coefficients vs log(α). Performance is near-constant for α ≤ 0.01, then degrades steadily. α = 0.01 was selected as it achieves the best validation R² whilst retaining 20 of 24 descriptors.

The validation performance (Figure 4) is nearly insensitive to α below 0.01, with R² remaining at approximately 0.509 and MAE near 40.9 K for the smallest three values tested. Above α = 0.1, performance degrades as the regularisation becomes strong enough to shrink informative coefficients. The value α = 0.01 was selected as the best, retaining 20 of 24 descriptors while achieving the highest validation R².

The four descriptors eliminated at this alpha were RingCount, NumAroRings, NumAliCarboC, and NumSatRing. These are likely collinear with retained descriptors: for example, RingCount correlates with the more specific NumAliRing, NumAroRings, and NumSatRing counts. The largest positive coefficients in the final model were associated with TPSA (+40.7) and LabuteASA (+43.8), consistent with their role as measures of molecular surface area and hydrogen-bond exposure, both of which favour stronger intermolecular interactions and higher melting points. The largest negative coefficient was on NumHAcceptors (−20.5), which at first appears counterintuitive but is compensated by the positive TPSA term; these two descriptors are correlated and the model partitions their contributions accordingly.

On the combined test set, the LASSO model achieves R² = 0.567, MAE = 50.2 K, and RMSE = 63.3 K, nearly identical to Model 3 from the linear regression section. This confirms that the four removed descriptors contribute negligible additional information: the model is no worse after regularisation, and it is more parsimonious. Evaluated separately, the Bradley test set gives R² = 0.539 and RMSE = 64.8 K, while the smaller Bergstrom dataset achieves R² = 0.311 and RMSE = 45.5 K. The lower Bergstrom R² despite a lower RMSE reflects the narrower melting point range in that dataset (a small absolute spread reduces R² even for reasonable absolute errors).

## 8. Neural Networks: Descriptor-Based

A feed-forward neural network was trained using all 24 standardised molecular descriptors as input, following the same 90/10 training-validation split and random state as the LASSO regression. The same 3,163-molecule combined test set was used for final evaluation. Seven activation functions were tested in separate models with identical architecture and training protocol, allowing a fair comparison of their effect on performance.

### 8.1. Architecture and Training

The network architecture consists of three hidden layers with decreasing width: Dense(128) → Dropout(0.2) → Dense(64) → Dropout(0.2) → Dense(32) → Dense(1). The progressive width reduction (128→64→32) follows a standard design pattern for regression networks: the decreasing widths compress the 24-dimensional input progressively toward the single output, while each stage learns increasingly abstract representations. Dropout (20% of neurons dropped during training) reduces overfitting. The output layer uses linear activation, as required for a regression target with an unbounded range. The Adam optimiser [10] was used with a learning rate of 0.001, MSE loss, and a batch size of 64. Early stopping with patience=25 monitored the validation loss, restoring the best weights when training was halted.

The seven activation functions tested were: ReLU, Tanh, ELU (Exponential Linear Unit), Swish, LeakyReLU (α=0.2), Mish, and GELU. LeakyReLU and Mish were implemented as custom TensorFlow functions [7]; the remaining five are available natively in Keras. The architecture was kept identical across all seven models so that only the activation function varies.

### 8.2. Results

**Table 3.** Test set performance of the descriptor-based neural network for all seven activation functions, ranked by RMSE.

| Activation | R² | MAE (K) | RMSE (K) |
|---|---|---|---|
| Tanh | 0.828 | 30.98 | 39.92 |
| Mish | 0.823 | 31.45 | 40.43 |
| LeakyReLU | 0.822 | 31.81 | 40.58 |
| Swish | 0.821 | 32.17 | 40.68 |
| ReLU | 0.820 | 31.94 | 40.86 |
| GELU ✓ | 0.808 | 33.11 | 42.20 |
| ELU | 0.797 | 33.96 | 43.36 |

The validation set selected GELU as the best activation (validation RMSE = 43.7 K, R² = 0.667), but on the test set Tanh achieves the best performance (RMSE = 39.9 K, R² = 0.828), as shown in Table 3. This discrepancy highlights the stochastic nature of neural network training and the limitations of a single validation split for model selection. Nevertheless, the five middle-ranking activations (Tanh through ReLU) are clustered within 1 K of each other on the test set, suggesting the architecture itself is more important than the specific activation function.

The improvement over the best linear model is substantial: the selected (GELU) descriptor NN reduces RMSE by 33% relative to LASSO (from 63.3 K to 42.2 K), accounting for 81% of test set variance (R² = 0.808). The size of this improvement suggests the descriptor–melting point relationship is not purely linear, and the network is likely picking up interactions between descriptors that the linear fit cannot represent.

> **Figure 5.** Predicted vs. experimental melting points for the validation-selected descriptor neural network (GELU). The scatter around the 1:1 line is approximately symmetric, with no strong systematic bias. Outliers correspond predominantly to very high melting point compounds, where crystal packing effects dominate and the descriptors are less informative.

## 9. Neural Networks: Morgan Fingerprint-Based

A second neural network was trained using 2048-bit Morgan fingerprints (radius=2) as input, loaded directly from the pre-computed matrix saved in the fingerprints section. The same training-validation-test split was used throughout.

### 9.1. Architecture and Training

Because the input layer is 2048 nodes wide rather than 24, a larger first hidden layer was used to avoid an overly abrupt compression. The architecture is Dense(256) → Dropout(0.3) → Dense(128) → Dropout(0.3) → Dense(64) → Dropout(0.2) → Dense(32) → Dense(1). Higher dropout rates (0.3 rather than 0.2) were used in the first two layers to compensate for the higher-dimensional input, which presents a greater risk of the network memorising training set bit patterns. The same Adam optimiser, learning rate of 0.001, batch size of 256, and early stopping (patience=25) were used as for the descriptor network.

### 9.2. Results

**Table 4.** Test set performance of the fingerprint-based neural network for all seven activation functions, ranked by RMSE. ✓ marks the selected model (Tanh), which showed the deepest validation convergence (115 epochs) and also achieved the best test performance.

| Activation | R² | MAE (K) | RMSE (K) |
|---|---|---|---|
| Tanh ✓ | 0.746 | 36.45 | 48.46 |
| ELU | 0.743 | 37.60 | 48.74 |
| LeakyReLU | 0.672 | 42.94 | 55.11 |
| Mish | 0.669 | 43.18 | 55.33 |
| Swish | 0.655 | 44.28 | 56.48 |
| GELU | 0.644 | 44.50 | 57.38 |
| ReLU | 0.613 | 46.50 | 59.85 |

The fingerprint network performs substantially differently depending on the activation function (Table 4). Tanh (RMSE = 48.5 K, R² = 0.746) and ELU (RMSE = 48.7 K, R² = 0.743) are closely matched and considerably better than the remaining five, which cluster between 55 and 60 K RMSE. The clear performance gap between the Tanh/ELU pair and the ReLU-family functions is consistent with the hypothesis that the sparse, binary fingerprint inputs favour smooth, zero-symmetric activation functions: Tanh and ELU can represent both the presence (positive) and absence (effectively zero) of a substructure, while ReLU and its variants may be less well-suited to binary inputs where the zero-state is meaningful. This is a post-hoc hypothesis supported by the observed results rather than a demonstrated mechanism.

The training curves (Figure S) reveal that several activation functions (ReLU, Swish, Mish, GELU) converge in very few epochs (27–33) to relatively high validation losses, indicating early stopping was triggered early. This is consistent with the poor test performance of these functions on fingerprints. By contrast, Tanh trains for 115 epochs before early stopping, exploiting the fingerprint space more thoroughly.

### 9.3. Comparison with Descriptor Network and Learning Curves

The selected fingerprint network (Tanh, RMSE = 48.5 K, R² = 0.746) underperforms the selected descriptor network (GELU, RMSE = 42.2 K, R² = 0.808) by 6.3 K RMSE. One interpretation is that descriptors encode domain knowledge, polarity, size, ring systems, hydrogen bonding, that directly maps onto the physical drivers of melting point, while fingerprints encode local substructural environments that require a more complex mapping to a global thermodynamic property.

To test whether the gap arises from data scarcity rather than representation quality, both networks were retrained on seven progressively larger fractions of the training set (5%, 10%, 20%, 30%, 50%, 75%, and 100%), using Tanh activation throughout. Test RMSE on the combined test set was recorded at each fraction.

> **Figure 6.** Learning curves for the descriptor (blue) and fingerprint (red) neural networks (Tanh activation, log-scale x-axis). Both curves are non-monotonic at intermediate training sizes due to training instability; the dashed lines show the general downward trend at full and near-full data sizes.

The learning curve data (Table E, Figure 6) show non-monotonic behaviour at small-to-intermediate training fractions. For the descriptor network, RMSE is high and erratic below 20% of training data (106–116 K), drops sharply once ~8,000 molecules are available (42–43 K), then spikes again at 50% (115 K) before stabilising below 41 K from 75% onward. The fingerprint network shows a similar instability: RMSE spikes to 114 K at 20% despite being lower at 10% (66 K), before descending to 48–50 K from 50% onward.

These spikes are consistent with the known behaviour of early stopping under unstable training: at certain fractions the randomly sampled subset does not provide sufficient gradient signal at the chosen learning rate, early stopping triggers before the model has escaped a poor local minimum, and the reported test RMSE reflects that unconverged state. From 75% of training data (~30,000 molecules) both networks converge reliably. At 100% (~40,000 molecules), the descriptor network reaches 40.6 K and the fingerprint network reaches 47.3 K, consistent with the main notebook results (GELU 42.2 K and Tanh 48.5 K; the ~2 K difference reflects the change from Tanh to the validation-selected GELU for descriptors). The descriptor network's advantage is present at every training fraction where both models converge. That suggests the gap is not simply due to limited training data, though it may also partly reflect the specific network sizes and training setup used here.

## 10. Discussion

The general trend across all models is that more complex models and richer representations give better predictions. The linear models show one interesting result: retaining 95% of descriptor variance in PCA does not mean retaining 95% of the information relevant to melting point. Compressing to 12 PCs worsens RMSE from 63.4 K to 69.0 K compared to using all 24 descriptors directly. LASSO also shows that four of the descriptors can be dropped without any real loss in performance, which makes sense given how many of the ring-count descriptors overlap.

The neural networks give the biggest improvement. The selected descriptor network (GELU) cuts RMSE by roughly a third relative to LASSO, which suggests the relationship between these descriptors and melting point is not simply additive. The fingerprint network still lags behind even when both models are trained on the largest data fractions, so the gap is probably not just a consequence of having less training data. The descriptor advantage more likely reflects the fact that these descriptors directly encode properties relevant to melting point, polarity, hydrogen bonding, ring geometry, whereas fingerprints encode local substructure that requires a less obvious mapping to a bulk thermodynamic property.

The choice of activation function also matters more for the fingerprint network (~11 K spread) than for the descriptor network (~3.4 K). One possible reason is that Tanh and ELU are zero-centred, so they treat a zero bit differently from a positive one, which could help the network distinguish between a substructure being absent versus present. ReLU-family functions do not have this property. This is speculative, but the pattern in the results is consistent with it.

All models remain well above experimental measurement uncertainty (~5–10 K). That is probably unavoidable given the inputs: no two-dimensional molecular representation captures how a compound actually packs into a crystal lattice.

### 10.1. What Additional Information Would Improve Future Models?

The most obvious gap in all the models is crystal-structure information. Melting point is fundamentally a solid-state property: it depends on how tightly and regularly molecules pack into a lattice, which is not captured by any two-dimensional representation. Two compounds with the same SMILES can have very different melting points if they adopt different crystal polymorphs. Including space group data or unit cell parameters from the Cambridge Structural Database would be the most direct way to address this.

Periodic DFT calculations can give estimated lattice energies, which relate more directly to melting point than any molecular descriptor. The cost is much higher, but for a targeted study of a smaller compound set this would be practical.

Within the existing approach, useful additions would include polymorphic form labels (polymorphs of the same compound can differ by 10–50 K), and more data at the high-melting-point end (above ~550 K) where all models currently struggle most. Better experimental metadata, measurement method, solvent used, reported purity, could also help, since the training data aggregates results from many different sources.

## 11. Conclusion

The linear models establish a baseline of RMSE ~63–69 K depending on representation. LASSO and full-descriptor regression give nearly identical results, and both outperform the PCA-based model, which shows that compressing descriptors to preserve variance does not preserve the information most useful for the target. The neural networks perform considerably better: the selected descriptor network (GELU) reaches RMSE = 42.2 K (R² = 0.808) and the fingerprint network (Tanh) reaches 48.5 K (R² = 0.746). The descriptor network's advantage over the fingerprint network holds across all training data sizes where both converge, which suggests the gap is not just a data-volume issue. The activation function matters noticeably more for the fingerprint network than for the descriptor one.

All models are still well above experimental uncertainty. The most likely reason is that melting point is set by crystal packing, which none of the inputs here, SMILES, descriptors, or fingerprints, can fully represent. Incorporating crystal structure data, calculated lattice energies, or polymorphic form labels would be the most direct way to improve predictions beyond the ceiling these models appear to be approaching.

## References

1. F. L. Nigsch, A. Bender, B. van Buuren, J. Tissen, E. Nigsch and J. B. O. Mitchell, *J. Chem. Inf. Model.*, 2006, **46**, 2412–2422.
2. I. Sushko, S. Novotarskyi, R. Körner, A. K. Pandya, M. Rupp, W. Teetz, S. Brandmaier, A. Abdelaziz, V. V. Prokopenko and I. V. Tetko, *J. Comput. Aided Mol. Des.*, 2011, **25**, 533–554.
3. J. S. Bradley, *Open Melting Point Dataset*, Figshare, 2010.
4. C. A. S. Bergstrom, U. Norinder, K. Luthman and P. Artursson, *J. Chem. Inf. Comput. Sci.*, 2002, **42**, 1125–1136.
5. RDKit: Open-Source Cheminformatics, https://www.rdkit.org.
6. F. Pedregosa *et al.*, *J. Mach. Learn. Res.*, 2011, **12**, 2825–2830.
7. M. Abadi *et al.*, *TensorFlow: Large-scale machine learning on heterogeneous systems*, 2015.
8. D. Rogers and M. Hahn, *J. Chem. Inf. Model.*, 2010, **50**, 742–754.
9. R. Tibshirani, *J. R. Statist. Soc. B*, 1996, **58**, 267–288.
10. D. P. Kingma and J. Ba, *Adam: A method for stochastic optimization*, arXiv:1412.6980, 2014.

## Appendix

Supplementary figures referenced in the main text and additional visualisations supporting the analysis.

> **Figure G.** Distribution of experimental melting points across the four datasets. The combined training data (OCHEM and Enamine) spans approximately 200–700 K; the Bradley and Bergstrom test sets cover a similar range. The distribution is unimodal and slightly right-skewed, with the modal melting point near 430 K.

> **Figure H.** Scatter plots of each of the 24 calculated descriptors against experimental melting point for the training set. Descriptors with the clearest trends include TPSA, MolWt, and LabuteASA, which show a broad positive correlation with melting point, while MolLogP and NumRotBond are weakly negatively correlated, consistent with the role of lipophilicity and flexibility in reducing crystal packing stability.

> **Figure I.** PC1 vs PC2 scores plotted separately for each of the four datasets. Enamine occupies a more compact, lower-PC1 region typical of focused pharmaceutical scaffolds; OCHEM spans a wider area reflecting its heterogeneous literature origins. The Bradley and Bergstrom test sets cluster within the training distribution in both cases.

> **Figure J.** Heatmap of PCA loadings for PC1 to PC5. PC1 is dominated by size and ring descriptors (positive); PC2 by aromaticity (positive) versus sp³ hybridisation (negative); PC3 by heteroatom content and hydrogen-bonding capacity; PC4 and PC5 capture more subtle structural features including bridgehead and spiro atoms.

> **Figure K.** Hierarchical clustering dendrogram for the 200-molecule stratified sample using (1 − Tanimoto similarity) as the pairwise distance metric. At a cutoff of 0.70, 157 distinct clusters are identified, reflecting the high structural diversity of the dataset and the fine-grained discrimination of Morgan fingerprints.

> **Figure L.** Hierarchical clustering dendrogram for the 200-molecule stratified sample using Euclidean distance on standardised 24-descriptor vectors. At a cutoff of 5.0, only 24 clusters are formed with the largest containing over 100 molecules, indicating that continuous descriptors aggregate structural diversity more coarsely than binary fingerprints.

> **Figure M.** Scatter plot of pairwise fingerprint Tanimoto distances versus normalised pairwise descriptor Euclidean distances for all 200×199/2 molecular pairs in the clustering sample. The weak positive correlation (Pearson r = 0.317) indicates that while the two representations broadly agree on which molecules are most dissimilar, they measure different aspects of chemical space and produce markedly different cluster structures.

> **Figure N.** RMSE bar chart comparing all four linear-style models on the combined test set. Performance improves progressively from the PC-based Model 1 to the full-descriptor LASSO model (Model 4), though the improvement from Model 3 to Model 4 is negligible.

> **Figure O.** Predicted versus experimental melting points on the combined test set for Model 1 (linear regression on top 12 principal components, R² = 0.485, RMSE = 69.0 K). The broad scatter and systematic under-prediction of high melting points reflect the information loss from PCA compression.

> **Figure P.** Predicted versus experimental melting points for Model 2 (linear regression on 12 hand-selected descriptors, R² = 0.530, RMSE = 65.9 K). The scatter is slightly tighter than Model 1 at the upper end of the melting point range.

> **Figure Q.** Predicted versus experimental melting points for Model 3 (linear regression on all 24 descriptors, R² = 0.566, RMSE = 63.4 K). The improvement over Models 1 and 2 is consistent across the full melting point range.

> **Figure R.** Predicted versus experimental melting points on the combined test set for the optimised LASSO model (α = 0.01, R² = 0.567, RMSE = 63.3 K). The scatter pattern is near-identical to Model 3, confirming that removing four collinear descriptors does not alter the model's predictive behaviour.

> **Figure S.** Training and validation loss curves for all seven activation functions on the fingerprint neural network. Tanh shows the slowest but deepest convergence over 115 epochs; most other activations converge (or stop via early stopping) within 30 epochs, suggesting they are more susceptible to getting stuck in local minima or stopping before fully exploiting the high-dimensional fingerprint space.

> **Figure T.** Predicted versus experimental melting points on the combined test set for all seven activation functions in the descriptor-based neural network. Tanh, Mish, LeakyReLU, Swish, and ReLU produce similar tight scatter patterns around the 1:1 line; ELU shows slightly wider residuals consistent with its lower R² of 0.797.

> **Figure U.** Training and validation loss curves for all seven activation functions on the descriptor-based neural network. ELU converges rapidly but to a higher loss; the remaining activations (particularly ReLU, Tanh, Swish, LeakyReLU, Mish, and GELU) show gradual convergence with the best models trained for 200–280 epochs before early stopping.

> **Figure V.** Predicted vs. experimental melting points on the test set for all seven fingerprint neural network activations. The Tanh and ELU models show the tightest scatter around the 1:1 line. The ReLU model shows a systematic compression of predictions toward the centre of the distribution, reflecting underfitting.

**Table E.** Test RMSE at each training fraction for the descriptor (Desc.) and fingerprint (FP) networks. Both networks used Tanh activation. Spikes at some intermediate fractions reflect training instability rather than a genuine performance reversal.

| Fraction | n<sub>train</sub> | Desc. RMSE (K) | FP RMSE (K) |
|---|---|---|---|
| 5% | 1,992 | 106.4 | 71.2 |
| 10% | 3,985 | 116.4 | 65.7 |
| 20% | 7,971 | 42.8 | 114.2 |
| 30% | 11,956 | 42.0 | 67.2 |
| 50% | 19,928 | 115.1 | 50.5 |
| 75% | 29,892 | 40.0 | 48.9 |
| 100% | 39,857 | 40.6 | 47.3 |
