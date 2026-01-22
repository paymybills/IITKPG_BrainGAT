# BrainGAT: Graph Attention Network for ASD Classification

**Pure Imaging-Based Architecture (No Phenotypic Data)**

---

## 📋 Overview

BrainGAT is a Graph Attention Network (GAT) inspired by the HCAN (Heterogeneous Graph Convolutional Attention Network) paper, but **critically modified to use only fMRI imaging data** without any phenotypic or demographic information.

### Key Innovation
- **HCAN approach**: Heterogeneous population graph with phenotype nodes
- **Our approach**: Homogeneous subject-level graphs with multi-head attention
- **Result**: Interpretable, leakage-free ASD classification

---

## 🧠 Architecture Components

### 1. Input: Graph Representation

Each subject's brain is represented as a graph $G = (V, E, X, E_{attr})$:

**Nodes (V)**:
- $N = 392$ ROIs (brain regions) from CC400 atlas
- Each node $v_i$ represents a specific brain region

**Node Features (X)**:
- $\mathbf{X} \in \mathbb{R}^{N \times N}$ (392 × 392)
- Feature vector for node $i$: $\mathbf{x}_i = \mathbf{C}_{i,:}$ (correlation vector)
- Captures how each ROI correlates with all other ROIs
- **100% imaging-derived** (no demographics!)

**Edges (E)**:
- Sparse connectivity via top-$k$ absolute correlations (k=20)
- For each ROI, keep only strongest $k$ connections
- Undirected graph: $(i,j) \in E \Leftrightarrow (j,i) \in E$

**Edge Attributes (E_attr)**:
- Edge weight $e_{ij} = \mathbf{C}_{ij}$ (Pearson correlation)
- Captures functional connectivity strength

---

## 🔧 Data Processing Pipeline

### Step 1: Time Series to Correlation Matrix

```
fMRI Time Series (T × N) 
    ↓ Pearson Correlation
Correlation Matrix (N × N)
    ↓ Top-k Selection
Sparse Graph (N nodes, ~k·N edges)
```

**Mathematical Formulation**:

$$
\mathbf{C}_{ij} = \text{corr}(\mathbf{T}_{:,i}, \mathbf{T}_{:,j}) = \frac{\sum_t (T_{ti} - \bar{T}_i)(T_{tj} - \bar{T}_j)}{\sqrt{\sum_t (T_{ti} - \bar{T}_i)^2} \sqrt{\sum_t (T_{tj} - \bar{T}_j)^2}}
$$

### Step 2: Node Feature Construction

Each ROI's feature vector is its correlation pattern:

$$
\mathbf{x}_i = [\mathbf{C}_{i,1}, \mathbf{C}_{i,2}, \ldots, \mathbf{C}_{i,N}]^T
$$

### Step 3: Sparse Edge Construction

For each node $i$, select top-$k$ neighbors:

$$
\mathcal{N}_i^{(k)} = \text{top-}k\{|\mathbf{C}_{i,j}| : j \neq i\}
$$

Create undirected edges:

$$
E = \{(i,j) : j \in \mathcal{N}_i^{(k)} \text{ or } i \in \mathcal{N}_j^{(k)}\}
$$

---

## 🏗️ Multi-Head Graph Attention Layer

### Motivation

Standard GCN treats all neighbors equally. **Attention mechanism** learns which connections are important!

### Architecture

**Multi-Head Attention Mechanism**:

For each attention head $h \in \{1, \ldots, H\}$ (we use $H=8$):

1. **Linear Transformation**:
$$
\mathbf{h}_i^{(h)} = \mathbf{W}^{(h)} \mathbf{x}_i
$$

2. **Attention Coefficients** (importance of edge $i \to j$):
$$
e_{ij}^{(h)} = \text{LeakyReLU}(\mathbf{a}^{(h)T} [\mathbf{W}^{(h)}\mathbf{x}_i \| \mathbf{W}^{(h)}\mathbf{x}_j])
$$

3. **Normalized Attention** (via softmax):
$$
\alpha_{ij}^{(h)} = \frac{\exp(e_{ij}^{(h)})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik}^{(h)})}
$$

4. **Weighted Aggregation**:
$$
\mathbf{x}_i^{(h)} = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^{(h)} \mathbf{W}^{(h)} \mathbf{x}_j\right)
$$

5. **Multi-Head Concatenation**:
$$
\mathbf{x}_i^{\text{out}} = \|_{h=1}^H \mathbf{x}_i^{(h)} \quad \text{or} \quad \mathbf{x}_i^{\text{out}} = \frac{1}{H}\sum_{h=1}^H \mathbf{x}_i^{(h)}
$$

### Implementation Details

```python
GATConv(
    in_channels=392,     # Input feature dimension
    out_channels=256,    # Output feature dimension per head
    heads=8,             # Number of attention heads
    dropout=0.3,         # Attention dropout
    edge_dim=1           # Use edge weights (correlations)
)
```

**Post-Processing**:
- Batch Normalization (stabilizes training)
- ELU activation (smooth, negative values allowed)
- Dropout (regularization)

---

## 📊 Complete BrainGAT Architecture

### Layer-by-Layer Breakdown

```
Input Graph: N=392 nodes, ~7,840 edges, features: 392-dim
    ↓
┌─────────────────────────────────────────────────┐
│ Layer 1: Multi-Head GAT                         │
│   - Input: 392-dim → Output: 256-dim × 8 heads │
│   - Attention learns important connections      │
│   - Batch Norm + ELU + Dropout(0.3)            │
└─────────────────────────────────────────────────┘
    ↓ (2048-dim: 256 × 8 concatenated)
┌─────────────────────────────────────────────────┐
│ Layer 2: Multi-Head GAT                         │
│   - Input: 2048-dim → Output: 256-dim × 8 heads│
│   - Higher-order connectivity patterns          │
│   - Batch Norm + ELU + Dropout(0.3)            │
└─────────────────────────────────────────────────┘
    ↓ (2048-dim)
┌─────────────────────────────────────────────────┐
│ Layer 3: Multi-Head GAT (Average)               │
│   - Input: 2048-dim → Output: 256-dim          │
│   - Average attention heads (not concatenate)   │
│   - Global integration features                 │
│   - Batch Norm + ELU + Dropout(0.3)            │
└─────────────────────────────────────────────────┘
    ↓ (256-dim per node)
┌─────────────────────────────────────────────────┐
│ Graph-Level Readout                             │
│   - Mean Pooling: g_mean = mean(x_i)           │
│   - Max Pooling: g_max = max(x_i)              │
│   - Concatenate: g = [g_mean || g_max]         │
└─────────────────────────────────────────────────┘
    ↓ (512-dim: 256 mean + 256 max)
┌─────────────────────────────────────────────────┐
│ MLP Classifier                                  │
│   - Linear: 512 → 256                           │
│   - BatchNorm + ReLU + Dropout(0.5)            │
│   - Linear: 256 → 128                           │
│   - BatchNorm + ReLU + Dropout(0.5)            │
│   - Linear: 128 → 2 (Control vs ASD)           │
└─────────────────────────────────────────────────┘
    ↓
Output: Logits [batch_size, 2] → Softmax → Probabilities
```

### Parameter Count

| Component | Parameters |
|-----------|-----------|
| GAT Layer 1 | ~820K |
| GAT Layer 2 | ~4.2M |
| GAT Layer 3 | ~4.2M |
| MLP Classifier | ~166K |
| **Total** | **~9.4M parameters** |

### Ultra Low Memory Configuration (Used in Script)
To ensure the model runs on GPUs with limited VRAM (e.g., RTX 2050 with 4GB), the provided script `scripts/train_braingat_fixed.py` uses a lightweight configuration:

- **Hidden Channels**: 64 (vs 256)
- **Attention Heads**: 4 (vs 8)
- **Batch Size**: 4
- **Parameters**: ~250K (vs ~9.4M)

This configuration retains the architectural benefits (attention, hierarchy) while fitting into <2GB VRAM.

---

## 🎯 Training Strategy

### Optimization

**Loss Function**: Cross-Entropy Loss
$$
\mathcal{L} = -\frac{1}{B}\sum_{i=1}^B [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]
$$

**Optimizer**: AdamW
- Learning rate: $5 \times 10^{-4}$
- Weight decay: $5 \times 10^{-4}$ (L2 regularization)

**Learning Rate Scheduler**: ReduceLROnPlateau
- Monitor: Validation accuracy
- Factor: 0.5 (halve LR on plateau)
- Patience: 7 epochs

**Early Stopping**:
- Monitor: Validation accuracy
- Patience: 20 epochs
- Save best model checkpoint

### GPU Optimization

**Mixed Precision Training** (when CUDA available):
```python
with torch.cuda.amp.autocast():
    output = model(batch)
    loss = criterion(output, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits**:
- ~2× faster training
- ~40% less memory usage
- Maintains numerical stability

### Regularization Techniques

1. **Dropout**: 0.3 in GAT layers, 0.5 in classifier
2. **Batch Normalization**: After each layer
3. **Weight Decay**: $5 \times 10^{-4}$
4. **Gradient Clipping**: Max norm = 1.0
5. **Early Stopping**: Prevent overfitting

---

## 📈 Data Split Strategy

### Subject-Level Stratified Splitting

**Critical for preventing data leakage!**

```
Total: 351 subjects (190 ASD, 161 Control)
    ↓ Stratified split (85% / 15%)
Train+Val: 298 subjects | Test: 53 subjects
    ↓ Stratified split (~82% / ~18%)
Train: 245 subjects | Val: 53 subjects

Final: 245 / 53 / 53 (Train / Val / Test)
       70% / 15% / 15%
```

### Anti-Leakage Measures

| Measure | Implementation |
|---------|----------------|
| **Subject-level splitting** | Each subject in exactly ONE split |
| **Stratified sampling** | Maintains ASD:Control ratio in all splits |
| **No cross-split normalization** | Features scaled independently per split |
| **Validation-guided stopping** | Best model selected by val accuracy |
| **Test set isolation** | Test set used ONCE for final evaluation |
| **Pure imaging features** | NO phenotypic/demographic variables |

---

## 🔍 Interpretability: Attention Weights

### Why Attention > Standard GCN?

**Standard GCN**: All neighbors contribute equally
$$
\mathbf{h}_i = \sigma\left(\frac{1}{|\mathcal{N}_i|}\sum_{j \in \mathcal{N}_i} \mathbf{W}\mathbf{x}_j\right)
$$

**GAT with Attention**: Neighbors weighted by learned importance
$$
\mathbf{h}_i = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W}\mathbf{x}_j\right)
$$

### Interpreting Attention Weights

**High attention weight** $\alpha_{ij}$ means:
- Connection between ROI $i$ and ROI $j$ is important for classification
- Model learned this connection is discriminative for ASD vs Control
- Can identify specific brain networks involved in autism

**Visualization Capabilities**:
1. **Attention distribution**: Histogram of $\alpha_{ij}$ values
2. **Top-k connections**: Most attended edges (e.g., top 20)
3. **Network graphs**: Visualize important connectivity patterns
4. **Layer-wise comparison**: How attention evolves through layers

---

## 📊 Comparison: BrainGAT vs HCAN

### Architecture Differences

| Aspect | HCAN (Paper) | BrainGAT (Ours) |
|--------|-------------|-----------------|
| **Graph Type** | Heterogeneous | Homogeneous |
| **Node Types** | ROI + Phenotype | ROI only |
| **Graph Scope** | Population-level | Subject-level |
| **Features** | fMRI + Demographics | fMRI only |
| **Attention** | Heterogeneous Conv | Multi-head GAT |
| **Phenotypic Data** | Age, Sex, Site, IQ | NONE ✓ |
| **Data Leakage Risk** | HIGH ⚠️ | NONE ✓ |
| **Interpretability** | Limited | High (attention weights) |
| **Training** | CPU/GPU | GPU-optimized (AMP) |

### Performance Expectations

| Metric | HCAN (with phenotype) | BrainGAT (imaging only) |
|--------|----------------------|------------------------|
| **Expected Accuracy** | 75-85% | 65-75% |
| **Expected AUC** | 0.78-0.88 | 0.68-0.78 |
| **Random Baseline** | 50-55% | 50-55% |
| **Interpretation** | Phenotype boost | Realistic imaging-only |

**Why lower accuracy is expected**:
- No demographic shortcuts (age, sex, IQ)
- No site information (scanner differences)
- Pure imaging signal is subtle for ASD
- **Lower accuracy with proper methodology = better science!**

---

## 🎓 Key Design Decisions

### 1. Why Multi-Head Attention?

**Multiple heads capture diverse patterns**:
- Head 1: Local connectivity (neighboring ROIs)
- Head 2: Long-range connections (distant ROIs)
- Head 3: Default mode network patterns
- Head 4-8: Other specialized patterns

Ensemble effect improves robustness.

### 2. Why 3 Layers?

**Hierarchical feature learning**:
- **Layer 1**: Direct connections (1-hop neighbors)
- **Layer 2**: Indirect connections (2-hop neighbors)
- **Layer 3**: Global network structure (3-hop)

More layers → larger receptive field → global brain patterns.

### 3. Why Mean + Max Readout?

**Complementary information**:
- **Mean pooling**: Average activation (typical patterns)
- **Max pooling**: Peak activation (salient features)
- **Concatenation**: Best of both worlds

### 4. Why Edge Attributes?

**Correlation strength matters**:
- Strong positive correlation (synchronized activity)
- Strong negative correlation (anti-correlated activity)
- Weak correlation (independent activity)

Edge weights provide richer information than binary edges.

---

## 🚀 Implementation Highlights

### Code Structure

```python
# 1. Data Loading
graphs = load_abide_graphs(data_dir, phenotype_file, topk=20)

# 2. Model Initialization
model = BrainGAT(
    in_channels=392,
    hidden_channels=256,
    num_layers=3,
    heads=8,
    dropout=0.3,
    num_classes=2
).to(device)

# 3. Training with GPU optimization
trained_model, history = train_brain_gat(
    model, train_loader, val_loader,
    num_epochs=100, lr=0.0005, patience=20
)

# 4. Evaluation
results = evaluate_gat(trained_model, test_loader, device)

# 5. Interpretation
attention_weights = model.get_attention_weights(sample_graph)
```

### GPU Optimization Features

- **Automatic mixed precision (AMP)**: FP16 operations where safe
- **Gradient accumulation**: Effective batch size scaling
- **CUDA memory management**: Efficient tensor allocation
- **Batch processing**: Vectorized operations

---

## 📚 Theoretical Foundation

### Graph Neural Network Theory

**Message Passing Framework**:

At each layer $l$, for each node $i$:

1. **Message**: $\mathbf{m}_{ij}^{(l)} = M^{(l)}(\mathbf{x}_i^{(l)}, \mathbf{x}_j^{(l)}, e_{ij})$
2. **Aggregation**: $\mathbf{m}_i^{(l)} = \text{AGG}^{(l)}(\{\mathbf{m}_{ij}^{(l)} : j \in \mathcal{N}_i\})$
3. **Update**: $\mathbf{x}_i^{(l+1)} = U^{(l)}(\mathbf{x}_i^{(l)}, \mathbf{m}_i^{(l)})$

**In GAT**:
- Message: Attention-weighted features
- Aggregation: Weighted sum with learned $\alpha_{ij}$
- Update: Linear transform + nonlinearity

### Attention Mechanism Theory

**Self-Attention in Graphs**:

Query-Key-Value framework adapted to graphs:
- **Query**: Node $i$ asking "what to attend to?"
- **Key**: Node $j$ responding "here's my relevance"
- **Value**: Node $j$'s features to aggregate

**Softmax normalization** ensures:
$$
\sum_{j \in \mathcal{N}_i} \alpha_{ij} = 1
$$

Attention weights are **learned**, not fixed!

---

## 🎯 Results Interpretation Guide

### What constitutes "good" performance?

**For ABIDE ASD classification (imaging only)**:

| Accuracy Range | Interpretation |
|---------------|----------------|
| 50-55% | Random chance (no learning) |
| 55-60% | Weak signal detection |
| 60-65% | Moderate performance ✓ |
| 65-70% | Good performance ✓✓ |
| 70-75% | Excellent performance ✓✓✓ |
| 75-80% | Exceptional (rare without phenotype) |
| >80% | **Suspicious - check for leakage!** ⚠️ |

### AUC Score Interpretation

| AUC Range | Interpretation |
|-----------|----------------|
| 0.50 | Random classifier |
| 0.60-0.65 | Weak discriminator |
| 0.65-0.70 | Fair discriminator ✓ |
| 0.70-0.75 | Good discriminator ✓✓ |
| 0.75-0.80 | Strong discriminator ✓✓✓ |
| >0.80 | Excellent (verify no leakage) |

---

## 🔬 Scientific Validity

### Why This Approach is Scientifically Sound

1. **No Information Leakage**:
   - Subject-level splitting (each subject in ONE split)
   - No phenotypic features as input
   - Test set completely held out

2. **Realistic Performance Expectations**:
   - ASD is heterogeneous (multiple subtypes)
   - ABIDE has multi-site variability
   - Imaging signal is subtle
   - 65-75% accuracy is publishable!

3. **Interpretability**:
   - Attention weights show which connections matter
   - Can identify brain networks involved
   - Biological plausibility can be verified

4. **Reproducibility**:
   - Fixed random seeds
   - Documented hyperparameters
   - Version-controlled code
   - Clear data processing pipeline

---

## 🛠️ Potential Improvements

### Short-Term Enhancements

1. **Dynamic Connectivity**:
   - Sliding window graphs (capture temporal dynamics)
   - Multiple graphs per subject

2. **Multi-Resolution Features**:
   - Combine CC400 + CC200 + AAL atlases
   - Different spatial scales

3. **Data Augmentation**:
   - Temporal jittering (add noise to timeseries)
   - Graph augmentation (edge dropout)
   - Mixup for graphs

4. **Ensemble Methods**:
   - Train multiple models with different seeds
   - Average predictions (reduces variance)

### Long-Term Research Directions

1. **Hierarchical Attention**:
   - ROI-level attention (which regions)
   - Network-level attention (which networks)
   - Global attention (whole brain)

2. **Contrastive Learning**:
   - Self-supervised pretraining on unlabeled scans
   - Transfer learning from larger datasets

3. **Explainability Methods**:
   - GNNExplainer: Identify subgraphs important for prediction
   - Integrated gradients: Attribution to input features
   - Attention flow: Track information through layers

4. **Biological Validation**:
   - Correlate attention patterns with known ASD neuroscience
   - Validate findings across independent datasets
   - Link to behavioral phenotypes

---

## 📖 References & Resources

### Key Papers

1. **Graph Attention Networks (GAT)**:
   - Veličković et al. (2018). "Graph Attention Networks." ICLR.
   
2. **HCAN Architecture** (inspiration):
   - Original paper on heterogeneous graph attention for ASD

3. **ABIDE Dataset**:
   - Di Martino et al. (2014). "The Autism Brain Imaging Data Exchange."
   
4. **Graph Neural Networks Survey**:
   - Wu et al. (2020). "A Comprehensive Survey on Graph Neural Networks."

### Implementation Resources

- **PyTorch Geometric**: [https://pytorch-geometric.readthedocs.io/](https://pytorch-geometric.readthedocs.io/)
- **ABIDE Preprocessed**: [http://preprocessed-connectomes-project.org/abide/](http://preprocessed-connectomes-project.org/abide/)

---

## 🏁 Conclusion

BrainGAT represents a **scientifically rigorous** approach to ASD classification using graph neural networks:

✅ **No data leakage** (pure imaging features)  
✅ **Interpretable** (attention weights)  
✅ **GPU-optimized** (mixed precision training)  
✅ **Biologically motivated** (brain connectivity graphs)  
✅ **Reproducible** (documented methodology)

While HCAN achieves higher accuracy by incorporating phenotypic information, BrainGAT demonstrates that **meaningful classification is possible using only brain imaging data**, making it more generalizable and scientifically valid for neuroimaging research.

---

**Document Version**: 1.0  
**Last Updated**: November 3, 2025  
**Author**: GitHub Copilot  
**Project**: ABIDE ASD Classification
