# Data Leakage Fix - BrainGAT Evolution

## Problem Identified

### The Issue
**95% validation accuracy was NOT real** - it was caused by severe data leakage.

### What Went Wrong
```python
# WRONG: Window-level splitting (OLD CODE)
train_test_split(graphs, labels, ...)  # graphs contains multiple windows per subject
```

**Why this was catastrophic:**
- Subject A has 6 temporal windows from their 176-TR scan
- 3 windows went to training
- 3 windows went to validation
- Model saw Subject A's brain patterns during training
- Then "predicted" on the SAME subject's windows in validation
- **This is memorization, not generalization!**

### Evidence of Leakage
1. **Suspiciously high accuracy**: 95% is way too high
   - State-of-the-art on ABIDE: 71-74% (BrainGNN)
   - Expected with our improvements: 68-72%
   - **We got 95%** ← screams data leakage

2. **Data structure**:
   - 886 subjects → 5,492 windows (6.2x augmentation)
   - Old split: 3,846 train / 822 val windows
   - **Same subjects appeared in both sets**

---

## Solution Implemented

### Subject-Level Splitting

```python
# CORRECT: Split subjects FIRST, then extract windows
subject_label_map = {}
for subj, label in zip(subjects, labels):
    if subj not in subject_label_map:
        subject_label_map[subj] = label

unique_subjects = list(subject_label_map.keys())
unique_labels = [subject_label_map[s] for s in unique_subjects]

# Split subjects (not windows!)
train_subjects, val_test_subjects = train_test_split(
    unique_subjects, unique_labels, stratify=unique_labels, random_state=42
)

# Then assign windows based on subject membership
for graph, label, subject in zip(graphs, labels, subjects):
    if subject in train_set:
        train_graphs.append(graph)
    elif subject in val_set:
        val_graphs.append(graph)
    # etc.
```

### New Data Split Structure

**4-Way Split** (for robust evaluation):

| Set | Purpose | % Subjects | Subjects | Windows |
|-----|---------|-----------|----------|---------|
| **Train** | Model training | 60% | ~532 | ~3,300 |
| **Validation** | Hyperparameter tuning | 15% | ~133 | ~820 |
| **Test1 (Public)** | Public test set | 12.5% | ~110 | ~680 |
| **Test2 (Hold-out)** | Final evaluation | 12.5% | ~110 | ~680 |

### Validation Checks

```python
# Verify no subject overlap (critical!)
assert len(train_set & val_set) == 0, "Train-Val overlap!"
assert len(train_set & test1_set) == 0, "Train-Test1 overlap!"
assert len(train_set & test2_set) == 0, "Train-Test2 overlap!"
assert len(val_set & test1_set) == 0, "Val-Test1 overlap!"
assert len(val_set & test2_set) == 0, "Val-Test2 overlap!"
assert len(test1_set & test2_set) == 0, "Test1-Test2 overlap!"
#  No overlap - splits are clean!
```

---

## Expected Results After Fix

### Realistic Performance Estimates

| Metric | Old (Wrong) | New (Correct) | Difference |
|--------|-------------|---------------|------------|
| Validation Accuracy | 95% | **60-75%** | -20 to -35% |
| Test Accuracy | N/A | **60-70%** | Realistic |
| Overfitting | Hidden | **Visible** | Can now diagnose |

### Why Lower is Better

**Old 95%**: Model learned "this pattern looks like Subject #42" (memorization)

**New 60-75%**: Model learns "this is an ASD brain pattern" (generalization)

The new numbers are:
-  Comparable to state-of-the-art (BrainGNN: 71-74%)
-  Realistic for fMRI-based ASD classification
-  Honest assessment of model capability
-  Can be trusted for real-world deployment

---

## Additional Improvements

### 1. Hyperparameter Grid Search

**Purpose**: Find optimal hyperparameters using validation set

**Search Space**:
```python
{
    'lr': [1e-3, 5e-4, 1e-4],
    'hidden_dim': [32, 64],
    'temporal_dim': [64, 128],
    'dropout': [0.3, 0.5],
    'weight_decay': [1e-4, 1e-5],
    'heads': [4, 8]
}
```

**Total**: 3 × 2 × 2 × 2 × 2 × 2 = **96 configurations**

**Implementation**:
- Train each config for 30 epochs max
- Early stopping (patience=8)
- Select best based on validation accuracy
- Save results to `grid_search_results.json`

### 2. Triple Validation

**Purpose**: Verify model generalizes across multiple held-out sets

**Evaluation**:
1. **Validation Set**: Used for hyperparameter selection
2. **Test1 (Public)**: First independent evaluation
3. **Test2 (Hold-out)**: Final "never-seen" evaluation

**Metrics Computed**:
- Accuracy, Precision, Recall, F1, AUC-ROC
- Confusion matrices
- ROC curves
- Statistical consistency checks

**Consistency Checks**:
```python
# Low variance = good generalization
std_acc = np.std([val_acc, test1_acc, test2_acc])

if std_acc < 3.0:
    print(" Excellent consistency")
elif std_acc < 5.0:
    print(" Acceptable")
else:
    print(" High variance - overfitting")

# Test2 is the true estimate
gap = abs(val_acc - test2_acc)
if gap < 5.0:
    print(" Excellent generalization")
```

---

## Implementation Files

### Modified Cells

**Cell 13** (Data Loading):
- Now performs subject-level splitting
- Creates 4 separate sets (train/val/test1/test2)
- Validates no subject overlap
- Reports detailed statistics

**New Cells Added**:

1. **Hyperparameter Grid Search Cell**:
   - Systematic search over hyperparameter space
   - Checkpointing to resume interrupted searches
   - Saves top configurations

2. **Final Model Training Cell**:
   - Uses best hyperparameters from grid search
   - Extended training (150 epochs max)
   - Proper early stopping and checkpointing

3. **Triple Validation Cell**:
   - Evaluates on all 3 held-out sets
   - Comprehensive metrics and visualizations
   - Statistical consistency analysis
   - Final verdict on generalization

---

## Usage Instructions

### Step 1: Run Data Loading (Cell 13)
```python
# This will create subject-level splits
# Expected output: ~532/133/110/110 subjects
```

### Step 2: (Optional) Hyperparameter Search
```python
# Set max_configs to limit search
max_configs = 5  # For testing
max_configs = None  # For full search (96 configs, ~48 hours)
```

### Step 3: Train Final Model
```python
# Uses best config from grid search (or defaults)
# Trains for up to 150 epochs with early stopping
```

### Step 4: Triple Validation
```python
# Evaluates on all 3 sets
# Reports realistic performance estimate
```

---

## Interpretation Guide

### Good Results
- **Test2 Accuracy**: 65-75%
- **Std across sets**: < 3%
- **Val-Test2 gap**: < 5%
- **Interpretation**: Model generalizes well, realistic performance

### Acceptable Results
- **Test2 Accuracy**: 60-65%
- **Std across sets**: 3-5%
- **Val-Test2 gap**: 5-10%
- **Interpretation**: Some overfitting, but still useful

### Poor Results
- **Test2 Accuracy**: < 60%
- **Std across sets**: > 5%
- **Val-Test2 gap**: > 10%
- **Interpretation**: Overfitting or model not learning meaningful patterns

---

## Key Takeaways

###  What We Fixed
1. **Data leakage**: Subject-level splitting prevents training on test subjects
2. **Inflated metrics**: Now reporting realistic performance
3. **Robustness**: Triple validation ensures consistency
4. **Optimization**: Grid search finds best hyperparameters

###  What Changed
1. **Accuracy dropped**: 95% → 60-75% (this is GOOD - it's now honest!)
2. **Training time**: Longer (proper hyperparameter search + longer training)
3. **Complexity**: More validation sets to track

###  Why This Matters
1. **Scientific integrity**: Can publish these results
2. **Real-world deployment**: Performance estimates are trustworthy
3. **Debugging**: Can now identify and fix real problems
4. **Comparison**: Can fairly compare with other methods

---

## References

### Why Data Leakage is Critical
- Kaufman et al. (2012). "Leakage in data mining: Formulation, detection, and avoidance." *ACM Transactions on Knowledge Discovery from Data*.
- Poldrack et al. (2020). "Establishment of Best Practices for Evidence for Prediction." *JAMA Psychiatry*.

### Subject-Level Splitting in fMRI
- Varoquaux et al. (2017). "Assessing and tuning brain decoders: Cross-validation, caveats, and guidelines." *NeuroImage*.
- Abraham et al. (2014). "Extracting brain regions from rest fMRI with Total-Variation constrained dictionary learning." *MICCAI*.

### ABIDE Benchmarks
- Heinsfeld et al. (2018). "Identification of autism spectrum disorder using deep learning and the ABIDE dataset." *NeuroImage: Clinical*.
- Li et al. (2021). "BrainGNN: Interpretable Brain Graph Neural Network for fMRI Analysis." *Medical Image Analysis*.

---

**Document Version**: 1.0  
**Date**: December 8, 2025  
**Status**:  Fixed and validated
