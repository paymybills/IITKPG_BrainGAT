# BrainGAT Evolution: Final Training Protocol
# ============================================
# This script implements the complete training protocol from the implementation plan.
# Copy these blocks into your BrainGAT_Evolution.ipynb notebook cells.
#
# BLOCKS TO REPLACE/ADD:
# - Block 1: Replace the data loading functions cell
# - Block 2: Replace the data loading and splitting cell  
# - Block 3: Keep your existing model definition
# - Block 4: Replace the training cell
# - Block 5: Add NEW cell after training for subject-level evaluation
# - Block 6: Add NEW cell at the end to save outputs

# ==============================================================================
# BLOCK 1: Updated Data Loading (stride=40 for 50% overlap)
# ==============================================================================
# REPLACE your existing `load_timeseries_1d`, `extract_temporal_windows`, and 
# related functions with this block.

import numpy as np
import pandas as pd
import glob
import torch
from torch_geometric.data import Data
from pathlib import Path

def load_timeseries_1d(path: str) -> np.ndarray:
    """Load .1D file as T x N array"""
    arr = np.loadtxt(path)
    if arr.shape[1] == 392:
        pass
    elif arr.shape[0] == 392:
        arr = arr.T
    else:
        raise ValueError(f"Expected 392 ROIs, got shape {arr.shape}")
    return arr

def extract_temporal_windows(timeseries: np.ndarray, window_length: int = 80, stride: int = 40):
    """Extract sliding windows with 50% overlap (stride=40)"""
    T, N = timeseries.shape
    if T < window_length:
        return []
    windows = []
    for start in range(0, T - window_length + 1, stride):
        end = start + window_length
        windows.append(timeseries[start:end, :])
    return windows

def compute_partial_correlation(ts: np.ndarray) -> np.ndarray:
    """Compute partial correlation matrix"""
    corr = np.corrcoef(ts, rowvar=False)
    try:
        corr_reg = corr + np.eye(corr.shape[0]) * 1e-6
        precision = np.linalg.inv(corr_reg)
        diag = np.sqrt(np.abs(np.diag(precision)))
        diag = np.where(diag < 1e-10, 1e-10, diag)
        partial = -precision / np.outer(diag, diag)
        np.fill_diagonal(partial, 1.0)
        partial = np.clip(partial, -1.0, 1.0)
        partial = np.nan_to_num(partial, nan=0.0)
        return partial
    except np.linalg.LinAlgError:
        return np.zeros_like(corr)

def compute_mutual_information(ts: np.ndarray) -> np.ndarray:
    """Compute MI matrix via Spearman correlation"""
    from scipy.stats import spearmanr
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        mi_matrix, _ = spearmanr(ts, axis=0)
    mi_matrix = np.abs(mi_matrix)
    mi_matrix = np.nan_to_num(mi_matrix, nan=0.0)
    max_val = np.max(mi_matrix)
    if max_val > 1e-8:
        mi_matrix = mi_matrix / max_val
    return mi_matrix

def compute_phase_synchrony(ts: np.ndarray) -> np.ndarray:
    """Compute phase synchronization via Hilbert transform"""
    from scipy.signal import hilbert
    analytic = hilbert(ts, axis=0)
    phases = np.angle(analytic)
    phase_diff = phases[:, :, np.newaxis] - phases[:, np.newaxis, :]
    plv = np.abs(np.mean(np.exp(1j * phase_diff), axis=0))
    return plv

def build_multiscale_graphs(ts: np.ndarray, k_values=[10, 30, 100]):
    """Build multiple graphs at different scales"""
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        corr = np.corrcoef(ts, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        corr = np.clip(corr, -1.0, 1.0)
        partial = compute_partial_correlation(ts)
        ts_down = ts[::4, :]
        mi = compute_mutual_information(ts_down)
        plv = compute_phase_synchrony(ts_down)
    
    N = corr.shape[0]
    graphs = []
    
    for k in k_values:
        np.fill_diagonal(corr, 0.0)
        absC = np.abs(corr)
        k_eff = min(k, max(1, N - 1))
        edge_src, edge_dst, edge_feats = [], [], []
        
        for i in range(N):
            nbrs = np.argpartition(absC[i], -k_eff)[-k_eff:]
            for j in nbrs:
                if i != j:
                    edge_src.append(i)
                    edge_dst.append(j)
                    edge_feats.append([corr[i, j], partial[i, j], mi[i, j], plv[i, j]])
        
        # Make bidirectional and deduplicate
        pairs = {}
        for idx in range(len(edge_src)):
            s, d = edge_src[idx], edge_dst[idx]
            key = (min(s, d), max(s, d))
            if key not in pairs:
                pairs[key] = edge_feats[idx]
        
        final_edges = list(pairs.keys())
        final_src = [e[0] for e in final_edges] + [e[1] for e in final_edges]
        final_dst = [e[1] for e in final_edges] + [e[0] for e in final_edges]
        final_attr = [pairs[e] for e in final_edges] * 2
        
        edge_index = torch.tensor([final_src, final_dst], dtype=torch.long)
        edge_attr = torch.tensor(final_attr, dtype=torch.float)
        graphs.append((edge_index, edge_attr))
    
    return graphs, corr

def graph_from_timeseries_enhanced(timeseries: np.ndarray, k_values=[10, 30, 100]):
    """Build PyG Data object from timeseries window"""
    ts_tensor = torch.tensor(timeseries.T, dtype=torch.float)  # (N, T)
    graphs, corr = build_multiscale_graphs(timeseries, k_values)
    x = torch.tensor(corr, dtype=torch.float)
    
    data = Data(x=x, timeseries=ts_tensor)
    for i, (edge_idx, edge_attr) in enumerate(graphs):
        setattr(data, f'edge_index_{i}', edge_idx)
        setattr(data, f'edge_attr_{i}', edge_attr)
    return data

print("Data loading functions defined (stride=40)")


# ==============================================================================
# BLOCK 2: Subject-Level Data Loading & Splitting
# ==============================================================================
# REPLACE your data loading cell with this.

import os
from sklearn.model_selection import train_test_split

data_dir = 'abide_data/Outputs/cpac/nofilt_noglobal/rois_cc400/'
phenotype_file = 'Phenotypic_V1_0b_preprocessed1.csv'

# Download if missing
if not os.path.exists(phenotype_file):
    import urllib.request as request
    pheno_url = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv'
    request.urlretrieve(pheno_url, phenotype_file)
    print("Downloaded phenotype file")

pheno_df = pd.read_csv(phenotype_file)
roi_files = sorted(glob.glob(f'{data_dir}/*.1D'))

print("="*60)
print("STEP 1: LOAD FULL TIMESERIES PER SUBJECT (NO WINDOWS YET)")
print("="*60)

# Site mapping
site_map = {
    'MaxMun': 'MAX_MUN', 'Leuven_1': 'LEUVEN_1', 'Leuven_2': 'LEUVEN_2',
    'UCLA_1': 'UCLA_1', 'UCLA_2': 'UCLA_2', 'UM_1': 'UM_1', 'UM_2': 'UM_2',
    'Trinity': 'TRINITY', 'Yale': 'YALE', 'Olin': 'OLIN', 'OHSU': 'OHSU',
    'SBL': 'SBL', 'SDSU': 'SDSU', 'Stanford': 'STANFORD', 'Caltech': 'CALTECH',
    'CMU': 'CMU', 'KKI': 'KKI', 'NYU': 'NYU', 'Pitt': 'PITT', 'USM': 'USM'
}

# Load raw timeseries per subject (NO windowing yet)
subject_data = {}  # {subject_id: {'timeseries': ts, 'label': label}}

for idx, file_path in enumerate(roi_files):
    if idx % 100 == 0:
        print(f"   Loading subject {idx+1}/{len(roi_files)}...")
    try:
        filename = Path(file_path).stem
        parts = filename.replace('_rois_cc400', '').split('_')
        if len(parts) < 2:
            continue
        
        site = parts[0]
        subject_id_idx = 1
        if len(parts) > 2 and parts[1].isdigit() and len(parts[1]) == 1:
            site = f"{parts[0]}_{parts[1]}"
            subject_id_idx = 2
        
        if site in site_map:
            site = site_map[site]
        elif site.upper() in site_map.values():
            site = site.upper()
        
        subject_id = None
        for part in parts[subject_id_idx:]:
            try:
                subject_id = int(part)
                break
            except ValueError:
                continue
        
        if subject_id is None:
            continue
        
        subject_row = pheno_df[(pheno_df['SITE_ID'] == site) & (pheno_df['SUB_ID'] == subject_id)]
        if subject_row.empty:
            continue
        
        dx_group = subject_row['DX_GROUP'].values[0]
        if dx_group not in [1, 2]:
            continue
        
        ts = load_timeseries_1d(file_path)
        if np.any(np.isnan(ts)) or np.any(np.isinf(ts)):
            continue
        if np.any(np.std(ts, axis=0) < 1e-10):
            continue
        if ts.shape[0] < 80:  # Too short for windowing
            continue
        
        subj_key = f"{site}_{subject_id}"
        subject_data[subj_key] = {
            'timeseries': ts,
            'label': dx_group - 1  # 0=Control, 1=ASD
        }
    except Exception as e:
        continue

print(f"\nLoaded {len(subject_data)} subjects")
asd_count = sum(1 for v in subject_data.values() if v['label'] == 1)
ctrl_count = len(subject_data) - asd_count
print(f"   ASD: {asd_count} | Control: {ctrl_count}")

# STEP 2: SPLIT AT SUBJECT LEVEL
print("\n" + "="*60)
print("STEP 2: SPLIT SUBJECTS (60% train, 20% val, 20% test)")
print("="*60)

subjects = list(subject_data.keys())
labels = [subject_data[s]['label'] for s in subjects]

train_subjects, temp_subjects, _, temp_labels = train_test_split(
    subjects, labels, test_size=0.4, random_state=42, stratify=labels
)
val_subjects, test_subjects, _, _ = train_test_split(
    temp_subjects, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"   Train: {len(train_subjects)} subjects")
print(f"   Val:   {len(val_subjects)} subjects")
print(f"   Test:  {len(test_subjects)} subjects")

# Save split indices for reproducibility
split_info = {
    'train_subjects': train_subjects,
    'val_subjects': val_subjects,
    'test_subjects': test_subjects
}
import json
with open('subject_splits.json', 'w') as f:
    json.dump(split_info, f, indent=2)
print("   Saved split indices to subject_splits.json")

# STEP 3: GENERATE WINDOWS AFTER SPLITTING
print("\n" + "="*60)
print("STEP 3: GENERATE WINDOWS (stride=40, 50% overlap)")
print("="*60)

WINDOW_LENGTH = 80
STRIDE = 40
K_VALUES = [10, 30, 100]

def generate_windows_for_subjects(subject_list, subject_data):
    """Generate windowed graphs for a list of subjects"""
    graphs = []
    labels = []
    subject_ids = []
    
    for subj in subject_list:
        ts = subject_data[subj]['timeseries']
        label = subject_data[subj]['label']
        windows = extract_temporal_windows(ts, WINDOW_LENGTH, STRIDE)
        
        for window in windows:
            graph = graph_from_timeseries_enhanced(window, k_values=K_VALUES)
            graph.y = torch.tensor([label], dtype=torch.long)
            graphs.append(graph)
            labels.append(label)
            subject_ids.append(subj)
    
    return graphs, labels, subject_ids

train_graphs, train_labels, train_subject_ids = generate_windows_for_subjects(train_subjects, subject_data)
val_graphs, val_labels, val_subject_ids = generate_windows_for_subjects(val_subjects, subject_data)
test_graphs, test_labels, test_subject_ids = generate_windows_for_subjects(test_subjects, subject_data)

print(f"   Train: {len(train_graphs)} windows from {len(train_subjects)} subjects")
print(f"   Val:   {len(val_graphs)} windows from {len(val_subjects)} subjects")
print(f"   Test:  {len(test_graphs)} windows from {len(test_subjects)} subjects")

# Create DataLoaders
from torch_geometric.loader import DataLoader as PyGDataLoader
BATCH_SIZE = 4
train_loader = PyGDataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
val_loader = PyGDataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
test_loader = PyGDataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

print("\nDataLoaders ready!")


# ==============================================================================
# BLOCK 3: Model Definition
# ==============================================================================
# KEEP YOUR EXISTING MODEL - No changes needed to TemporalSpatialBrainGAT


# ==============================================================================
# BLOCK 4: Training with Label Smoothing & Protocol Settings
# ==============================================================================
# REPLACE your training cell with this.

import gc
import json
import torch
import matplotlib.pyplot as plt

# Assumes 'device' is already defined (e.g., device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Memory cleanup
gc.collect()
torch.cuda.empty_cache()

# Configuration (from your best search result or defaults)
config = {
    'lr': 5e-4,
    'hidden_dim': 32,
    'temporal_dim': 64,
    'dropout': 0.5,
    'weight_decay': 1e-4,
    'heads': 4,
    'label_smoothing': 0.1  # Protocol requirement
}

print("="*60)
print("FINAL TRAINING (PROTOCOL SETTINGS)")
print("="*60)
print(f"Config: {json.dumps(config, indent=2)}")

# Initialize model
# NOTE: Assumes TemporalSpatialBrainGAT is already defined
model = TemporalSpatialBrainGAT(
    in_channels=392,
    hidden_dim=config['hidden_dim'],
    temporal_dim=config['temporal_dim'],
    num_scales=3,
    heads=config['heads'],
    dropout=config['dropout'],
    num_classes=2
).to(device)

params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")

# Class weights
num_ctrl = train_labels.count(0)
num_asd = train_labels.count(1)
w_ctrl = len(train_labels) / (2 * num_ctrl)
w_asd = len(train_labels) / (2 * num_asd)
class_weights = torch.tensor([w_ctrl, w_asd], dtype=torch.float).to(device)
print(f"Class weights: Control={w_ctrl:.2f}, ASD={w_asd:.2f}")

# Loss with label smoothing (PROTOCOL REQUIREMENT)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config['label_smoothing'])
print(f"Label smoothing: {config['label_smoothing']}")

optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
scaler = torch.amp.GradScaler('cuda')

# Checkpointing
CHECKPOINT_PATH = 'braingat_checkpoint.pth'
BEST_MODEL_PATH = 'braingat_best.pth'
CONFIG_PATH = 'training_config.json'

# Save config
with open(CONFIG_PATH, 'w') as f:
    json.dump({**config, 'window_length': WINDOW_LENGTH, 'stride': STRIDE}, f, indent=2)
print(f"Saved config to {CONFIG_PATH}")

# Training settings (PROTOCOL REQUIREMENTS)
TOTAL_EPOCHS = 120
PATIENCE = 20
ACCUM_STEPS = 8

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_acc = 0.0
best_val_loss = float('inf')
patience_counter = 0
start_epoch = 1

# Resume if checkpoint exists
if os.path.exists(CHECKPOINT_PATH):
    print(f"\nResuming from {CHECKPOINT_PATH}...")
    ckpt = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_val_acc = ckpt['best_val_acc']
    history = ckpt['history']
    patience_counter = ckpt.get('patience_counter', 0)
    print(f"   Starting from epoch {start_epoch}, best acc: {best_val_acc:.2f}%")
else:
    print("\nStarting fresh training...")

print(f"\nConfig: {TOTAL_EPOCHS} epochs, patience={PATIENCE}, batch={BATCH_SIZE}x{ACCUM_STEPS}")
print("="*60)

try:
    for epoch in range(start_epoch, TOTAL_EPOCHS + 1):
        # TRAIN
        model.train()
        total_loss, correct, total = 0, 0, 0
        optimizer.zero_grad()
        
        for i, data in enumerate(train_loader):
            data = data.to(device)
            with torch.amp.autocast('cuda'):
                out = model(data)
                loss = criterion(out, data.y) / ACCUM_STEPS
            
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * ACCUM_STEPS * data.num_graphs
            correct += (out.argmax(dim=1) == data.y).sum().item()
            total += data.num_graphs
        
        train_loss = total_loss / total
        train_acc = 100.0 * correct / total
        
        # VALIDATE
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                with torch.amp.autocast('cuda'):
                    out = model(data)
                    loss = criterion(out, data.y)
                if not torch.isnan(loss):
                    val_loss += loss.item() * data.num_graphs
                val_correct += (out.argmax(dim=1) == data.y).sum().item()
                val_total += data.num_graphs
        
        val_loss = val_loss / val_total if val_total > 0 else float('inf')
        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
        
        scheduler.step(val_loss)
        
        # History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Best model check
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            patience_counter = 0
        
        # Early stopping (loss-based)
        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc': best_val_acc,
            'history': history,
            'patience_counter': patience_counter
        }, CHECKPOINT_PATH)
        
        # Print progress
        if epoch % 5 == 0 or is_best:
            status = "BEST!" if is_best else f"(p:{patience_counter}/{PATIENCE})"
            print(f"E{epoch:03d} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e} {status}")
        
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break
        
        if epoch % 20 == 0:
            torch.cuda.empty_cache()

except KeyboardInterrupt:
    print("\nInterrupted - checkpoint saved!")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
print("="*60)

# Load best model
if os.path.exists(BEST_MODEL_PATH):
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    print(f"Loaded best model from {BEST_MODEL_PATH}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history['train_loss'], label='Train')
axes[0].plot(history['val_loss'], label='Val')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(history['train_acc'], label='Train')
axes[1].plot(history['val_acc'], label='Val')
axes[1].set_title(f'Accuracy (Best: {best_val_acc:.1f}%)'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()


# ==============================================================================
# BLOCK 5: Subject-Level Evaluation (CRITICAL)
# ==============================================================================
# ADD this as a NEW CELL after training.

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import numpy as np

def evaluate_subject_level(model, loader, subject_ids, subject_data, set_name="Test"):
    """
    Aggregate window predictions to subject level.
    Returns subject-level metrics (not window-level).
    """
    model.eval()
    
    # Collect window-level predictions
    window_preds = []
    window_probs = []
    window_labels = []
    window_subjects = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)
            with torch.amp.autocast('cuda'):
                out = model(data)
            
            probs = torch.softmax(out, dim=1)
            preds = out.argmax(dim=1)
            
            # Get subject IDs for this batch
            start_idx = batch_idx * BATCH_SIZE
            end_idx = start_idx + data.num_graphs
            batch_subjects = subject_ids[start_idx:end_idx]
            
            window_preds.extend(preds.cpu().numpy())
            window_probs.extend(probs[:, 1].cpu().numpy())  # P(ASD)
            window_labels.extend(data.y.cpu().numpy())
            window_subjects.extend(batch_subjects)
    
    # Aggregate to subject level
    subject_preds = {}
    subject_probs = {}
    subject_labels = {}
    
    for subj, pred, prob, label in zip(window_subjects, window_preds, window_probs, window_labels):
        if subj not in subject_preds:
            subject_preds[subj] = []
            subject_probs[subj] = []
            subject_labels[subj] = label
        subject_preds[subj].append(pred)
        subject_probs[subj].append(prob)
    
    # Compute subject-level predictions (mean probability)
    final_subjects = list(subject_preds.keys())
    final_labels = [subject_labels[s] for s in final_subjects]
    final_probs = [np.mean(subject_probs[s]) for s in final_subjects]
    final_preds = [1 if p > 0.5 else 0 for p in final_probs]
    
    # Metrics
    acc = accuracy_score(final_labels, final_preds) * 100
    try:
        auc = roc_auc_score(final_labels, final_probs)
    except:
        auc = 0.5
    
    cm = confusion_matrix(final_labels, final_preds)
    report = classification_report(final_labels, final_preds, target_names=['Control', 'ASD'], digits=3)
    
    print(f"\n{'='*60}")
    print(f"SUBJECT-LEVEL RESULTS: {set_name}")
    print(f"{'='*60}")
    print(f"Subjects evaluated: {len(final_subjects)}")
    print(f"Subject-Level Accuracy: {acc:.2f}%")
    print(f"Subject-Level AUC: {auc:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"              Pred Control  Pred ASD")
    print(f"True Control      {cm[0,0]:4d}        {cm[0,1]:4d}")
    print(f"True ASD          {cm[1,0]:4d}        {cm[1,1]:4d}")
    print(f"\nClassification Report:")
    print(report)
    
    return {
        'accuracy': acc,
        'auc': auc,
        'confusion_matrix': cm,
        'subjects': final_subjects,
        'predictions': final_preds,
        'probabilities': final_probs,
        'labels': final_labels
    }

# Evaluate on validation set
print("\n" + "="*60)
print("SUBJECT-LEVEL EVALUATION")
print("="*60)

val_results = evaluate_subject_level(model, val_loader, val_subject_ids, subject_data, "Validation")
test_results = evaluate_subject_level(model, test_loader, test_subject_ids, subject_data, "Test")

# Save results
results_summary = {
    'validation': {'accuracy': val_results['accuracy'], 'auc': val_results['auc']},
    'test': {'accuracy': test_results['accuracy'], 'auc': test_results['auc']}
}
with open('subject_level_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)
print(f"\nSaved results to subject_level_results.json")


# ==============================================================================
# BLOCK 6: Save All Outputs
# ==============================================================================
# ADD this as the FINAL CELL.

import shutil

print("="*60)
print("SAVING ALL OUTPUTS")
print("="*60)

# Copy to Kaggle working directory (for persistence)
output_files = [
    'braingat_best.pth',
    'braingat_checkpoint.pth',
    'training_config.json',
    'subject_splits.json',
    'subject_level_results.json',
    'training_curves.png'
]

for f in output_files:
    if os.path.exists(f):
        try:
            shutil.copy(f, f'/kaggle/working/{f}')
            print(f"Saved {f}")
        except:
            print(f"   {f} (local only)")
    else:
        print(f"   {f} not found")

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Window params: length={WINDOW_LENGTH}, stride={STRIDE}")
print(f"Best validation accuracy (window-level): {best_val_acc:.2f}%")
print(f"Subject-level validation accuracy: {val_results['accuracy']:.2f}%")
print(f"Subject-level test accuracy: {test_results['accuracy']:.2f}%")
print(f"Subject-level test AUC: {test_results['auc']:.3f}")
