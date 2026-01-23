import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Check for CUDA/GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data Loading Functions
def load_timeseries_1d(path: str) -> np.ndarray:
    """Load .1D file as T×N array"""
    arr = np.loadtxt(path)
    return arr

def corr_matrix(timeseries: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation matrix"""
    C = np.corrcoef(timeseries, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    C = np.clip(C, -1.0, 1.0)
    return C

def build_sparse_edges_from_corr(C: np.ndarray, k: int = 20):
    """Create sparse edge_index via top-k correlations"""
    N = C.shape[0]
    np.fill_diagonal(C, 0.0)
    
    idx_src, idx_dst, weights = [], [], []
    absC = np.abs(C)
    k_eff = min(k, max(1, N - 1))
    
    for i in range(N):
        nbrs = np.argpartition(absC[i], -k_eff)[-k_eff:]
        for j in nbrs:
            if i != j:
                idx_src.append(i)
                idx_dst.append(j)
                weights.append(C[i, j])
    
    # Make undirected
    idx_all = np.concatenate([np.vstack([idx_src, idx_dst]), 
                              np.vstack([idx_dst, idx_src])], axis=1)
    w_all = np.array(weights + weights, dtype=np.float32)
    
    # Deduplicate
    pairs = set()
    uniq_src, uniq_dst, uniq_w = [], [], []
    for (s, d), w in zip(idx_all.T, w_all):
        key = (int(s), int(d))
        if key not in pairs and s != d:
            pairs.add(key)
            uniq_src.append(s)
            uniq_dst.append(d)
            uniq_w.append(w)
    
    edge_index = torch.tensor([uniq_src, uniq_dst], dtype=torch.long)
    edge_attr = torch.tensor(uniq_w, dtype=torch.float)
    return edge_index, edge_attr

def graph_from_timeseries(timeseries: np.ndarray, topk: int = 20):
    """Build PyG Data from fMRI timeseries"""
    C = corr_matrix(timeseries)
    N = C.shape[0]
    
    # Node features: correlation vectors (purely imaging-derived)
    x = torch.tensor(C, dtype=torch.float)
    
    # Sparse edges
    edge_index, edge_attr = build_sparse_edges_from_corr(C.copy(), k=topk)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def load_abide_graphs(data_dir, phenotype_file, topk=20):
    """Load ABIDE subjects as graphs (NO PHENOTYPIC FEATURES!)"""
    pheno_df = pd.read_csv(phenotype_file)
    roi_files = sorted(glob.glob(f'{data_dir}/*.1D'))
    
    print(f"Loading ABIDE data...")
    print(f"   Phenotype CSV: {len(pheno_df)} subjects")
    print(f"   .1D files found: {len(roi_files)}")
    
    graphs, labels, subjects = [], [], []
    
    # Site mapping to handle naming discrepancies
    site_map = {
        'MaxMun': 'MAX_MUN',
        'Leuven_1': 'LEUVEN_1',
        'Leuven_2': 'LEUVEN_2',
        'UCLA_1': 'UCLA_1',
        'UCLA_2': 'UCLA_2',
        'UM_1': 'UM_1',
        'UM_2': 'UM_2',
        'Trinity': 'TRINITY',
        'Yale': 'YALE',
        'Olin': 'OLIN',
        'OHSU': 'OHSU',
        'SBL': 'SBL',
        'SDSU': 'SDSU',
        'Stanford': 'STANFORD',
        'Caltech': 'CALTECH',
        'CMU': 'CMU',
        'KKI': 'KKI',
        'NYU': 'NYU',
        'Pitt': 'PITT',
        'USM': 'USM'
    }
    
    for file_path in roi_files:
        try:
            filename = Path(file_path).stem
            parts = filename.replace('_rois_cc400', '').split('_')
            
            if len(parts) < 2:
                continue
            
            # Robust site parsing
            site = parts[0]
            subject_id_idx = 1
            
            # Check for multi-part site names (e.g., Leuven_1)
            if len(parts) > 2 and parts[1].isdigit() and len(parts[1]) == 1:
                 site = f"{parts[0]}_{parts[1]}"
                 subject_id_idx = 2
            
            # Map site name
            if site in site_map:
                site = site_map[site]
            elif site.upper() in site_map.values():
                site = site.upper()
                
            # Find subject ID (first numeric part after site)
            subject_id = None
            for part in parts[subject_id_idx:]:
                try:
                    subject_id = int(part)
                    break
                except ValueError:
                    continue
            
            if subject_id is None:
                continue
            
            # Match to phenotype for LABEL ONLY
            subject_row = pheno_df[
                (pheno_df['SITE_ID'] == site) & 
                (pheno_df['SUB_ID'] == subject_id)
            ]
            
            if not subject_row.empty:
                dx_group = subject_row['DX_GROUP'].values[0]
                
                if dx_group in [1, 2]:
                    ts = load_timeseries_1d(file_path)
                    graph = graph_from_timeseries(ts, topk=topk)
                    graph.y = torch.tensor([dx_group - 1], dtype=torch.long)
                    
                    graphs.append(graph)
                    labels.append(dx_group - 1)
                    subjects.append(f"{site}_{subject_id}")
        except Exception as e:
            continue
    
    print(f"\nLoaded {len(graphs)} subjects")
    print(f"  - ASD: {labels.count(1)} | Control: {labels.count(0)}")
    if graphs:
        print(f"  - Graph: {graphs[0].x.shape[0]} nodes, {graphs[0].edge_index.shape[1]} edges")
        print(f"  - Features: {graphs[0].x.shape[1]}-dim correlation vectors")
    
    return graphs, labels, subjects

# Model Architecture
class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.3, concat=True):
        super().__init__()
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            concat=concat,
            edge_dim=1
        )
        self.bn = nn.BatchNorm1d(out_channels * heads if concat else out_channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_attr=None):
        x = self.gat(x, edge_index, edge_attr=edge_attr)
        x = self.bn(x)
        x = F.elu(x)
        x = self.dropout(x)
        return x

class BrainGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, num_layers=3, heads=8, dropout=0.3, num_classes=2):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        self.convs.append(
            MultiHeadGATLayer(in_channels, hidden_channels, heads, dropout, concat=True)
        )
        
        for _ in range(num_layers - 2):
            self.convs.append(
                MultiHeadGATLayer(hidden_channels * heads, hidden_channels, heads, dropout, concat=True)
            )
        
        self.convs.append(
            MultiHeadGATLayer(hidden_channels * heads, hidden_channels, heads, dropout, concat=False)
        )
        
        self.readout_dim = hidden_channels * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(self.readout_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels // 2, num_classes)
        )
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        graph_features = torch.cat([x_mean, x_max], dim=-1)
        
        out = self.classifier(graph_features)
        return out

if __name__ == "__main__":
    workspace_root = '/home/moew/Documents/ABIDE'
    data_dir = os.path.join(workspace_root, 'abide_data/Outputs/cpac/nofilt_noglobal/rois_cc400/')
    phenotype_file = os.path.join(workspace_root, 'Phenotypic_V1_0b_preprocessed1.csv')
    
    # Load Data
    graphs, labels, subjects = load_abide_graphs(data_dir, phenotype_file, topk=20)
    
    if not graphs:
        print("No graphs loaded. Exiting.")
        exit(1)

    # Split Data
    train_val_graphs, test_graphs, train_val_labels, test_labels = train_test_split(
        graphs, labels, test_size=0.15, random_state=42, stratify=labels
    )

    train_graphs, val_graphs, train_labels, val_labels = train_test_split(
        train_val_graphs, train_val_labels, test_size=0.176, random_state=42, stratify=train_val_labels
    )

    print(f"\nData split: {len(train_graphs)}/{len(val_graphs)}/{len(test_graphs)} (train/val/test)")
    
    # Ultra Low Memory Config
    print("\nUltra Low Memory Mode")
    train_loader = PyGDataLoader(train_graphs, batch_size=4, shuffle=True)
    val_loader = PyGDataLoader(val_graphs, batch_size=4, shuffle=False)
    test_loader = PyGDataLoader(test_graphs, batch_size=4, shuffle=False)
    
    sample_graph = train_graphs[0]
    in_features = sample_graph.x.size(1)
    
    model = BrainGAT(
        in_channels=in_features,
        hidden_channels=64,
        num_layers=3,
        heads=4,
        dropout=0.3,
        num_classes=2
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    # --- Checkpointing & Early Stopping ---
    class EarlyStopping:
        def __init__(self, patience=10, min_delta=0):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = None
            self.early_stop = False

        def __call__(self, val_loss):
            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss > self.best_loss - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_loss = val_loss
                self.counter = 0

    class ModelCheckpoint:
        def __init__(self, filepath='models/best_model.pth'):
            self.filepath = filepath
            self.best_acc = 0.0
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        def __call__(self, model, val_acc, epoch):
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(model.state_dict(), self.filepath)
                print(f"    New best model saved! (Acc: {val_acc:.2f}%)")
            
            # Also save latest checkpoint for crash recovery
            latest_path = self.filepath.replace('best_model', 'latest_checkpoint')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': self.best_acc
            }, latest_path)

    early_stopping = EarlyStopping(patience=15, min_delta=0.001)
    checkpoint = ModelCheckpoint(filepath='models/braingat_fixed_best.pth')

    print("Starting training...")
    
    epochs = 100
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += data.num_graphs
            
        train_loss = total_loss / total
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y)
                val_loss += loss.item() * data.num_graphs
                pred = out.argmax(dim=1)
                val_correct += int((pred == data.y).sum())
                val_total += data.num_graphs
        
        val_loss /= val_total
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Checkpointing
        checkpoint(model, val_acc, epoch)
        
        # Early Stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("⏹ Early stopping triggered!")
            break
            
    print("Training complete.")
