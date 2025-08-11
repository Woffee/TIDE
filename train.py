import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import auc, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, precision_recall_curve
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import os
from datetime import datetime
import logging
from pathlib import Path
import itertools
import glob
import pickle
import json
# from dadaptation import DAdaptAdam

seed = 12
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
#seed_value = 42
#np.random.seed(seed_value)
ROOT_PATH = Path(Path.cwd()).resolve().parent
NOW_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")





parser = argparse.ArgumentParser(description="Embedding script for peptide and CDR3 sequences.")
parser.add_argument('--train_base', type=str, default=os.path.join(ROOT_PATH, "dataset"), help='Root path of the train dataset')
parser.add_argument('--embedbase', type=str, default=os.path.join(ROOT_PATH, "meta_data/ChemBERTa"), help='Path to the embeddings output directory')
parser.add_argument('--mode', type=str, default="hard", help='random or hard split')
parser.add_argument('--use_fixed_splits', action='store_true', help='Use fixed splits without data leakage')
parser.add_argument('--few_shot_num', type=int, default=0, help='Number of few shot')
parser.add_argument('--results_dir', type=str, default="./MLP_finetune_dadapt", help='Path to to save the output model')
parser.add_argument('--hidden_layers', nargs='+', type=int, default=[32], help='Hidden layer sizes')
parser.add_argument('--dropout', type=float, default=0.3, help='drop out rate')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--GPU', type=int, default=0, help='GPU index')
args = parser.parse_args()

def init_logger(log_filename):
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # output to console at the same time
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

os.makedirs(args.results_dir, exist_ok=True)
log_filename = os.path.join(args.results_dir, f"{NOW_TIME}.log")
logger = init_logger(log_filename)


args_json = json.dumps(vars(args), indent=2)
logger.info(f"args: {args_json}")

if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.GPU}")
else:
    device = "cpu"
logger.info(f'running on {device}')

dataset_indices = [0, 1, 2, 3, 4]
# dataset_indices = [1]
num_epochs = 1000
momentum = 0.9
weight_decay = 0.001
learning_rate = args.learning_rate
batch_size = args.batch_size
dropout = args.dropout
hidden_layers = args.hidden_layers
results_dir = args.results_dir
mode = args.mode
embedbase = os.path.join(args.embedbase, args.mode)


class SimpleMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[512, 256, 128], dropout=0.2):
        """
        Simple but effective MLP for peptide-TCR interaction prediction
        """
        super(SimpleMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

        layers = []
        current_dim = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_size

        layers.append(nn.Linear(current_dim, output_size))
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.model(x)

class CrossAttnMLP(nn.Module):
    """
    Input X (B, 864)  # 384 peptide + 480 TCR
        │
        ├─ Split
        │    ├─ Peptide_emb = X[:, :384]
        │    └─ TCR_emb = X[:, 384:]
        │
        ├─ Linear Projection
        │    ├─ peptide_proj = Linear(384 → 128)
        │    └─ tcr_proj     = Linear(480 → 128)
        │
        ├─ Add sequence dimension
        │    ├─ peptide_seq = (B, 1, 128)
        │    └─ tcr_seq     = (B, 1, 128)
        │
        ├─ Cross-Attention (Peptide → TCR)
        │    peptide_attended = MHA(query=peptide_seq, key=tcr_seq, value=tcr_seq)
        │    peptide_attended = LayerNorm(peptide_seq + peptide_attended)
        │    peptide_attended = LayerNorm(peptide_attended + FFN_peptide(peptide_attended))
        │
        ├─ Cross-Attention (TCR → Peptide)
        │    tcr_attended = MHA(query=tcr_seq, key=peptide_seq, value=peptide_seq)
        │    tcr_attended = LayerNorm(tcr_seq + tcr_attended)
        │    tcr_attended = LayerNorm(tcr_attended + FFN_tcr(tcr_attended))
        │
        ├─ Remove sequence dimension (squeeze)
        │    peptide_attended = (B, 128)
        │    tcr_attended     = (B, 128)
        │
        ├─ Concatenate
        │    combined = concat(peptide_attended, tcr_attended)  # (B, 256)
        │
        ├─ Output Projection MLP
        │    Linear(256 → 128) → BatchNorm → GELU → Dropout
        │    Linear(128 → 64)  → BatchNorm → GELU → Dropout
        │
        └─ Final Output Layer
             Linear(64 → output_size)
             ↓
           Prediction (B, output_size)
    """
    def __init__(self, input_size, output_size, hidden_sizes=[128, 64], dropout=0.2, attn_heads=8):
        """
        Improved Cross-Attention MLP for peptide-TCR interaction prediction
        """
        super(CrossAttnMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.attn_heads = attn_heads
        
        # Split input size for peptide and TCR embeddings
        self.peptide_dim = 384
        self.tcr_dim = 480
        
        # Larger projection dimensions for better representation
        # self.projection_dim = 128
        self.projection_dim = hidden_sizes[0]
        
        # Projections for peptide and TCR
        self.peptide_projection = nn.Linear(self.peptide_dim, self.projection_dim)
        self.tcr_projection = nn.Linear(self.tcr_dim, self.projection_dim)
        
        # Cross-attention layers with proper residual connections
        self.cross_attention_peptide_to_tcr = nn.MultiheadAttention(
            embed_dim=self.projection_dim, 
            num_heads=attn_heads, 
            batch_first=True,
            dropout=dropout
        )
        self.cross_attention_tcr_to_peptide = nn.MultiheadAttention(
            embed_dim=self.projection_dim, 
            num_heads=attn_heads, 
            batch_first=True,
            dropout=dropout
        )
        
        # Layer normalization for cross-attention
        self.norm1_peptide = nn.LayerNorm(self.projection_dim)
        self.norm2_peptide = nn.LayerNorm(self.projection_dim)
        self.norm1_tcr = nn.LayerNorm(self.projection_dim)
        self.norm2_tcr = nn.LayerNorm(self.projection_dim)
        
        # Feed-forward networks for cross-attention
        self.ffn_peptide = nn.Sequential(
            nn.Linear(self.projection_dim, self.projection_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.projection_dim * 4, self.projection_dim),
            nn.Dropout(dropout)
        )
        self.ffn_tcr = nn.Sequential(
            nn.Linear(self.projection_dim, self.projection_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.projection_dim * 4, self.projection_dim),
            nn.Dropout(dropout)
        )
        
        # Output projection layers
        output_layers = []
        current_dim = 2 * self.projection_dim  # Concatenated peptide + TCR
        
        for hidden_size in hidden_sizes:
            output_layers.extend([
                nn.Linear(current_dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_size
        
        if output_layers:
            self.output_projection = nn.Sequential(*output_layers)
            final_input_size = hidden_sizes[-1]
        else:
            self.output_projection = nn.Identity()
            final_input_size = 2 * self.projection_dim
        
        # Final output layer
        self.output_layer = nn.Linear(final_input_size, output_size)
        
        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with better scaling"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)  # Increased gain
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Split input into peptide and TCR embeddings
        peptide_emb = x[:, :self.peptide_dim]
        tcr_emb = x[:, self.peptide_dim:]
        
        # Project to same dimension
        peptide_proj = self.peptide_projection(peptide_emb)  # (B, peptide_dim) -> (B, projection_dim)
        tcr_proj = self.tcr_projection(tcr_emb)  # (B, tcr_dim) -> (B, projection_dim)
        
        # Reshape for attention (add sequence dimension)
        peptide_seq = peptide_proj.unsqueeze(1)  # (B, 1, projection_dim)
        tcr_seq = tcr_proj.unsqueeze(1)  # (B, 1, projection_dim)
        
        # Cross-attention: peptide attends to TCR
        peptide_attended, _ = self.cross_attention_peptide_to_tcr(
            query=peptide_seq, 
            key=tcr_seq, 
            value=tcr_seq
        )
        # Proper residual connection
        peptide_attended = self.norm1_peptide(peptide_seq + peptide_attended)
        peptide_attended = self.norm2_peptide(peptide_attended + self.ffn_peptide(peptide_attended))
        
        # Cross-attention: TCR attends to peptide
        tcr_attended, _ = self.cross_attention_tcr_to_peptide(
            query=tcr_seq, 
            key=peptide_seq, 
            value=peptide_seq
        )
        # Proper residual connection
        tcr_attended = self.norm1_tcr(tcr_seq + tcr_attended)
        tcr_attended = self.norm2_tcr(tcr_attended + self.ffn_tcr(tcr_attended))
        
        # Combine attended representations
        combined = torch.cat([
            peptide_attended.squeeze(1),  # (B, projection_dim)
            tcr_attended.squeeze(1)       # (B, projection_dim)
        ], dim=1)  # (B, 2 * projection_dim)
        
        # Output projection
        x = self.output_projection(combined)  # (B, final_hidden_size)
        
        # Final output layer
        x = self.output_layer(x)  # (B, output_size)
        
        return x



def save_best_model_state(model, filename):
    torch.save(model.state_dict(), filename)


def load_best_model_state(model, filename):
    model.load_state_dict(torch.load(filename))


def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler, device, dataset_index,
                output_model_path):
    patience = 20
    counter = 0
    best_val_roc_auc = 0.0
    best_model_path = output_model_path + f"/best_mol-esm_{dataset_index}.pth"

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        model.eval()
        val_running_loss = 0.0
        y_true_val = []
        val_probabilities = []
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = model(val_inputs.float())
                val_loss = criterion(val_outputs, val_targets)
                val_running_loss += val_loss.item() * val_inputs.size(0)
                val_probabilities.extend(torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy())
                y_true_val.extend(val_targets.cpu().numpy())
            val_loss = val_running_loss / len(val_loader.dataset)
            val_auc = roc_auc_score(y_true_val, val_probabilities)
            precision, recall, _ = precision_recall_curve(y_true_val, val_probabilities)
            pr_auc = auc(recall, precision)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation AUC: {val_auc:.4f}, PR-AUC: {pr_auc:.4f}')

        if val_auc > best_val_roc_auc:
            best_val_roc_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print("Saved best model")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}.')
                break
        scheduler.step(val_auc)
    print("Training complete. Best Val ROC-AUC: {:.4f}".format(best_val_roc_auc))

    return best_model_path


def evaluate_and_save(model, test_loader, criterion, device, dataset_index, best_model_path,
                      results_dir=""):
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    with torch.no_grad():
        test_probabilities = []
        y_true_test = []
        test_running_loss = 0.0
        for test_inputs, test_targets in test_loader:
            test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
            test_outputs = model(test_inputs.float())
            test_loss = criterion(test_outputs, test_targets)
            test_running_loss += test_loss.item() * test_inputs.size(0)
            test_probabilities.extend(torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy())
            y_true_test.extend(test_targets.cpu().numpy())

    test_loss = test_running_loss / len(test_loader.dataset)
    test_auc = roc_auc_score(y_true_test, test_probabilities)
    test_predictions = [1 if prob > 0.5 else 0 for prob in test_probabilities]
    precision, recall, _ = precision_recall_curve(y_true_test, test_probabilities)

    metrics = {
        'AUROC': test_auc,
        'Accuracy': accuracy_score(y_true_test, test_predictions),
        'Recall': recall_score(y_true_test, test_predictions),
        'Precision': precision_score(y_true_test, test_predictions),
        'F1 score': f1_score(y_true_test, test_predictions),
        'AUPR': auc(recall, precision),
    }

    result_df = pd.DataFrame({
        'score': list(metrics.values()),
        'metrics': list(metrics.keys()),
        'experiment': [dataset_index] * len(metrics)
    })

    csv_path = os.path.join(results_dir, f"evaluation_{dataset_index}.csv")
    # result_df.to_csv(csv_path, index=False)

    print(f"\nBest Model Performance and Evaluation of dataset{dataset_index}:")
    for metric, score in metrics.items():
        print(f"{metric}: {score * 100:.4f}%")

    return result_df

with open(os.path.join(embedbase, "peptide_dict.pkl"), 'rb') as f:
    peptide_embed_dict = pickle.load(f)

with open(os.path.join(embedbase, "tcrb_dict.pkl"), 'rb') as f:
    tcrb_embed_dict = pickle.load(f)

def load_data(dataset_index):
    logger.info(f"Processing dataset {args.train_base}, {mode}, {dataset_index}...")
    train_path = os.path.join(args.train_base, "pep+cdr3b")

    if args.few_shot_num > 0:
        # TCHard/dataset/few_shot_split/pep+cdr3b/train/random/1-train-0.csv
        train_path = os.path.join(args.train_base, 'few_shot_split/pep+cdr3b')
        train_df =      pd.read_csv(f'{train_path}/train/{mode}/{args.few_shot_num}-train-{dataset_index}.csv', low_memory=False).drop_duplicates()
        validation_df = pd.read_csv(f'{train_path}/validation/{mode}/{args.few_shot_num}-validation-{dataset_index}.csv', low_memory=False)
        test_df =       pd.read_csv(f'{train_path}/test/{mode}/{args.few_shot_num}-test-{dataset_index}.csv', low_memory=False)

    else:
        train_df =      pd.read_csv(f'{train_path}/train/{mode}/train-{dataset_index}.csv', low_memory=False).drop_duplicates()
        validation_df = pd.read_csv(f'{train_path}/validation/{mode}/validation-{dataset_index}.csv', low_memory=False)
        test_df =       pd.read_csv(f'{train_path}/test/{mode}/test-{dataset_index}.csv', low_memory=False)

    logger.info(f"Using dataset from {train_path}")

    tcrb_seq_train = np.vstack(train_df['tcrb'].apply(lambda x: tcrb_embed_dict[x]).values)
    tcrb_seq_validation = np.vstack(validation_df['tcrb'].apply(lambda x: tcrb_embed_dict[x]).values)
    tcrb_seq_test = np.vstack(test_df['tcrb'].apply(lambda x: tcrb_embed_dict[x]).values)

    peptide_seq_train = np.vstack(train_df['peptide'].apply(lambda x: peptide_embed_dict[x]).values)
    peptide_seq_validation = np.vstack(validation_df['peptide'].apply(lambda x: peptide_embed_dict[x]).values)
    peptide_seq_test = np.vstack(test_df['peptide'].apply(lambda x: peptide_embed_dict[x]).values)

    label_seq_train = train_df['label'].values
    label_seq_validation = validation_df['label'].values
    label_seq_test = test_df['label'].values

    # Separate scaling for peptide and TCR embeddings
    peptide_scaler = MinMaxScaler()
    tcr_scaler = MinMaxScaler()
    
    # Scale peptide embeddings
    peptide_seq_train_scaled = peptide_scaler.fit_transform(peptide_seq_train)
    peptide_seq_validation_scaled = peptide_scaler.transform(peptide_seq_validation)
    peptide_seq_test_scaled = peptide_scaler.transform(peptide_seq_test)
    
    # Scale TCR embeddings
    tcrb_seq_train_scaled = tcr_scaler.fit_transform(tcrb_seq_train)
    tcrb_seq_validation_scaled = tcr_scaler.transform(tcrb_seq_validation)
    tcrb_seq_test_scaled = tcr_scaler.transform(tcrb_seq_test)
    
    # Concatenate scaled embeddings
    X_train = np.column_stack((peptide_seq_train_scaled, tcrb_seq_train_scaled))
    y_train = label_seq_train

    X_validation = np.column_stack((peptide_seq_validation_scaled, tcrb_seq_validation_scaled))
    y_validation = label_seq_validation

    X_test = np.column_stack((peptide_seq_test_scaled, tcrb_seq_test_scaled))
    y_test = label_seq_test

    # Add data validation
    logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_validation.shape}, Test: {X_test.shape}")
    logger.info(f"Label distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_validation)}, Test: {np.bincount(y_test)}")
    logger.info(f"Data ranges - Train: [{X_train.min():.3f}, {X_train.max():.3f}], Val: [{X_validation.min():.3f}, {X_validation.max():.3f}]")
    
    return X_train, y_train, X_validation, y_validation, X_test, y_test



all_results = []
os.makedirs(results_dir, exist_ok=True)

# Initialize as empty list instead of empty DataFrame
combined_metrics_list = []
for dataset_index in dataset_indices:
    # Pass param_str to run_training so model checkpoint is unique
    def run_training_with_param(dataset_index, num_epochs=1000, batch_size=batch_size, learning_rate=learning_rate, momentum=0.9, weight_decay=0.001, dropout=dropout, device = device, results_dir = results_dir, param_str=''):
        X_train, y_train, X_val, y_val, X_test, y_test = load_data(dataset_index)
        # Use simple dataset without augmentation for now
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.tensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.tensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataset = TensorDataset(torch.from_numpy(X_test), torch.tensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # criterion = nn.CrossEntropyLoss()
        class_weights_train = torch.tensor(compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)).float().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_train)

        model = CrossAttnMLP(X_train.shape[1], 2, hidden_layers, dropout=dropout).to(device)
        logger.info(f"Using CrossAttnMLP model")

        # SGD with momentum
        # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1, verbose=True)

        # Use AdamW instead of DAdaptAdam for better stability
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Use a simpler scheduler for now
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=15, factor=0.7, verbose=True)

        # Save model with param_str in filename
        best_model_path = os.path.join(results_dir, f"best_mol-esm_{param_str}_{dataset_index}.pth")
        # Remove any old checkpoint for this param/dataset
        for f in glob.glob(os.path.join(results_dir, f"best_mol-esm_*_{dataset_index}.pth")):
            os.remove(f)
        # Train and save
        patience = 30  # Increased patience for better convergence
        counter = 0
        best_val_roc_auc = 0.0
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs.float())
                loss = criterion(outputs, targets)
                loss.backward()
                # Add gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
            
            # Add training loss monitoring
            train_loss = running_loss / len(train_loader.dataset)
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}")
            model.eval()
            val_running_loss = 0.0
            y_true_val = []
            val_probabilities = []
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs = model(val_inputs.float())
                    val_loss = criterion(val_outputs, val_targets)
                    val_running_loss += val_loss.item() * val_inputs.size(0)
                    val_probabilities.extend(torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy())
                    y_true_val.extend(val_targets.cpu().numpy())
                val_loss = val_running_loss / len(val_loader.dataset)
                val_auc = roc_auc_score(y_true_val, val_probabilities)
                precision, recall, _ = precision_recall_curve(y_true_val, val_probabilities)
                pr_auc = auc(recall, precision)

            current_lr = optimizer.param_groups[0]['lr']
            # Add prediction statistics for debugging
            val_probs = np.array(val_probabilities)
            val_preds = (val_probs > 0.5).astype(int)
            pos_ratio = np.mean(val_probs)
            pred_pos_ratio = np.mean(val_preds)
            
            print(f'Epoch [{epoch + 1}/{num_epochs}], LR: {current_lr:.6f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, PR-AUC: {pr_auc:.4f}')
            print(f'  Val pred stats: mean_prob={pos_ratio:.3f}, pred_pos_ratio={pred_pos_ratio:.3f}, true_pos_ratio={np.mean(y_true_val):.3f}')

            if val_auc > best_val_roc_auc:
                best_val_roc_auc = val_auc
                torch.save(model.state_dict(), best_model_path)
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break
            scheduler.step(val_auc)
        # Evaluate
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        with torch.no_grad():
            test_probabilities = []
            y_true_test = []
            test_running_loss = 0.0
            for test_inputs, test_targets in test_loader:
                test_inputs, test_targets = test_inputs.to(device), test_targets.to(device)
                test_outputs = model(test_inputs.float())
                test_loss = criterion(test_outputs, test_targets)
                test_running_loss += test_loss.item() * test_inputs.size(0)
                test_probabilities.extend(torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy())
                y_true_test.extend(test_targets.cpu().numpy())
        test_loss = test_running_loss / len(test_loader.dataset)
        test_auc = roc_auc_score(y_true_test, test_probabilities)
        test_predictions = [1 if prob > 0.5 else 0 for prob in test_probabilities]
        precision, recall, _ = precision_recall_curve(y_true_test, test_probabilities)
        metrics = {
            'AUROC': test_auc,
            'Accuracy': accuracy_score(y_true_test, test_predictions),
            'Recall': recall_score(y_true_test, test_predictions),
            'Precision': precision_score(y_true_test, test_predictions),
            'F1 score': f1_score(y_true_test, test_predictions),
            'AUPR': auc(recall, precision),
        }
        result_df = pd.DataFrame({
            'score': list(metrics.values()),
            'metrics': list(metrics.keys()),
            'experiment': [dataset_index] * len(metrics)
        })
        to_file_result = os.path.join(results_dir, f'result_{dataset_index}.json')
        result_df.to_json(to_file_result, orient='records', lines=True)
        return result_df
    result_df = run_training_with_param(dataset_index)
    # Append result_df to list instead of concatenating empty DataFrame
    combined_metrics_list.append(result_df)
# Save results for this param combo
to_file_result = os.path.join(results_dir, 'result.csv')
# Concatenate all DataFrames in the list at once
combined_metrics_df = pd.concat(combined_metrics_list, ignore_index=True)

# Save in append mode - add param info to the DataFrame
combined_metrics_df_with_params = combined_metrics_df.copy()
combined_metrics_df_with_params['params'] = ''

# Append to existing file or create new one
if os.path.exists(to_file_result):
    combined_metrics_df_with_params.to_csv(to_file_result, mode='a', header=False, index=False)
else:
    combined_metrics_df_with_params.to_csv(to_file_result, index=False)

logger.info(f"saved combined_metrics_df: {to_file_result}")
stats = combined_metrics_df.groupby('metrics')['score'].agg(['mean', 'std']).reset_index()
to_file_stats = os.path.join(results_dir, 'stats.csv')
stats.to_csv(to_file_stats)
# Track best by mean AUROC
mean_auroc = stats[stats['metrics'] == 'AUROC']['mean'].values[0]


# Save all results in append mode
all_results_df = pd.DataFrame(all_results)
all_results_file = os.path.join(results_dir, 'all_results.csv')
if os.path.exists(all_results_file):
    all_results_df.to_csv(all_results_file, mode='a', header=False, index=False)
else:
    all_results_df.to_csv(all_results_file, index=False)

logger.info(f"mean AUROC: {mean_auroc:.4f}")
logger.info(f"results_dir: {results_dir}")
