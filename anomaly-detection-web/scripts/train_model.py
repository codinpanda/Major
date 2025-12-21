import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, sosfiltfilt, resample, welch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from torch.utils.data import TensorDataset, DataLoader

# Setup
sns.set_style("whitegrid")
np.random.seed(42)
torch.manual_seed(42)

# Paths
CURRENT_DIR = os.getcwd() # Executing from project root
PROJECT_ROOT = CURRENT_DIR 
DATA_PATH = os.path.join(PROJECT_ROOT, 'datasets', 'WESAD', 'S2', 'S2.pkl')
MODEL_EXPORT_PATH = os.path.join(PROJECT_ROOT, 'public', 'model', 'hybrid_model.onnx')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'public', 'results')

# Hyperparameters
FS = 700
WINDOW_SECONDS = 5
SUB_WINDOW_SECONDS = 0.5
SUB_WINDOW_STEP = 0.25

SEQ_LEN = int(FS * WINDOW_SECONDS) # 3500
FRAME_LEN = int(FS * SUB_WINDOW_SECONDS) # 350
FRAME_STEP = int(FS * SUB_WINDOW_STEP)   # 175

WINDOW_STEP = 1750 # 50% Overlap (Paper Standard)
BATCH_SIZE = 64
EPOCHS = 50       
PATIENCE = 10      
NOISE_FACTOR = 0.05 
HIDDEN_SIZE = 64 # Match Paper Specs

print(f"Project Root: {PROJECT_ROOT}")

# --- 1. Data Loading ---
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

print(f"Loading {DATA_PATH}...")
with open(DATA_PATH, 'rb') as file:
    data = pickle.load(file, encoding='latin1')

ecg = data['signal']['chest']['ECG'].flatten()
eda = data['signal']['chest']['EDA'].flatten()
resp = data['signal']['chest']['Resp'].flatten()
acc = data['signal']['chest']['ACC']
bvp = data['signal']['wrist']['BVP'].flatten()

# Resample BVP
bvp = resample(bvp, len(ecg))

# --- 2. Preprocessing (SOS Filtering) ---
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, data)

def clean(x):
    return np.nan_to_num(x, nan=np.nanmean(x) if not np.isnan(np.nanmean(x)) else 0.0)

print("Filtering signals...")
ecg = butter_bandpass_filter(clean(ecg), 0.5, 12, FS)
eda = butter_bandpass_filter(clean(eda), 0.5, 12, FS)
resp = butter_bandpass_filter(clean(resp), 0.5, 12, FS)
bvp = butter_bandpass_filter(clean(bvp), 0.5, 12, FS)
for i in range(3):
    acc[:, i] = butter_bandpass_filter(clean(acc[:, i]), 0.5, 12, FS)

def normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-6)

ecg = normalize(ecg)
eda = normalize(eda)
resp = normalize(resp)
bvp = normalize(bvp)
acc = (acc - np.mean(acc, axis=0)) / (np.std(acc, axis=0) + 1e-6)

labels = data['label']
mask = np.isin(labels, [1, 2])
ecg, eda, resp, bvp, acc = ecg[mask], eda[mask], resp[mask], bvp[mask], acc[mask]
y = labels[mask]
y = np.where(y == 2, 1, 0).astype(np.float32)

# --- 3. Feature Extraction Functions ---
def get_rmssd(x):
    diff = np.diff(x)
    return np.sqrt(np.mean(diff**2))
def get_energy(x):
    return np.sum(x**2)
def get_entropy(x):
    _, psd = welch(x, fs=FS, nperseg=len(x))
    psd_norm = psd / (np.sum(psd) + 1e-10)
    return -np.sum(psd_norm * np.log(psd_norm + 1e-10))
def get_dom_freq(x):
    f, psd = welch(x, fs=FS, nperseg=len(x))
    return f[np.argmax(psd)]

def extract_features_from_frame(frame_data):
    feats = []
    for ch in range(7):
        sig = frame_data[:, ch]
        feats.append(get_rmssd(sig))
        feats.append(get_energy(sig))
        feats.append(get_entropy(sig))
        feats.append(get_dom_freq(sig))
    return np.array(feats)

def create_feature_sequences(ecg, eda, resp, acc, bvp, y, seq_len, step, frame_len, frame_step):
    xs, ys = [], []
    all_signals = np.column_stack((ecg, eda, resp, acc, bvp))
    
    print(f"Generating sequences from {len(y)} samples...")
    for i in range(0, len(y) - seq_len, step):
        window_data = all_signals[i:i+seq_len] 
        label = y[i+seq_len-1]
        
        frame_feats = []
        for j in range(0, seq_len - frame_len + 1, frame_step):
            frame = window_data[j:j+frame_len]
            f_vec = extract_features_from_frame(frame)
            frame_feats.append(f_vec)
            
        xs.append(np.array(frame_feats))
        ys.append(label)
        
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

print("Starting Feature Extraction (This may take a moment)...")
X, Y = create_feature_sequences(ecg, eda, resp, acc, bvp, y, SEQ_LEN, WINDOW_STEP, FRAME_LEN, FRAME_STEP)

print(f"Input Shape: {X.shape}") 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

noise = np.random.normal(0, NOISE_FACTOR, X_train.shape)
X_train_final = np.vstack((X_train, X_train + noise))
y_train_final = np.hstack((y_train, y_train))

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_final).float(), torch.from_numpy(y_train_final).float()), 
                          batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()), 
                         batch_size=BATCH_SIZE, shuffle=False)

# --- 4. Model ---
class HybridModel(nn.Module):
    def __init__(self, input_dim, hidden_size=64):
        super(HybridModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(x)
        lstm_feat = lstm_out[:, -1, :]
        gru_feat = gru_out[:, -1, :]
        combined = torch.cat((lstm_feat, gru_feat), dim=1)
        
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc3(x)
        return out

INPUT_FEATS = X_train.shape[2]
model = HybridModel(input_dim=INPUT_FEATS, hidden_size=HIDDEN_SIZE)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

num_neg = len(y[y==0])
num_pos = len(y[y==1])
pos_weight = torch.tensor([num_neg / num_pos])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=EPOCHS)

# --- 3. Training Loop ---
best_val_loss = float('inf')
patience_counter = 0
train_history = []
val_history = []

print(f"Starting Training (Max Epochs: {EPOCHS})...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
    
    avg_train = running_loss / len(train_loader)
    train_history.append(avg_train)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    avg_val = val_loss / len(test_loader)
    val_history.append(avg_val)
    print(f"Epoch {epoch+1:02d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at {epoch+1}")
            break

model.load_state_dict(best_model_state)

# --- 4. Visualization & Analysis ---
print("\\n--- Model Performance Analysis ---")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(train_history, label='Training Loss', color='blue')
plt.plot(val_history, label='Validation Loss', color='orange')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'loss_curve.png'))
print(f"Saved Learning Curve to {RESULTS_DIR}/loss_curve.png")

# Get Predictions
model.eval()
y_true, y_prob = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs).squeeze()
        probs = torch.sigmoid(outputs)
        y_prob.extend(probs.numpy())
        y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_prob = np.array(y_prob)

# Find Optimal Threshold
thresholds = np.arange(0.1, 0.9, 0.01)
best_f1 = 0
best_thresh = 0.5
for t in thresholds:
    y_pred_t = (y_prob > t).astype(float)
    f1 = f1_score(y_true, y_pred_t)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"Optimal Threshold: {best_thresh:.2f}")

# Final Predictions
y_pred_final = (y_prob > best_thresh).astype(float)

# 2. Confusion Matrix
cm = confusion_matrix(y_true, y_pred_final)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Baseline', 'Stress'], yticklabels=['Baseline', 'Stress'])
plt.title(f'Confusion Matrix (Threshold={best_thresh:.2f})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
print(f"Saved Confusion Matrix to {RESULTS_DIR}/confusion_matrix.png")

# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'))
print(f"Saved ROC Curve to {RESULTS_DIR}/roc_curve.png")

# 4. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_prob)
avg_precision = average_precision_score(y_true, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, 'pr_curve.png'))
print(f"Saved PR Curve to {RESULTS_DIR}/pr_curve.png")

# Analysis Text
print("--- Final Verdict ---")
print(classification_report(y_true, y_pred_final, target_names=['Baseline', 'Stress']))
print(f"AUC-ROC Score: {roc_auc:.4f}")
print(f"Average Precision: {avg_precision:.4f}")

if roc_auc > 0.95:
    print("✅ Model Quality: EXCELLENT (Highly separable classes)")
elif roc_auc > 0.85:
    print("✅ Model Quality: GOOD (Strong performance, minor errors)")
else:
    print("⚠️ Model Quality: MODERATE/POOR (Needs improvement)")

if best_f1 > 0.90:
    print("✅ F1-Score indicates high accuracy on Stress detection.")
else:
    print("⚠️ F1-Score suggests struggle with False Positives/Negatives.")

# Export
os.makedirs(os.path.dirname(MODEL_EXPORT_PATH), exist_ok=True)

# Define Wrapper for Probability Output
class ExportModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.model(x))

# Export Wrapped Model
export_model = ExportModel(model)
export_model.eval()

dummy_input = torch.randn(1, X_train.shape[1], X_train.shape[2])

# Export with Opset 12 for better Web compatibility
torch.onnx.export(
    export_model, 
    dummy_input, 
    MODEL_EXPORT_PATH, 
    input_names=['input'], 
    output_names=['output'], 
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    opset_version=12
)
print(f"Model exported to {MODEL_EXPORT_PATH} (with Sigmoid & Opset 12)")
