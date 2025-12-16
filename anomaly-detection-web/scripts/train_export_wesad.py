import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import os
import onnx
import onnxruntime

# Config
DATA_PATH = '../datasets/WESAD/S2/S2.pkl'
MODEL_PATH = 'public/model/hybrid_model.onnx'
SAMPLE_DATA_PATH = 'src/simulation/wesad_sample.json'
SEQUENCE_LENGTH = 60
INPUT_DIM = 2 # ECG, EDA

# Ensure directories exist
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

print("Loading WESAD S2 Data...")
with open(DATA_PATH, 'rb') as file:
    data = pickle.load(file, encoding='latin1')

# Extract Signals
ecg = data['signal']['chest']['ECG'].flatten()
eda = data['signal']['chest']['EDA'].flatten()
labels = data['label']

# Normalize (Simple Min-Max for Web Compatibility)
ecg = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-6)
eda = (eda - np.mean(eda)) / (np.std(eda) + 1e-6)

# FilterBaseline (1) and Stress (2)
mask = np.isin(labels, [1, 2])
ecg = ecg[mask]
eda = eda[mask]
y = labels[mask]
y = np.where(y == 2, 1, 0).astype(np.float32) # 1 = Stress, 0 = Baseline

print(f"Data Loaded: {len(y)} samples")

# Create Sequences
def create_sequences(ecg, eda, y, seq_len):
    xs, ys = [], []
    # Step 100 for speed/size reduction
    for i in range(0, len(ecg) - seq_len, 100):
        xs.append(np.column_stack((ecg[i:i+seq_len], eda[i:i+seq_len])))
        ys.append(y[i+seq_len-1])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

print("Creating sequences...")
X, Y = create_sequences(ecg, eda, y, SEQUENCE_LENGTH)
print(f"Sequences: {X.shape}")

# Define Model (Lightweight Hybrid for Web)
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.lstm = nn.LSTM(INPUT_DIM, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.sigmoid(out)

model = HybridModel()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train (Quick Epochs for Demo)
print("Training Model...")
X_tensor = torch.from_numpy(X)
Y_tensor = torch.from_numpy(Y).unsqueeze(1)

for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, Y_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Export to ONNX
print(f"Exporting to {MODEL_PATH}...")
dummy_input = torch.randn(1, SEQUENCE_LENGTH, INPUT_DIM)
torch.onnx.export(model, dummy_input, MODEL_PATH, 
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

# Export Sample Data for Web Simulator
# Take 500 samples of Baseline and 500 of Stress
print(f"Exporting Sample Data to {SAMPLE_DATA_PATH}...")

# Indices for Baseline (0) and Stress (1)
baseline_indices = np.where(Y == 0)[0][:500]
stress_indices = np.where(Y == 1)[0][:500]

sample_data = {
    'baseline': X[baseline_indices].tolist(), # [500, 60, 2]
    'stress': X[stress_indices].tolist()      # [500, 60, 2]
}

with open(SAMPLE_DATA_PATH, 'w') as f:
    json.dump(sample_data, f)

print("Done!")
