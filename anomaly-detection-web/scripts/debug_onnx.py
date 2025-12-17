import torch
import torch.nn as nn
import torch.onnx
import os

SEQUENCE_LENGTH = 60
INPUT_DIM = 2

class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        
        # LSTM Path (2 Stacked Layers, 64 Units)
        self.lstm = nn.LSTM(input_size=INPUT_DIM, hidden_size=64, num_layers=2, batch_first=True, dropout=0.3)
        
        # GRU Path (2 Stacked Layers, 64 Units)
        self.gru = nn.GRU(input_size=INPUT_DIM, hidden_size=64, num_layers=2, batch_first=True, dropout=0.3)
        
        # Dual-Path Concatenation -> 64 + 64 = 128
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Dual Path Processing
        lstm_out, _ = self.lstm(x)  # (Batch, Seq, 64)
        gru_out, _ = self.gru(x)    # (Batch, Seq, 64)
        
        # Pooling (Take last time step)
        lstm_feat = lstm_out[:, -1, :]
        gru_feat = gru_out[:, -1, :]
        
        # Fusion
        combined = torch.cat((lstm_feat, gru_feat), dim=1) # (Batch, 128)
        
        # Classification
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        out = self.fc3(x) # Logits
        return out

# Export Wrapper
class ExportModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.model(x))

model = HybridModel()
model.eval()
export_model = ExportModel(model)

dummy_input = torch.randn(1, SEQUENCE_LENGTH, INPUT_DIM)
output_path = "debug_model.onnx"

print("Attempting ONNX export...")
torch.onnx.export(export_model, dummy_input, output_path, 
                  input_names=['input'], output_names=['output'],
                  opset_version=12,
                  do_constant_folding=False)
print("Export successful!")
