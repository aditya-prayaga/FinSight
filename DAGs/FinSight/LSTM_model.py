# import torch
# import torch.nn as nn
# import torch.optim as optim

# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, dropout):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)
    
#     def forward(self, x):
#         x, _ = self.lstm(x)
        
#         if not x.shape[1] == 1:  # Check if sequence length dimension exists
#             x = x.reshape(x.shape[0], -1, x.shape[2])  # Reshape if needed (batch_size, seq_len, hidden_size)
#         print(f"Output shape after reshape: {x.shape}")

#         x = self.fc(x[:, -1, :])        
#         print(f"Output shape after FC: {x.shape}")      
#         return x
