import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.lstm_emotion_model import LSTMEmotionModel

# Hyperparameters
embedding_dim = 128
hidden_dim = 256
num_layers = 2
output_dim = 3  # Number of emotion classes
num_epochs = 10
batch_size = 16
learning_rate = 0.001

# Load data
train_dataset = torch.load('../data/train_dataset.pth')
test_dataset = torch.load('../data/test_dataset.pth')

# Convert lists to tensors
train_inputs, train_labels = zip(*train_dataset)
test_inputs, test_labels = zip(*test_dataset)

train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
test_inputs = torch.tensor(test_inputs)
test_labels = torch.tensor(test_labels)

# Create DataLoader
train_dataloader = DataLoader(TensorDataset(train_inputs, train_labels), batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(TensorDataset(test_inputs, test_labels), batch_size=batch_size, shuffle=False)

# Model, loss function, optimizer
vocab_size = 30522  # BERT tokenizer vocab size

model = LSTMEmotionModel(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()#train comes from model(torch.nn.Module)
    total_loss = 0
    for batch in train_dataloader:
        input_ids, labels = batch
        optimizer.zero_grad()#torch.optim has a zero_grad() method.
        
        # Forward pass
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward() #The backward() method is part of the torch.Tensor class.
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataloader)}')

# Save the model
torch.save(model.state_dict(), '../model/lstm_emotion_model.pth')
