import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    le = LabelEncoder()
    data['emotion'] = le.fit_transform(data['emotion'])
    return data, le.classes_

class ConversationDataset(Dataset):
    def __init__(self, conversations, emotions, tokenizer, max_length):
        self.conversations = conversations
        self.emotions = emotions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        text = self.conversations[idx]
        label = self.emotions[idx]
        
        tokens = self.tokenizer(text, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        input_ids = tokens['input_ids'].squeeze()
        
        return input_ids, label
