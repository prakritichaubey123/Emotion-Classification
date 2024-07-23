import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('../data/conversation_data.csv')

# Encode labels
le = LabelEncoder()
data['emotion'] = le.fit_transform(data['emotion'])

# Tokenize texts
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 50

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_texts = train_data['text'].tolist()
train_labels = train_data['emotion'].tolist()
test_texts = test_data['text'].tolist()
test_labels = test_data['emotion'].tolist()

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

train_dataset = list(zip(train_encodings['input_ids'], train_labels))
test_dataset = list(zip(test_encodings['input_ids'], test_labels))

torch.save(train_dataset, '../data/train_dataset.pth')
torch.save(test_dataset, '../data/test_dataset.pth')
