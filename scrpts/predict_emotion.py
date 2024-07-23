import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.lstm_emotion_model import LSTMEmotionModel
import torch
from transformers import BertTokenizer

# Load model
vocab_size = 30522
embedding_dim = 128
hidden_dim = 256
num_layers = 2
output_dim = 3

model = LSTMEmotionModel(vocab_size, embedding_dim, hidden_dim, output_dim, num_layers)
model.load_state_dict(torch.load('../model/lstm_emotion_model.pth'))
model.eval()

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Mapping for decoding
emotion_labels = ['happy', 'sad', 'neutral']

def predict_emotion(text):
    tokens = tokenizer(text, padding='max_length', max_length=50, truncation=True, return_tensors='pt')
    input_ids = tokens['input_ids']
    
    with torch.no_grad():
        outputs = model(input_ids)
        predicted_class = torch.argmax(outputs, dim=1).item()
    
    return emotion_labels[predicted_class]

# Predict emotion
text = "I am feeling great today!"
predicted_emotion = predict_emotion(text)
print(f'Text: {text}')
print(f'Predicted Emotion: {predicted_emotion}')
