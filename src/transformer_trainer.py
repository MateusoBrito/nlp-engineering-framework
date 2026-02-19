import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoConfig
import os
import numpy as np

class Transformer(nn.Module):
    def __init__(self, model_name, num_labels):
        super(Transformer, self).__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)

        self.classifier1 = nn.Linear(self.config.hidden_size, 256)
        self.relu = nn.ReLU()
        self.classifier2 = nn.Linear(256, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        x = self.classifier1(pooled_output)
        x = self.relu(x)
        logits = self.classifier2(x)

        return logits


class TransformerTrainer:
    def __init__(self, model_name, num_labels, params):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Transformer(model_name, num_labels).to(self.device)

        self.lr = float(params.get('learning_rate', 2e-5))
        self.epochs = int(params.get('num_train_epochs', 3))
        self.batch_size = int(params.get('batch_size', 16))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, train_loader):
        self.model.train()
        total_loss = 0

        for batch in train_loader:
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def predict(self, test_loader):
        self.model.eval() 
        all_predictions = []
        
        with torch.no_grad(): 
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                # Pega a classe com maior probabilidade (Logits -> Argmax)
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                
        return np.array(all_predictions)
    
    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)