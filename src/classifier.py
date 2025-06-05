import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

import torch
from model import BERTClassifier
from sentiment_dataset import AspectPolarityDataset, collate_fn
from torch.utils.data import DataLoader

from transformers import BertModel, BertTokenizer, BertTokenizerFast, AutoTokenizer, DebertaV2TokenizerFast
from tqdm import tqdm



class Classifier:
    """
    The Classifier: complete the definition of this class template by completing the __init__() function and
    the 2 methods train() and predict() below. Please do not change the signature of these methods
     """


    ############################################# complete the classifier class below
    
    def __init__(self, ollama_url: str):
        """
        This should create and initilize the model.
        !!!!! If the approach you have choosen is in-context-learning with an LLM from Ollama, you should initialize
         the ollama client here using the 'ollama_url' that is provided (please do not use your own ollama
         URL!)
        !!!!! If you have choosen an approach based on training an MLM or a generative LM, then your model should
        be defined and initialized here.
        """
        self.batch_size = 32
        # self.tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-uncased")
        self.tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[TGT]']})
        
        
        self.model = BERTClassifier(pretrained_model_name="FacebookAI/roberta-large")
        self.model.bert.resize_token_embeddings(len(self.tokenizer))
        
        # for param in self.model.bert.parameters():
        #     param.requires_grad = False
        self.label_id2str = {0: "positive", 1: "neutral", 2: "negative"}


    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
        If the approach you have choosen is in-context-learning with an LLM from Ollama, you must
          not train the model, and this method should contain only the "pass" instruction
        Otherwise:
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS

        """
        print(device)
        
        num_epochs = 10
        
        train_dataset = AspectPolarityDataset(train_filename, self.tokenizer)
        print("train label count:", train_dataset.label_count)
        
        dev_dataset = AspectPolarityDataset(dev_filename, self.tokenizer)
        print("dev label count:", dev_dataset.label_count)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        dev_loader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        
        model = self.model.to(device)
        optimizer = torch.optim.AdamW([
            {"params": model.bert.parameters(), "lr": 1e-5}, 
            {"params": model.fc.parameters(), "lr": 2e-5},
            # {"params": model.classifier.parameters(), "lr": 2e-5}
        ])
        
        # train label count: [1055, 58, 390], we calculate weights in mind
        weights = torch.tensor([1, 3, 2], dtype=torch.float, device=device)
        # criterion = nn.CrossEntropyLoss(weight=weights)
        
        best_dev_acc = 0.0
        patience = 3
        patience_counter = 0
        best_model_path = "model.pth"

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            total_batches = 0
            all_preds = []
            all_labels = []
            correct = 0
            total = 0
                
            ### Training Loop ###    
                
            for batch in tqdm(train_loader, desc = f'epoch {epoch+1}'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = None
                # token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['label'].to(device)
                offset_mappings = batch['offset_mapping']
                spans = batch["span"]
                

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask, token_type_ids, offset_mappings, spans)
                loss = F.cross_entropy(
                    logits,
                    labels,
                    weight=weights,     
                    label_smoothing=0.05     
                )
                loss.backward()
                optimizer.step()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                total_loss += loss.item()
                total_batches += 1
                
            avg_loss = total_loss/total_batches
            accuracy = correct/total
            print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f} | Train Accuracy: {accuracy:.4f}")
            
            ### Evaluating the model ###
            
            model.eval()
            total_loss = 0.0
            total_batches = 0
            all_preds = []
            all_labels = []
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in dev_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    # token_type_ids = batch['token_type_ids'].to(device)
                    labels = batch['label'].to(device)
                    offset_mappings = batch['offset_mapping']
                    spans = batch["span"]

                    logits = model(input_ids, attention_mask, token_type_ids, offset_mappings, spans)
                    loss = F.cross_entropy(
                        logits,
                        labels,
                        weight=weights,         
                        label_smoothing=0.05       
                    )
                    preds = torch.argmax(logits, dim=1)

                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())

                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    
                    total_loss += loss.item()
                    total_batches += 1
            
            avg_loss = total_loss / total_batches
            accuracy = correct / total
            print(f"[Epoch {epoch+1}] Dev Loss: {avg_loss:.4f} | Dev Accuracy: {accuracy:.4f}")
            
            if accuracy > best_dev_acc :
                best_dev_acc = accuracy
                patience_counter = 0
                print(f"New best model at epoch {epoch + 1}, saving model at {best_model_path}")
                torch.save(model.state_dict(), best_model_path)
            else :
                patience_counter += 1
                if patience_counter >= patience :
                    print(f"Performance on validation set didn't improve after {patience} epochs, ending training...")
                    break

    
        print(f"\nLoading best model from {best_model_path} with dev acc = {best_dev_acc:.4f}")
        self.model.load_state_dict(torch.load(best_model_path, map_location=device))
        

    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
        If the approach you have choosen is in-context-learning with an LLM from Ollama, ignore the 'device'
        parameter (because the device is specified when launching the Ollama server, and not by the client side)
        Otherwise:
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        dataset = AspectPolarityDataset(data_filename, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        
        self.model.eval()
        
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = None
                # token_type_ids = batch['token_type_ids'].to(device)
                offset_mappings = batch['offset_mapping']
                spans = batch["span"]

                logits = self.model(input_ids, attention_mask, token_type_ids, offset_mappings, spans)
                preds = torch.argmax(logits, dim=1)
                preds_str = [self.label_id2str[p.item()] for p in preds]
                predictions.extend(preds_str)

        return predictions
        
        





