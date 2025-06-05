import torch
import torch.nn as nn

import torch
from transformers import BertModel, BertTokenizer, AutoModel
import torch.nn as nn

class BERTClassifier(nn.Module):
    def __init__(self, pretrained_model_name="google-bert/bert-base-uncased", num_labels=3):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(2 * self.bert.config.hidden_size, num_labels)
        # self.fc = nn.Linear(2 * self.bert.config.hidden_size, 128)
        # self.classifier = nn.Linear(128, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None, offset_mappings=None, spans=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        # # Use the [CLS] token representation
        # cls_output = outputs.last_hidden_state[:, 0, :]  # shape: (batch_size, hidden_dim)
        # cls_output = self.dropout(cls_output)
        # fc_out = self.relu(self.fc(cls_output))
        # logits = self.classifier(fc_out)
        # return logits
        
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        batch_embeddings = []

        for i in range(input_ids.size(0)):
            span_start, span_end = spans[i]  # (char_start, char_end)
            offset_map = offset_mappings[i]  # list of (start, end) per token

            # Collect all token indices that fall fully within the span
            token_indices = [
                j for j, (start, end) in enumerate(offset_map)
                if start >= span_start and end <= span_end
            ]

            # Fallback: use CLS token if none matched
            if not token_indices:
                aspect_embedding = hidden_states[i, 0, :]
            else:
                aspect_embedding = hidden_states[i, token_indices, :].mean(dim=0)

            cls_embedding = hidden_states[i, 0, :]
            concat = torch.cat([cls_embedding, aspect_embedding], dim=-1)
            batch_embeddings.append(concat)

        pooled_output = torch.stack(batch_embeddings)  # shape: (batch_size, hidden_dim)
        pooled_output = self.dropout(pooled_output)
        # fc_out = self.relu(self.fc(pooled_output))
        # logits = self.classifier(fc_out)
        logits = self.fc(pooled_output)
        return logits
