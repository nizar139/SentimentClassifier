from torch.utils.data import Dataset
import torch

class AspectPolarityDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        label2id = {"positive": 0, "neutral": 1, "negative": 2}
        self.label_count = [0]*3

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                
                if len(parts) != 5:
                    continue  # skip malformed lines

                label_str, aspect, target, span, sentence = parts
                label = label2id[label_str]
                self.label_count[label] += 1
                
                start_char, end_char = map(int, span.split(":"))
                
                before = sentence[:start_char]
                term = sentence[start_char:end_char]
                after = sentence[end_char:]

                # Wrap the term
                new_sentence = f"{before}[TGT] {term} [TGT]{after}"
                sentence_B = f"{aspect} {target}"

                # Tokenize using sentence and aspect category
                encoded = tokenizer(
                    new_sentence,
                    sentence_B,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                )
                self.samples.append({
                    "input_ids": encoded["input_ids"].squeeze(0),
                    "attention_mask": encoded["attention_mask"].squeeze(0),
                    # "token_type_ids": encoded["token_type_ids"].squeeze(0),
                    "offset_mapping": encoded['offset_mapping'].squeeze(0),
                    "label": torch.tensor(label, dtype=torch.long),
                    "span": (start_char + 6, end_char + 6)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    
def collate_fn(batch):
    input_ids = torch.stack([x['input_ids'] for x in batch])
    attention_mask = torch.stack([x['attention_mask'] for x in batch])
    # token_type_ids = torch.stack([x['token_type_ids'] for x in batch])
    labels = torch.stack([x['label'] for x in batch])
    spans = [x['span'] for x in batch]
    offset_mappings = [x['offset_mapping'] for x in batch]  # still a list of lists

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        # "token_type_ids": token_type_ids,
        "label": labels,
        "span": spans,
        "offset_mapping": offset_mappings
    }
