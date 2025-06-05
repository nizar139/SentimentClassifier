**Students Names** :

   - Nizar EL‑GHAZAL  
   - Mohamed Reda MOUQED  
   - Oussama ER‑RAHMANY

---

**Sentiment Classifier**

This project implements an *aspect‑based* sentiment classifier using **FacebookAI/roberta-large**.  
Given a sentence and a target aspect, it predicts the sentiment polarity (**positive, neutral, or negative**) expressed toward that aspect.

---

**Input Format**

Each data line is tab‑separated and contains five fields:
```
<label>    <aspect>    <target>    <charStart:charEnd>    <sentence>
```

During preprocessing:
- The target term in the sentence is **highlighted** using custom `[TGT]` tokens:  
  Example: `... it's the best [TGT] pie [TGT] on the UWS!`
- The final input to the model is:
  ```
  [CLS] sentence with [TGT] target [TGT] ... [SEP] aspect category [SEP]
  ```
- Character spans are adjusted accordingly, allowing accurate mapping of the target term back to token positions via `offset_mapping`.

---

**Model Architecture**

- **RoBERTa encoder**  
  `RobertaModel.from_pretrained("FacebookAI/roberta-large")`
- **Input representation**: concatenation of `[CLS]` token vector and **mean-pooled aspect token vector**
- **Dropout**  p = 0.1 
- **Linear**  (2×1024) → 3  

---

**Training**

- Optimizer: **AdamW**, with separate learning rates:
  ```python
  [
    {"params": model.bert.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(), "lr": 2e-5},
  ]
  ```
- Loss: **CrossEntropyLoss**, with optional **label smoothing** and **class weighting** to balance underrepresented labels (label count are as following for the train set [1055, 58, 390])
- Maximum epochs: **10**
- Early stopping based on dev accuracy
- Training and evaluation follow `tester.py`, which supports GPU via `-g <gpu_id>`

---

**Accuracy**

- **91.49 ± 0.69 %** average accuracy on the development dataset using **roberta-large**, best accuracy reached is **92.02 %** .
- We also experimented with **microsoft/deberta-v3-base**, which achieved **92.5 % accuracy**,  
  but could not include it in the final submission because it depends on the `protobuf` library.  
  This leads to a dilemma: although DeBERTa-v3 is explicitly allowed, the assignment does **not allow installing any extra libraries** beyond those listed — and `protobuf` is required by deberta but not installed automatically by `transformers==4.50.3`.
