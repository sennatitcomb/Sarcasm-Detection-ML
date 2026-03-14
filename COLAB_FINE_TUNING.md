## Fine-Tuning on Google Colab

This guide walks through fine-tuning the sarcasm detector on the Joshi et al. (2016) dataset using Google Colab.

### Why Fine-Tune?

The pre-trained `cardiffnlp/twitter-roberta-base-irony` model:
- Trained on Twitter data (domain bias)
- May overfit to irony/sarcasm on Twitter
- Detects many false positives on narrative text (as seen in testing)

Fine-tuning on the **Joshi Book Snippets dataset**:
- Domain-specific sarcasm examples from literature
- Better alignment with narrative/dialogue text
- Reduces false positives
- Improves precision on your use case

### Step 1: Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Copy the code from the **Colab Fine-Tuning Script** section below

### Step 2: Configure GPU

In Colab:
1. Click **Runtime** → **Change runtime type**
2. Select **GPU** (T4 preferred, V100 if available)
3. Click **Save**

### Step 3: Run the Fine-Tuning Script

Run all cells in order. The script will:
1. Install dependencies
2. Download the Joshi dataset
3. Prepare the data
4. Fine-tune the model (2-3 epochs, ~15-20 minutes on T4 GPU)
5. Evaluate on validation set
6. Save the fine-tuned model

### Step 4: Export and Use the Model

After fine-tuning:
1. Model is saved to Colab's `/content/sarcasm-detector-finetuned/`
2. Download it or push to Hugging Face Hub
3. Update your local code to use the new model:
   ```python
   detector = DocumentSarcasmDetector(
       model_name="path/to/sarcasm-detector-finetuned"
   )
   ```

---

## Colab Fine-Tuning Script

Copy and paste this into a Google Colab cell:

```python
# Fine-Tuning Sarcasm Detector on Joshi Dataset
# Run in Google Colab with GPU enabled

import os
import json
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import requests
from zipfile import ZipFile
from io import BytesIO

# Install dependencies
import subprocess
subprocess.run(['pip', 'install', '-q', 'datasets', 'transformers', 'scikit-learn'], check=True)

print("Setting up fine-tuning environment...")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
MODEL_NAME = "cardiffnlp/twitter-roberta-base-irony"
OUTPUT_DIR = "/content/sarcasm-detector-finetuned"
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
MAX_LENGTH = 128

# Download and prepare Joshi dataset
print("\nDownloading Joshi dataset...")

# Download from GitHub
url = "https://raw.githubusercontent.com/bdeiskandaryans/sarc/master/data/processed/balanced/"

datasets_to_download = {
    "train": f"{url}train.json",
    "test": f"{url}test.json"
}

data = {}
for split, url in datasets_to_download.items():
    response = requests.get(url)
    lines = response.text.strip().split('\n')
    data[split] = [json.loads(line) for line in lines if line]

print(f"Train samples: {len(data['train'])}")
print(f"Test samples: {len(data['test'])}")

# Prepare data for training
def prepare_data(raw_data):
    texts = []
    labels = []
    
    for item in raw_data:
        # Extract text from comment
        comment = item.get('comment', '')
        label = item.get('is_sarcastic', 0)
        
        if comment:  # Only include non-empty comments
            texts.append(comment)
            labels.append(label)
    
    return texts, labels

train_texts, train_labels = prepare_data(data['train'])
test_texts, test_labels = prepare_data(data['test'])

print(f"Prepared train: {len(train_texts)} samples")
print(f"Prepared test: {len(test_texts)} samples")

# Split train into train/validation (80/20)
split_idx = int(0.8 * len(train_texts))
val_texts = train_texts[split_idx:]
val_labels = train_labels[split_idx:]
train_texts = train_texts[:split_idx]
train_labels = train_labels[:split_idx]

print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenize
def tokenize_function(texts, labels):
    encodings = tokenizer(
        texts,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': torch.tensor(labels)
    }

print("Tokenizing data...")
train_encodings = tokenize_function(train_texts, train_labels)
val_encodings = tokenize_function(val_texts, val_labels)
test_encodings = tokenize_function(test_texts, test_labels)

# Create PyTorch datasets
class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encodings['labels'][idx]
        }
    
    def __len__(self):
        return len(self.labels)

train_dataset = SarcasmDataset(train_encodings, train_labels)
val_dataset = SarcasmDataset(val_encodings, val_labels)
test_dataset = SarcasmDataset(test_encodings, test_labels)

# Compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Load model
print("\nLoading base model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=2
)
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
    logging_steps=100,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Fine-tune
print("\nStarting fine-tuning (this will take 15-20 minutes)...")
trainer.train()

# Evaluate on test set
print("\nEvaluating on test set...")
test_results = trainer.evaluate(test_dataset)
print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")
print(f"Test F1: {test_results['eval_f1']:.4f}")
print(f"Test Precision: {test_results['eval_precision']:.4f}")
print(f"Test Recall: {test_results['eval_recall']:.4f}")

# Save model
print(f"\nSaving model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nFine-tuning complete!")
print(f"Model saved to: {OUTPUT_DIR}")
print("\nTo use this model locally:")
print(f"  detector = DocumentSarcasmDetector(model_name='{OUTPUT_DIR}')")
```

### Step 5 (Optional): Push to Hugging Face Hub

To make the model shareable:

```python
from huggingface_hub import HfApi, create_repo

# Authenticate with Hugging Face
from huggingface_hub import login
login()  # Follow prompts to authenticate

# Push to Hub
repo_id = "your-username/sarcasm-detector-joshi"
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)

print(f"Model pushed to: https://huggingface.co/{repo_id}")
```

---

## Expected Results

After fine-tuning on Joshi dataset:

**Before fine-tuning (pre-trained only):**
- Jessica example: 12 detections, many false positives
- F1 on Joshi test: ~0.75

**After fine-tuning (3 epochs):**
- Jessica example: 2-3 key detections, fewer false positives
- F1 on Joshi test: ~0.85-0.88
- Better precision for your use case

---

## Further Model Strengthening

If you want to improve beyond basic fine-tuning:

### 1. Ensemble Methods
Combine multiple models for better predictions:

```python
from transformers import pipeline

models_to_ensemble = [
    "cardiffnlp/twitter-roberta-base-irony",  # Pre-trained
    "your-username/sarcasm-detector-joshi",    # Fine-tuned
    "roberta-base",                             # Alternative
]

def ensemble_predict(text):
    scores = []
    for model_name in models_to_ensemble:
        pipe = pipeline("text-classification", model=model_name)
        result = pipe(text, truncation=True)
        score = result[0]['score'] if result[0]['label'] == 'LABEL_1' else 1 - result[0]['score']
        scores.append(score)
    
    return np.mean(scores)  # Average ensemble score
```

### 2. Domain Adaptation
Fine-tune on multiple datasets (Joshi + SemEval + Reddit):

```python
# Combine datasets
all_texts = train_texts + semeval_texts + reddit_texts
all_labels = train_labels + semeval_labels + reddit_labels

# Weight them by domain importance
weights = [1.0, 0.8, 0.6]  # Joshi most important

# Train with sample_weight in Trainer
```

### 3. Data Augmentation
Generate synthetic training examples:

```python
from nlpaug.augmenter.word import ContextualWordEmbsAug, SynonymAug
import nlpaug.augmenter.sentence as nass

augmenter = SynonymAug(aug_p=0.3)
negation_aug = ContextualWordEmbsAug(model_path="bert-base-uncased", action="substitute")

# Create 2x training data with augmentation
augmented_texts = []
augmented_labels = []
for text, label in zip(train_texts, train_labels):
    augmented_texts.append(augmenter.augment(text))
    augmented_labels.append(label)

train_texts_augmented = train_texts + augmented_texts
train_labels_augmented = train_labels + augmented_labels
```

### 4. Multi-Task Learning
Add auxiliary tasks:

```python
class MultiTaskSarcasmModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = AutoModelForSequenceClassification.from_pretrained(base_model)
        
        # Auxiliary heads
        self.sentiment_head = torch.nn.Linear(768, 3)      # Sentiment (pos/neu/neg)
        self.emotion_head = torch.nn.Linear(768, 6)        # Emotions
        self.sarcasm_head = torch.nn.Linear(768, 2)        # Sarcasm (main task)
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base.roberta(input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        
        sarcasm_logits = self.sarcasm_head(pooled)
        sentiment_logits = self.sentiment_head(pooled)
        emotion_logits = self.emotion_head(pooled)
        
        return sarcasm_logits, sentiment_logits, emotion_logits

# Train with combined loss:
# loss = 0.7 * sarcasm_loss + 0.2 * sentiment_loss + 0.1 * emotion_loss
```

### 5. Active Learning
Iteratively improve with user feedback:

```python
def active_learning_loop(unlabeled_texts, model, num_to_label=100):
    """
    1. Get predictions on unlabeled data
    2. Find examples model is least confident about
    3. Have user label those
    4. Add to training set and retrain
    """
    predictions = model(unlabeled_texts)
    confidence = np.max(predictions, axis=1)
    
    # Most uncertain examples
    uncertain_idx = np.argsort(confidence)[:num_to_label]
    
    return [unlabeled_texts[i] for i in uncertain_idx]
```

### 6. Hyperparameter Optimization
Use Bayesian optimization to find best hyperparams:

```python
from optuna import create_study, Trial

def objective(trial: Trial):
    lr = trial.suggest_float('lr', 1e-5, 5e-5)
    batch_size = trial.suggest_int('batch_size', 8, 32)
    
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        num_train_epochs=NUM_EPOCHS,
    )
    
    trainer = Trainer(model=model, args=args, ...)
    metrics = trainer.evaluate()
    
    return metrics['eval_f1']

study = create_study(direction='maximize')
study.optimize(objective, n_trials=20)
```

### 7. Knowledge Distillation
Train a smaller, faster model from the large one:

```python
# Large teacher model
teacher = AutoModelForSequenceClassification.from_pretrained(
    "your-username/sarcasm-detector-joshi"
)

# Small student model
student = AutoModelForSequenceClassification.from_pretrained(
    "distilroberta-base"
)

# Knowledge distillation loss
from torch.nn import KLDivLoss

def distillation_loss(student_logits, teacher_logits, temperature=4.0):
    soft_targets = torch.nn.functional.softmax(teacher_logits / temperature, dim=1)
    soft_predictions = torch.nn.functional.log_softmax(student_logits / temperature, dim=1)
    return KLDivLoss()(soft_predictions, soft_targets)
```

### Priority Order for Improvement:

1. **Start**: Basic fine-tuning on Joshi (15-20 minutes)
2. **Then**: Evaluate and identify weak cases
3. **Next**: Try ensemble with pre-trained + fine-tuned models
4. **Then**: Add data augmentation to training
5. **Advanced**: Domain adaptation with multiple datasets
6. **Final**: Multi-task learning or knowledge distillation

Each step provides diminishing returns but increases robustness.

---

## Troubleshooting

**GPU Out of Memory:**
- Reduce `BATCH_SIZE` from 16 to 8
- Reduce `MAX_LENGTH` from 128 to 96

**Poor Results:**
- Train longer: increase `NUM_EPOCHS` to 5
- Lower learning rate: change `LEARNING_RATE` to 1e-5
- Try different datasets (SemEval, Reddit)

**Model Not Saving:**
- Check Google Drive is mounted
- Ensure `/content/` directory is writable

---

## Next Steps

1. Run fine-tuning in Colab (linked above)
2. Download the model or push to Hub
3. Update your code to use the new model
4. Test on your dataset and measure improvement
5. Iterate with ensemble or augmentation if needed
