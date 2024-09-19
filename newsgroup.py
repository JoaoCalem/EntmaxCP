from sklearn.datasets import fetch_20newsgroups
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report

import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd 
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report

# Load the 20 newsgroups dataset
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
data = pd.DataFrame({'text_data': newsgroups.data, 'label': newsgroups.target})

# Visualize newsgroup data object
entry_index = 0
print(f"Text:\n{newsgroups['data'][entry_index]}\n\n")
print(f"Label index: {newsgroups['target'][entry_index]}")
print(f"Label name: {newsgroups['target_names'][newsgroups['target'][entry_index]]}")

# Shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# Split the dataset into training and validation sets (80:20 ratio)
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Initialize BERT tokenizer using the pretrained 'bert-base-uncased' model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
max_seq_len = 128

def tokenize_data(data, tokenizer, max_seq_len):
    input_ids, attention_masks, labels = [], [], []

    # Iterate through each row in the dataset
    for index, row in tqdm(data.iterrows(), total=len(data)):
        # Tokenize the text using BERT's tokenizer with additional parameters
        encoded = tokenizer.encode_plus(
            row["text_data"],
            add_special_tokens=True,  # Add [CLS] and [SEP] tokens
            max_length=max_seq_len,  # Set max sequence length to 128
            padding="max_length",  # Pad shorter sequences to max_seq_len
            truncation=True,  # Truncate longer sequences to max_seq_len
            return_attention_mask=True,  # Return attention masks
        )

        # Append tokenized data to respective lists
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])
        labels.append(row["label"])

    # Convert lists to tensors
    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels)

# Tokenize both the training and validation data using the defined function
train_input_ids, train_attention_masks, train_labels = tokenize_data(train_data, tokenizer, max_seq_len)
val_input_ids, val_attention_masks, val_labels = tokenize_data(val_data, tokenizer, max_seq_len)

batch_size = 16

# Create a TensorDataset object for the training set
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
# Use RandomSampler to shuffle the samples in the dataset
train_sampler = RandomSampler(train_dataset)
# Create DataLoader for the training set using dataset, sampler, and batch size
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# Create a TensorDataset object for the validation set
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
# Use SequentialSampler to process the validation dataset sequentially
val_sampler = SequentialSampler(val_dataset)
# Create DataLoader for the validation set using dataset, sampler, and batch size
val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size)
     
# Load the pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=20,  # Number of labels (20) corresponds to the 20 newsgroups dataset
    output_attentions=False,  # Do not output attention weights
    output_hidden_states=False,  # Do not output hidden states
)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


num_epochs = 3
total_steps = len(train_dataloader) * num_epochs

# Create the optimizer and scheduler for fine-tuning the model
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    # Use a progress bar during training
    progress_bar = tqdm(dataloader, desc="Training", position=0, leave=True)

    # Iterate through each batch in a training epoch
    for batch in progress_bar:
        input_ids, attention_masks, labels = [t.to(device) for t in batch]

        # Zero out gradients before each backward pass
        optimizer.zero_grad()

        # Forward pass to compute the outputs and loss
        outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
        loss = outputs[0]
        total_loss += loss.item()
        
        # Perform a backward pass and update optimizer/scheduler steps
        loss.backward()
        optimizer.step()
        scheduler.step()

        progress_bar.set_description(f"Training - Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_eval_accuracy = 0

    # Use a progress bar during evaluation
    progress_bar = tqdm(dataloader, desc="Evaluation", position=0, leave=True)

    # Iterate through each batch in a validation epoch
    for batch in progress_bar:
        input_ids, attention_masks, labels = [t.to(device) for t in batch]

        # Disable gradient calculations during evaluation
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_masks)

        logits = outputs[0].detach().cpu().numpy()
        label_ids = labels.cpu().numpy()

        # Calculate accuracy for the current batch
        batch_accuracy = accuracy_score(label_ids, logits.argmax(axis=-1))
        total_eval_accuracy += batch_accuracy

        progress_bar.set_description(f"Evaluation - Batch Accuracy: {batch_accuracy:.4f}")

    return total_eval_accuracy / len(dataloader)

# Train and evaluate the model for 'num_epochs' times
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
    val_accuracy = evaluate(model, val_dataloader, device)

    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print(f"Loss: {train_loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")