from sklearn.model_selection import KFold
import os
import torch
from torch import nn
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel


class RoundnessDeterminerBERT(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_size=768, freeze_base=True):
        """
        Args:
            model_name (str): Name of the pretrained model to use 
                            ('bert-base-uncased', 'roberta-base', etc.)
            hidden_size (int): Size of hidden layers in the MLP
        """
        super().__init__()

        # Initialize tokenizer and base model
        if "roberta" in model_name:
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.base_model = RobertaModel.from_pretrained(model_name)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.base_model = BertModel.from_pretrained(model_name)

        # Freeze the base model parameters (optional)
        for param in self.base_model.parameters():
            if freeze_base:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # MLP for processing the [CLS] token embedding
        self.mlp = torch.nn.Sequential(
            # 768 is BERT/RoBERTa base hidden size
            torch.nn.Linear(768, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size * 3),
            torch.nn.BatchNorm1d(hidden_size * 3),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size * 3, hidden_size * 2),
            torch.nn.BatchNorm1d(hidden_size * 2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid()
        )

        # Initialize weights using Xavier initialization
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, input_ids, attention_mask):
        # Get the BERT/RoBERTa embeddings
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Get the [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        # Pass through MLP
        return self.mlp(cls_embedding)

    def inference(self, texts):
        """
        Perform inference on a single string or list of strings.
        Returns float values representing roundness scores.
        """
        self.eval()
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize the input texts
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  # BERT/RoBERTa max length
        )

        # Move to the same device as the model
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        # Get predictions
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)

        # Convert to float values
        results = outputs.cpu().numpy().flatten()

        return results[0] if len(texts) == 1 else results


def tokenize_batch(model, texts):
    return model.tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        max_length=16,
        truncation=True
    )


def evaluate(criterion, model, roundness, texts, device="cuda" if torch.cuda.is_available() else "cpu", batch_size=32):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(roundness), batch_size):
            batch_end = min(i + batch_size, len(roundness))
            roundness_batch = torch.tensor(
                roundness[i:batch_end].values,
                dtype=torch.float32,
                device=device
            ).view(-1, 1)
            texts_batch = texts[i:batch_end].tolist()
            inputs = tokenize_batch(model, texts_batch)
            outputs = model.forward(inputs["input_ids"].to(device), inputs["attention_mask"].to(device))
            loss = criterion(outputs, roundness_batch)
            total_loss += loss.item() * (batch_end - i)
    return total_loss / len(roundness)


def train_kfold(
    model,
    roundness,
    texts,
    batch_size,
    k=4,
    tst_roundness=None,
    tst_texts=None,
    optimizer=None,
    epochs=100,
    patience=5,
    scheduler=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # Initialize optimizer if not provided
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model = model.to(device)

    # Loss function
    criterion = torch.nn.BCELoss()

    # Enhanced training history tracking
    training_history = {
        'folds': [],  # Track each fold separately
        'best_val_loss': float('inf'),
        'best_fold': None,
        'test_loss': None
    }

    # Save initial model state to reset between folds
    initial_model_state = model.state_dict().copy()

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(roundness), 1):
        print(f"\nFold {fold_idx}/{k}")

        # Reset model for each fold
        model.load_state_dict(initial_model_state)

        # Split data into training and validation sets
        trn_roundness = roundness.iloc[train_idx]
        val_roundness = roundness.iloc[val_idx]
        trn_texts = texts.iloc[train_idx]
        val_texts = texts.iloc[val_idx]

        # Initialize fold history
        fold_history = {
            'train_losses': [],
            'val_losses': [],
            'best_epoch': 0,
            'best_val_loss': float('inf')
        }

        # Reset optimizer and scheduler for each fold
        if optimizer is not None:
            optimizer = type(optimizer)(model.parameters(), **optimizer.defaults)
        if scheduler is not None:
            scheduler = type(scheduler)(optimizer, scheduler.step_size)

        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            num_batches = 0

            # Use DataLoader for better memory efficiency
            train_dataset = list(zip(trn_texts.values, trn_roundness.values))
            for i in range(0, len(train_dataset), batch_size):
                batch = train_dataset[i:i + batch_size]
                texts_batch, roundness_batch = zip(*batch)

                roundness_batch = torch.tensor(
                    roundness_batch,
                    dtype=torch.float32,
                    device=device
                ).view(-1, 1)

                optimizer.zero_grad()
                inputs = tokenize_batch(model, list(texts_batch))
                outputs = model(inputs["input_ids"].to(device), inputs["attention_mask"].to(device))
                loss = criterion(outputs, roundness_batch)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            avg_train_loss = train_loss / num_batches

            # Validation phase
            with torch.no_grad():
                val_loss = evaluate(
                    criterion, model, val_roundness, val_texts,
                    batch_size=batch_size, device=device
                )

            # Update fold history
            fold_history['train_losses'].append(avg_train_loss)
            fold_history['val_losses'].append(val_loss)

            # Print progress with more detail
            print(f"Epoch {epoch+1:>4}/{epochs:<4} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Best Val: {fold_history['best_val_loss']:.4f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                fold_history['best_val_loss'] = val_loss
                fold_history['best_epoch'] = epoch
                epochs_no_improve = 0
                best_model_state = model.state_dict().copy()
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

            if scheduler is not None:
                scheduler.step()

        # Store fold history
        training_history['folds'].append(fold_history)

        # Update best overall model if this fold performed better
        if best_val_loss < training_history['best_val_loss']:
            training_history['best_val_loss'] = best_val_loss
            training_history['best_fold'] = fold_idx
            training_history['best_model_state'] = best_model_state.copy()

    # Load the best model state from all folds
    model.load_state_dict(training_history['best_model_state'])

    # Final evaluation on test set
    if tst_roundness is not None and tst_texts is not None:
        with torch.no_grad():
            test_loss = evaluate(
                criterion, model, tst_roundness, tst_texts,
                batch_size=batch_size, device=device
            )
        training_history['test_loss'] = test_loss
        print(f"\nFinal Test Loss: {test_loss:.4f}")

    return training_history


def save_model(model, directory="outputs/", filename="roundness_determiner_v0x.pth"):
    """Save model to disk"""
    path = os.path.join(directory, filename)

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(directory="outputs/", filename="roundness_determiner_v0x.pth", model_name="bert-base-uncased"):
    """Load model from disk"""
    path = os.path.join(directory, filename)
    model = RoundnessDeterminerBERT(model_name=model_name)
    model.load_state_dict(torch.load(path, weights_only=True))
    print(f"Model loaded from {path}")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


if __name__ == "__main__":
    model = RoundnessDeterminerBERT()
    text = "bouba"
    output = model.forward(text)
    print(output)
