from sklearn.model_selection import KFold
import os
import torch
from torch import nn
from transformers import ByT5Tokenizer


class roundness_determiner(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        self.tokenizer = ByT5Tokenizer.from_pretrained("google/byt5-small")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(16, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size * 2, 1),
            torch.nn.Sigmoid()
        )
        # Initialize weights using Xavier initialization
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, token_ids):
        return self.mlp(token_ids.float())

    def inference(self, texts):
        """
        Perform inference on a single string or list of strings.
        Returns float values representing roundness scores.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize the input texts
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=16,
            truncation=True
        )

        # Move to the same device as the model
        token_ids = tokens["input_ids"].to(self.mlp[0].weight.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.forward(token_ids)

        # Convert to float values
        results = outputs.cpu().numpy().flatten()

        return results[0] if len(texts) == 1 else results


def train(
    model,
    trn_roundness,
    trn_texts,
    val_roundness,
    val_texts,
    batch_size,
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

    # Training tracking
    best_val_loss = float("inf")
    epochs_no_improve = 0
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': None
    }

    def tokenize_batch(texts):
        return model.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=16,
            truncation=True
        )["input_ids"].to(device)

    def evaluate(roundness, texts):
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
                token_ids = tokenize_batch(texts_batch)
                outputs = model.forward(token_ids)
                loss = criterion(outputs, roundness_batch)
                total_loss += loss.item() * (batch_end - i)
        return total_loss / len(roundness)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for i in range(0, len(trn_roundness), batch_size):
            batch_end = min(i + batch_size, len(trn_roundness))
            roundness_batch = torch.tensor(
                trn_roundness[i:batch_end].values,
                dtype=torch.float32,
                device=device
            ).view(-1, 1)
            texts_batch = trn_texts[i:batch_end].tolist()

            optimizer.zero_grad()
            token_ids = tokenize_batch(texts_batch)
            outputs = model.forward(token_ids)
            loss = criterion(outputs, roundness_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * (batch_end - i)

        train_loss /= len(trn_roundness)
        training_history['train_loss'].append(train_loss)

        # Validation phase
        val_loss = evaluate(val_roundness, val_texts)
        training_history['val_loss'].append(val_loss)

        # Print progress
        print(f"Epoch {epoch+1:>4}/{epochs:<4}", end=" ")
        print(f"Train Loss: {train_loss:.4f}", end=" ")
        print(f"Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best model
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        if scheduler is not None:
            scheduler.step()

    # Load best model state
    model.load_state_dict(best_model_state)

    # Final evaluation on test set
    if tst_roundness is not None:
        test_loss = evaluate(tst_roundness, tst_texts)
        training_history['test_loss'] = test_loss
        print(f"\nFinal Test Loss: {test_loss:.4f}")

    return training_history


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

    # Training tracking
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'test_loss': None
    }

    def tokenize_batch(texts):
        return model.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            max_length=16,
            truncation=True
        )["input_ids"].to(device)

    def evaluate(roundness, texts):
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
                token_ids = tokenize_batch(texts_batch)
                outputs = model.forward(token_ids)
                loss = criterion(outputs, roundness_batch)
                total_loss += loss.item() * (batch_end - i)
        return total_loss / len(roundness)

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_idx = 1
    for train_idx, val_idx in kf.split(roundness):
        print(f"\nFold {fold_idx}/{k}")
        fold_idx += 1

        # Split data into training and validation sets
        trn_roundness, val_roundness = roundness.iloc[train_idx], roundness.iloc[val_idx]
        trn_texts, val_texts = texts.iloc[train_idx], texts.iloc[val_idx]

        # Initialize early stopping variables
        best_val_loss = float("inf")
        epochs_no_improve = 0

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for i in range(0, len(trn_roundness), batch_size):
                batch_end = min(i + batch_size, len(trn_roundness))
                roundness_batch = torch.tensor(
                    trn_roundness[i:batch_end].values,
                    dtype=torch.float32,
                    device=device
                ).view(-1, 1)
                texts_batch = trn_texts[i:batch_end].tolist()

                optimizer.zero_grad()
                token_ids = tokenize_batch(texts_batch)
                outputs = model.forward(token_ids)
                loss = criterion(outputs, roundness_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * (batch_end - i)

            train_loss /= len(trn_roundness)

            # Validation phase
            val_loss = evaluate(val_roundness, val_texts)

            # Print progress
            print(f"Epoch {epoch+1:>4}/{epochs:<4}", end=" ")
            print(f"Train Loss: {train_loss:.4f}", end=" ")
            print(f"Val Loss: {val_loss:.4f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                # Save best model state
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

            if scheduler is not None:
                scheduler.step()

        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(best_val_loss)

        # Load best model state for the fold
        model.load_state_dict(best_model_state)

    # Final evaluation on test set
    if tst_roundness is not None:
        test_loss = evaluate(tst_roundness, tst_texts)
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


def load_model(directory="outputs/", filename="roundness_determiner_v0x.pth"):
    """Load model from disk"""
    path = os.path.join(directory, filename)
    model = roundness_determiner()
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model


if __name__ == "__main__":
    model = roundness_determiner()
    text = "bouba"
    output = model.forward(text)
    print(output)
