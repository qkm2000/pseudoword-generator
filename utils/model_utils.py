import torch


def trn_step(model, optimizer, roundness_batch, target_texts):
    """
    Single training step
    Args:
        model: RoundnessToTextModel instance
        optimizer: PyTorch optimizer
        roundness_batch (torch.Tensor): Batch of roundness values
        target_texts (list[str]): Batch of target texts
    """
    optimizer.zero_grad()
    outputs = model(roundness_batch, target_texts)
    outputs['loss'].backward()
    optimizer.step()
    return outputs


def train(
    model,
    optimizer,
    trn_roundness,
    val_roundness,
    tst_roundness,
    trn_texts,
    val_texts,
    tst_texts,
    batch_size,
    epochs,
    patience,
    scheduler=None,
):
    """
    Full training function with early stopping
    Args:
        model: RoundnessToTextModel instance
        optimizer: PyTorch optimizer
        trn_roundness (pd.Series): Training roundness values
        val_roundness (pd.Series): Validation roundness values
        tst_roundness (pd.Series): Test roundness values
        trn_texts (pd.Series): Training target texts
        val_texts (pd.Series): Validation target texts
        tst_texts (pd.Series): Test target texts
        batch_size (int): Batch size
        epochs (int): Number of epochs
        patience (int): Number of epochs to wait for improvement before stopping
        scheduler: PyTorch learning rate scheduler (optional)
    """
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        trn_loss = 0.0
        for i in range(0, len(trn_roundness), batch_size):
            roundness_batch = torch.tensor(trn_roundness[i:i+batch_size].values, dtype=torch.float32).view(-1, 1).to(model.device)
            target_texts_batch = trn_texts[i:i+batch_size].tolist()
            outputs = trn_step(model, optimizer, roundness_batch, target_texts_batch)
            trn_loss += outputs['loss'].item()
        trn_loss /= len(trn_roundness) // batch_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(val_roundness), batch_size):
                roundness_batch = torch.tensor(val_roundness[i:i+batch_size].values, dtype=torch.float32).view(-1, 1).to(model.device)
                target_texts_batch = val_texts[i:i+batch_size].tolist()
                outputs = model(roundness_batch, target_texts_batch)
                val_loss += outputs['loss'].item()

        val_loss /= len(val_roundness) // batch_size
        print(f"Epoch {epoch+1:>3}/{epochs:>3}, Train Loss: {trn_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if scheduler:
            scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            model.load_state_dict(best_model_state)
            break

    # Test evaluation
    model.eval()
    tst_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(tst_roundness), batch_size):
            roundness_batch = torch.tensor(tst_roundness[i:i+batch_size].values, dtype=torch.float32).view(-1, 1).to(model.device)
            target_texts_batch = tst_texts[i:i+batch_size].tolist()
            outputs = model(roundness_batch, target_texts_batch)
            tst_loss += outputs['loss'].item()

    tst_loss /= len(tst_roundness) // batch_size
    print(f"Test Loss: {tst_loss:.4f}")


def inference(model, roundness_value):
    """
    Run inference for a single roundness value
    Args:
        model: RoundnessToTextModel instance
        roundness_value (float): Single roundness value
    """
    model.eval()
    with torch.no_grad():
        roundness_tensor = torch.tensor([[roundness_value]], dtype=torch.float32).view(-1, 1).to(model.device)
        outputs = model(roundness_tensor)
    return outputs['generated_text'][0]
