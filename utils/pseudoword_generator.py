import os
import torch
import torch.nn as nn
from transformers import ByT5Tokenizer, T5ForConditionalGeneration


class RoundnessToTextModel(nn.Module):
    def __init__(
        self,
        byt5_model_name="google/byt5-small",
        freeze_byt5=False,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.device = device

        # Processing layers for roundness value
        self.processing_layers = nn.Sequential(
            nn.Linear(1, 512),
            nn.Tanh(),
            nn.Linear(512, 1472),
            nn.BatchNorm1d(1472),
            nn.Tanh(),
            nn.Linear(1472, 1472),
        ).to(self.device)

        # Attention layer
        self.attention_layer = nn.MultiheadAttention(embed_dim=1472, num_heads=16).to(self.device)

        # Xavier initialization for the processing layers
        for layer in self.processing_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        # ByT5 model
        self.byt5 = T5ForConditionalGeneration.from_pretrained(byt5_model_name).to(self.device)
        self.tokenizer = ByT5Tokenizer.from_pretrained(byt5_model_name)

        if freeze_byt5:
            self.freeze_byt5_parameters()

    def freeze_byt5_parameters(self):
        """Freeze all ByT5 parameters"""
        for param in self.byt5.parameters():
            param.requires_grad = False

    def forward(self, roundness, target_text=None):
        """
        Forward pass of the model
        Args:
            roundness (torch.Tensor): Shape [batch_size, 1]
            target_text (list[str], optional): Target texts for loss calculation
        Returns:
            dict containing generated text and loss (if target_text provided)
        """
        # Process roundness through layers
        roundness = roundness.to(self.device)
        bytes_logits = self.processing_layers(roundness)

        # Apply attention layer
        bytes_logits = bytes_logits.unsqueeze(1)  # Add sequence dimension
        attn_output, _ = self.attention_layer(bytes_logits, bytes_logits, bytes_logits)
        bytes_logits = attn_output.squeeze(1)  # Remove sequence dimension

        # Pass to ByT5
        if target_text is not None:
            # Training mode
            target_encoding = self.tokenizer(
                target_text,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.byt5(
                inputs_embeds=bytes_logits.unsqueeze(1),
                labels=target_encoding['input_ids']
            )

            loss = outputs.loss
            logits = outputs.logits

            # Generate text for return value
            generated_ids = torch.argmax(logits, dim=-1)
            generated_text = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

            return {
                'generated_text': generated_text,
                'loss': loss
            }

        else:
            # Inference mode
            outputs = self.byt5.generate(
                inputs_embeds=bytes_logits.unsqueeze(1),
                max_length=20,
                num_return_sequences=1,
                num_beams=1,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
            )

            generated_text = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )

            return {
                'generated_text': generated_text,
                'loss': None
            }


def save_model(model, directory="outputs/", filename="model_v0x.pth"):
    """Save model to disk"""
    path = os.path.join(directory, filename)

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(directory="outputs/", filename="model_v0x.pth"):
    """Load model from disk"""
    path = os.path.join(directory, filename)
    model = RoundnessToTextModel()
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model


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
            roundness_batch = torch.tensor(
                trn_roundness[i:i+batch_size].values, dtype=torch.float32).view(-1, 1).to(model.device)
            target_texts_batch = trn_texts[i:i+batch_size].tolist()
            outputs = trn_step(
                model, optimizer, roundness_batch, target_texts_batch)
            trn_loss += outputs['loss'].item()
        trn_loss /= len(trn_roundness) // batch_size

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(val_roundness), batch_size):
                roundness_batch = torch.tensor(
                    val_roundness[i:i+batch_size].values, dtype=torch.float32).view(-1, 1).to(model.device)
                target_texts_batch = val_texts[i:i+batch_size].tolist()
                outputs = model(roundness_batch, target_texts_batch)
                val_loss += outputs['loss'].item()

        val_loss /= len(val_roundness) // batch_size
        print(f"Epoch {epoch+1:>3}/{epochs:>3}, Train Loss: {
              trn_loss:.4f}, Validation Loss: {val_loss:.4f}")

        if scheduler:
            scheduler.step()

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
            roundness_batch = torch.tensor(
                tst_roundness[i:i+batch_size].values, dtype=torch.float32).view(-1, 1).to(model.device)
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
        roundness_tensor = torch.tensor(
            [[roundness_value]], dtype=torch.float32).view(-1, 1).to(model.device)
        outputs = model(roundness_tensor)
    return outputs['generated_text'][0]
