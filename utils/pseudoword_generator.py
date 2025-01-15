import os
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer


class RoundnessToTextModel(nn.Module):
    def __init__(
        self,
        t5_model_name="sonoisa/t5-base-japanese",
        freeze_t5=False,
        hidden_dim=256,
        output_dim=1472,
        tokenizer=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.device = device

        # Processing layers for roundness value
        self.processing_layers = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, output_dim),
        ).to(self.device)

        # Initialize weights using Kaiming initialization
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # t5 model and tokenizer
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(self.device)
        self.t5.gradient_checkpointing_enable()
        if tokenizer is None:
            self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        else:
            self.tokenizer = tokenizer

        self.freeze_t5_parameters(freeze_t5)

    def freeze_t5_parameters(self, freeze):
        """Freeze all T5 parameters"""
        for param in self.t5.parameters():
            param.requires_grad = not freeze

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
        logits = self.processing_layers(roundness)

        # Pass to t5
        if target_text is not None:
            # Training mode
            target_encoding = self.tokenizer(
                target_text,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.t5(
                inputs_embeds=logits.unsqueeze(1),
                labels=target_encoding['input_ids'],
                attention_mask=torch.ones_like(logits[:, :1]).to(self.device)
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
            outputs = self.t5.generate(
                inputs_embeds=logits.unsqueeze(1),
                max_length=8,
                num_return_sequences=1,
                num_beams=1,
                do_sample=True,
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
            roundness_batch = torch.tensor(trn_roundness[i:i+batch_size].values, dtype=torch.float32).view(-1, 1).to(model.device)
            target_texts_batch = trn_texts[i:i+batch_size].tolist()
            optimizer.zero_grad()
            outputs = model(roundness_batch, target_texts_batch)
            outputs['loss'].backward()
            optimizer.step()
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

        if scheduler:
            scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch+1:>3}/{epochs:>3}, Train Loss: {trn_loss:.4f}, Validation Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}")

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
            [[roundness_value]], dtype=torch.float32
        ).view(-1, 1).to(model.device)
        outputs = model(roundness_tensor)
    return outputs['generated_text'][0]
