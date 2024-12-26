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
            nn.Linear(1, 1472),
        ).to(self.device)

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
        # print(f"after processing layers: {bytes_logits[0][:5]}")

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
            print(f"first few logits: {bytes_logits[0][:5]}")
            outputs = self.byt5.generate(
                inputs_embeds=bytes_logits.unsqueeze(1),
                max_length=100,
                num_return_sequences=1,
            )
            print(outputs)

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


def load_model(directory="outputs/", filename="model_v0x.pth"):
    """Load model from disk"""
    path = os.path.join(directory, filename)
    model = RoundnessToTextModel()
    model.load_state_dict(torch.load(path))
    return model
