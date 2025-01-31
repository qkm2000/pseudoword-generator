from torch import nn
import torch
import os


class WordTransformer(nn.Module):
    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_layers=4,
        max_length=12
    ):
        super().__init__()
        self.max_length = max_length

        # Input embedding: map single number to d_model dimensions
        self.input_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Position encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_length, d_model))

        # Token embedding for output (26 letters + 3 special tokens)
        self.token_embed = nn.Embedding(29, d_model)

        # Transformer layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, 29)

    def forward(self, x, target=None, teacher_forcing_ratio=0):
        batch_size = x.shape[0]

        # Embed input number
        memory = self.input_embed(x.unsqueeze(-1))
        memory = memory.unsqueeze(1).repeat(1, self.max_length, 1)

        # Initialize decoder input with start token
        decoder_input = torch.full((batch_size, 1), 1, device=x.device)

        outputs = []
        for t in range(self.max_length):
            # Embed current tokens and add position encoding
            tgt = self.token_embed(decoder_input)
            tgt = tgt + self.pos_encoding[:decoder_input.size(1)]

            # Generate output
            decoder_output = self.transformer_decoder(tgt, memory)
            prediction = self.output_proj(decoder_output[:, -1:])
            outputs.append(prediction)

            # Teacher forcing or use model's prediction
            if target is not None and torch.rand(1) < teacher_forcing_ratio:
                next_token = target[:, t:t+1]
            else:
                next_token = prediction.argmax(-1)

            decoder_input = torch.cat([decoder_input, next_token], dim=1)

        return torch.cat(outputs, dim=1)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(
    path,
    d_model=64,
    nhead=4,
    num_layers=4,
    max_length=12
):
    model = WordTransformer(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        max_length=max_length
    )
    model.load_state_dict(torch.load(path))
    return model


def train(
    model,
    optimizer,
    trainLoader,
    valLoader,
    testLoader=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=10,
    patience=3,
    min_delta=1e-4
):
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        num_train_batches = 0

        for i, data in enumerate(trainLoader):
            optimizer.zero_grad()
            number, target_word = data
            number, target_word = number.to(device), target_word.to(device)

            # Forward pass
            output = model(number, target_word)
            output = output.permute(0, 2, 1)

            # Calculate loss
            loss = criterion(output, target_word)
            total_train_loss += loss.item()
            num_train_batches += 1

            # Backward pass
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / num_train_batches

        # Validation phase
        model.eval()
        total_val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for val_data in valLoader:
                val_number, val_target_word = val_data
                val_number = val_number.to(device)
                val_target_word = val_target_word.to(device)

                val_output = model(val_number, val_target_word)
                val_output = val_output.permute(0, 2, 1)

                val_loss = criterion(val_output, val_target_word)
                total_val_loss += val_loss.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / num_val_batches

        # Print epoch summary
        print(f'Epoch {epoch + 1}: Average Training Loss: {avg_train_loss:.4f}, Average Validation Loss: {avg_val_loss:.4f}')

        # Early stopping
        if avg_val_loss + min_delta < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'\nEarly stopping triggered after epoch {epoch}')
            model.load_state_dict(best_model)
            break

    # Final testing phase
    if testLoader is not None:
        total_test_loss = 0
        num_test_batches = 0
        correct_predictions = 0
        total_predictions = 0

        print("\nEvaluating on test set...")
        model.eval()
        with torch.no_grad():
            for test_data in testLoader:
                test_number, test_target_word = test_data
                test_number = test_number.to(device)
                test_target_word = test_target_word.to(device)

                test_output = model(test_number, test_target_word)
                test_output = test_output.permute(0, 2, 1)

                test_loss = criterion(test_output, test_target_word)
                total_test_loss += test_loss.item()
                num_test_batches += 1

                # Calculate accuracy (ignoring padding tokens)
                predictions = test_output.argmax(dim=1)
                mask = test_target_word != 0  # Ignore padding tokens
                correct_predictions += (predictions[mask] == test_target_word[mask]).sum().item()
                total_predictions += mask.sum().item()

        avg_test_loss = total_test_loss / num_test_batches
        test_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        print('Final Test Results:')
        print(f'Average Test Loss: {avg_test_loss:.4f}')
        print(f'Test Accuracy: {test_accuracy:.2%}')

        return {
            'best_val_loss': best_val_loss,
            'final_test_loss': avg_test_loss,
            'test_accuracy': test_accuracy
        }

    return {'best_val_loss': best_val_loss}


def inference(model, roundness, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Generate a pseudoword from a roundness value using the trained model.

    Args:
        model: The trained WordTransformer model
        roundness: A float value representing the roundness
        tokenizer: Tokenizer for decoding the output
        device: Device to run inference on ('cuda' or 'cpu')

    Returns:
        str: The generated pseudoword
    """
    model.eval()
    with torch.no_grad():
        # Convert input to tensor and move to device
        if isinstance(roundness, (int, float)):
            roundness = torch.tensor([roundness], dtype=torch.float32)
        roundness = roundness.to(device)

        # Generate output
        output = model(roundness)
        predicted_tokens = output.argmax(-1)

        # Decode the predicted tokens
        word = tokenizer.decode(predicted_tokens[0])

        return word
