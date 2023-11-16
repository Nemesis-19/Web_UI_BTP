import torch
import speechbrain as sb
import sentencepiece as spm


class CustomModel(torch.nn.Module):
    """Basic LSTM model for language modeling.

    Arguments
    ---------
    embedding_dim : int
        The dimension of the embeddings.The input indexes are transformed into
        a latent space with this dimensionality.
    rnn_size : int
        Number of neurons to use in rnn (for each direction -> and <-).
    layers : int
        Number of RNN layers to use.
    output_dim : int
        Dimensionality of the output.
    return_hidden : bool
        If True, returns the hidden state of the RNN as well.
    """

    def __init__(
        self,
        embedding_dim=256,
        rnn_size=512,
        layers=2,
        output_dim=1000,
        return_hidden=False,
    ):
        super().__init__()
        self.return_hidden = return_hidden
        self.reshape = False

        # Embedding model
        self.embedding = sb.nnet.embedding.Embedding(
            num_embeddings=output_dim, embedding_dim=embedding_dim
        )

        # LSTM
        self.rnn = torch.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=rnn_size,
            bidirectional=False,
            num_layers=layers,
        )

        # Final output transformation + softmax
        self.out = sb.nnet.linear.Linear(
            input_size=rnn_size, n_neurons=output_dim
        )
        self.log_softmax = sb.nnet.activations.Softmax(apply_log=True)

    def forward(self, x, hx=None):
        """Forward pass that predicts only the next token based on the input sequence."""
        x = self.embedding(x)

        if len(x.shape) == 2:
            x = x.unsqueeze(dim=1)
            self.reshape = True

        x, hidden = self.rnn(x, hx)

        last_time_step_output = x[-1, :, :].unsqueeze(0)

        x = self.out(last_time_step_output)

        x = self.log_softmax(x)

        if self.reshape:
            x = x.squeeze(dim=1)

        return x, hidden


state_dict = torch.load(
    "/content/speechbrain/templates/speech_recognition/LM/results/RNNLM/save/CKPT+2023-11-09+14-22-39+00/model.ckpt"
)

# for key, value in state_dict.items():
#   print(key, value.shape)

model = CustomModel()
model.load_state_dict(state_dict)
model.eval()

# load the sentencepiece model
sp = spm.SentencePieceProcessor()
sp.load(
    "/content/speechbrain/templates/speech_recognition/Tokenizer/save/1000_unigram.model"
)
# generate a sentence
# text = "this is a test sentence"
# ids = sp.encode_as_ids(text)
# ids = torch.tensor(ids, dtype=torch.long)
# ids = ids.unsqueeze(0)
# output = model(ids)
# output = output.squeeze(0)
# output = torch.argmax(output, dim=-1)
# print(output)
# output = sp.DecodeIds(output)
# print(output)

# Assume max_length_to_generate and model are already defined
max_length_to_generate = 100  # You can choose a different value
# Assuming sp is your loaded SentencePiece model
# seed_text = "WHO CAME IN TO BRING HIM FOOD AND DRINK AND BY THE MEANS OF THIS"
# seed_text = "THIS WAS A BOLD AND RASH UNDERTAKING THE TWO PIRATES WERE BOLD AND RASH ENOUGH"
# seed_text = "THESE FISH HE SOLD TO THE OFFICERS AND WE ARE"
# seed_text = "RENOWNED PIRATE FROM BRAZIL MUST HAVE BEEN A TERRIBLE FELLOW"
seed_text = "IS THERE EVEN A GOD?"
seed_tokens = sp.encode_as_ids(seed_text)

print(seed_text)
print(seed_tokens)

# Convert seed tokens to a PyTorch tensor and add a batch dimension
# input_tensor = torch.tensor([seed_tokens[-1]], dtype=torch.long).unsqueeze(0)
# print(input_tensor.shape)
input_tensor = torch.tensor([seed_tokens], dtype=torch.long)
# print(input_tensor.shape)

# print(input_tensor.shape)

# We will store our generated tokens here, starting with the seed
generated_tokens = seed_tokens

# Initialize the hidden state for the first forward pass
hidden_size = model.rnn.hidden_size
num_layers = model.rnn.num_layers
num_directions = 2 if model.rnn.bidirectional else 1

# Initialize hidden state (h) and cell state (c)
hidden = (
    torch.zeros(num_layers * num_directions, len(seed_tokens), hidden_size),
    torch.zeros(num_layers * num_directions, len(seed_tokens), hidden_size),
)

# hidden = None
# Generation loop
with torch.no_grad():
    for _ in range(max_length_to_generate):
        # Forward pass through the model
        logits, hidden = model(input_tensor, hidden)

        # print(logits.shape)
        # for p in hidden:
        #   print(p.shape)
        #   break

        # hidden = list(hidden)

        # logits, h = model(input_tensor)

        # for p in hidden:
        #   print(p.shape)
        #   break

        # print(logits[-1].shape)

        # print(logits.shape)

        # logits should be 2-dimensional [batch_size, vocab_size]
        # We only use the last output for the next token prediction
        logits = logits[:, -1, :].squeeze(0)

        # print(logits.shape)

        indices_to_remove = logits < torch.topk(logits, 10)[0][..., -1, None]
        logits[indices_to_remove] = -float("Inf")

        # print(logits.shape)

        # Convert the logits to probabilities
        probabilities = torch.softmax(logits, dim=-1)

        # Sample from the probability distribution
        next_token_id = torch.multinomial(
            probabilities, num_samples=1
        ).squeeze()
        if not next_token_id:
            next_token_id = torch.argmax(probabilities, dim=-1)

        # Append the predicted token to the sequence
        generated_tokens.append(next_token_id.item())

        # print(input_tensor.shape)

        # Update the input tensor to contain only the new token, preserving batch dimension
        # input_tensor = torch.cat((input_tensor, next_token_id.unsqueeze(0).unsqueeze(0)), dim=1)
        next_token_id = next_token_id.unsqueeze(0).unsqueeze(0)

        # print(next_token_id.shape)
        # print(input_tensor.shape)

        input_tensor = torch.concat((input_tensor[:, 1:], next_token_id), dim=1)

        # print(input_tensor.shape)

        # Check if the end-of-sentence token was generated (assuming you have EOS token id)
        if next_token_id.item() == sp.eos_id():
            break

# Decode the generated tokens to text
generated_text = sp.decode_ids(generated_tokens)
print(generated_tokens)
print(generated_text)
