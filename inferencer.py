import torch
import speechbrain as sb
import sentencepiece as spm


class InferenceModel(torch.nn.Module):
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
