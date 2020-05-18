from models.embed import Embedding
from models.attention import Attention
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import optim


class Encoder(nn.Module):
    def __init__(self, word2int, hidden_size, num_layers=1):
        super(Encoder, self).__init__()
        self.embedding = Embedding(word2int)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size=self.embedding.embed_dim, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, inputs, input_lengths):
        outputs, hiddens = self.rnn(pack_padded_sequence(self.embedding(inputs), lengths=input_lengths.numpy()))

        return pad_packed_sequence(outputs)[0], hiddens


class Decoder(nn.Module):
    def __init__(self, word2int, hidden_size, encoder_hidden_size, num_layers=1, dropout_prob=0.2):
        super(Decoder, self).__init__()
        self.embedding = Embedding(word2int)
        self.attention = Attention(encoder_hidden_size, hidden_size)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size=self.embedding.embed_dim, hidden_size=hidden_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, self.embedding.vocab_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs, encoder_hiddens, encoder_outputs, encoder_output_lengths):
        outputs, decoder_hiddens = self.rnn(self.dropout(self.embedding(inputs)), encoder_hiddens)

        # Get attention layer output
        outputs = self.attention(outputs.transpose(0, 1).contiguous(), encoder_outputs.transpose(0, 1).contiguous(),
                                 encoder_output_lengths)

        return self.linear(outputs), decoder_hiddens


class Seq2Seq:
    def __init__(self, word2int, hidden_size, encoder_hidden_size, learning_rate, batch_generator):
        self.encoder = Encoder(word2int, hidden_size)
        self.decoder = Decoder(word2int, hidden_size, encoder_hidden_size)

        if torch.cuda.is_available():
            self.encoder.cuda()
            self.decoder.cuda()

        self.params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(self.params, lr=learning_rate)


    def train(self, epochs):
        pass
