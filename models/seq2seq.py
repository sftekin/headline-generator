import random
import torch
import torch.nn as nn

from models.embed import Embedding


class Encoder(nn.Module):
    def __init__(self, vocab, hidden_dim, num_layers, device):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.embedding_layer = Embedding(vocab, device)
        self.embed_dim = self.embedding_layer.embed_dim

        self.rnn = nn.LSTM(input_size=self.embed_dim,
                           hidden_size=self.hidden_dim,
                           num_layers=num_layers,
                           batch_first=True)

    def forward(self, input_tensor):
        """
        :param input_tensor: (B,)
        :param hidden: ((l, b, d), (l, b, d))
        :return:
        """
        embed = self.embedding_layer.embedding(input_tensor).float()
        output, hidden = self.rnn(embed)

        return output, hidden


class Attention(nn.Module):
    def __init__(self, enc_hidden, dec_hidden):
        super(Attention, self).__init__()
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden

        self.w_in = nn.Linear(self.dec_hidden, self.enc_hidden, bias=False)
        self.w_out = nn.Linear(self.dec_hidden+self.enc_hidden, self.dec_hidden)

        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

    def forward(self, dec_out, enc_out):
        """
        :param torch.Tensor dec_out: (batch, t_out, d_dec)
        :param torch.Tensor enc_out: (batch, t_in, d_enc)
        :return:
        """
        batch, t_out, d_dec = dec_out.shape
        batch, t_in, d_enc = enc_out.shape

        dec_out_ = dec_out.contiguous().view(batch*t_out, d_dec)
        energy = self.w_in(dec_out_)
        energy = energy.view(batch, t_out, d_enc)

        # swap dimensions: (batch, d_enc, t_in)
        enc_out_ = enc_out.transpose(1, 2)

        # (batch, t_out, d_enc) x (batch, d_enc, t_in) -> (batch, t_out, t_in)
        attn_energies = torch.bmm(energy, enc_out_)

        alpha = self.softmax(attn_energies.view(batch*t_out, t_in))
        alpha = alpha.view(batch, t_out, t_in)

        # (batch, t_out, t_in) * (batch, t_in, d_enc) -> (batch, t_out, d_enc)
        context = torch.bmm(alpha, enc_out)

        # \hat{h_t} = tanh(W [context, dec_out])
        concat_c = torch.cat([context, dec_out], dim=2)
        concat_c = concat_c.view(batch*t_out, d_enc + d_dec)
        concat_c = self.w_out(concat_c)
        attn_h = self.tanh(concat_c.view(batch, t_out, d_dec))

        return attn_h, alpha


class Decoder(nn.Module):
    def __init__(self, vocab, hidden_dim, enc_hidden, num_layers, dropout_prob, device):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.enc_hidden = enc_hidden
        self.num_layers = num_layers
        self.drop_prob = dropout_prob

        self.embedding_layer = Embedding(vocab, device)
        self.embed_dim = self.embedding_layer.embed_dim
        self.rnn = nn.LSTM(input_size=self.embed_dim,
                           hidden_size=self.hidden_dim,
                           num_layers=self.num_layers,
                           batch_first=True)

        self.attention = Attention(dec_hidden=self.hidden_dim, enc_hidden=self.enc_hidden)
        self.out_lin = nn.Linear(self.hidden_dim, self.embedding_layer.vocab_size)
        self.dropout = nn.Dropout(self.drop_prob)

    def forward(self, input_tensor, hidden, enc_out):
        # Teacher-forcing,
        input_tensor = input_tensor.unsqueeze(1)
        embed = self.embedding_layer.embedding(input_tensor).float()
        embed = self.dropout(embed)

        dec_out, dec_hid = self.rnn(embed, hidden)
        attn_out, alpha = self.attention(dec_out, enc_out)
        output = self.out_lin(attn_out)

        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, vocab, device, **model_params):
        super(Seq2Seq, self).__init__()
        self.vocab = vocab
        self.enc_hidden = model_params['encoder_hidden_dim']
        self.dec_hidden = model_params['decoder_hidden_dim']
        self.enc_num_layer = model_params['encoder_num_layer']
        self.dec_num_layer = model_params['decoder_num_layer']
        self.drop_prob = model_params['dropout_prob']
        self.device = device

        self.encoder = Encoder(vocab=self.vocab,
                               hidden_dim=self.enc_hidden,
                               num_layers=self.enc_num_layer,
                               device=self.device)
        self.decoder = Decoder(vocab=self.vocab,
                               hidden_dim=self.dec_hidden,
                               enc_hidden=self.enc_hidden,
                               num_layers=self.dec_num_layer,
                               dropout_prob=self.drop_prob,
                               device=self.device)

    def forward(self, contents, titles, tf_ratio=0.0):
        """

        :param contents: (b, t')
        :param titles: (b, t)
        :param tf_ratio: teacher forcing ratio
        :return:
        """
        batch, title_len = titles.shape

        enc_out, hidden = self.encoder(contents)

        # <start> is mapped to id=2
        dec_inputs = torch.ones(batch, dtype=torch.long) * 2
        dec_inputs = dec_inputs.to(self.device)
        pred, hidden = self.decoder(dec_inputs, hidden, enc_out)

        outputs = [pred]
        for i in range(1, title_len):
            use_tf = random.random() < tf_ratio
            if use_tf:
                dec_inputs = pred.max(dim=2)[1]
                dec_inputs = dec_inputs.squeeze().to(self.device)
            else:
                dec_inputs = titles[:, i]
            pred, hidden = self.decoder(dec_inputs, hidden, enc_out)
            outputs.append(pred)

        self.__repackage_hidden(hidden)

        outputs = torch.cat(outputs, dim=1)

        return outputs

    def __repackage_hidden(self, h):
        """
        Wraps hidden states in new Tensors, to detach them from their history.

        :param h: list of states, e.g [state, state, ...]
        """
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.__repackage_hidden(v) for v in h)

