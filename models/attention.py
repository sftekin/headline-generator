import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size):
        super(Attention, self).__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.linear1 = nn.Linear(encoder_hidden_size, decoder_hidden_size, bias=False)
        self.linear2 = nn.Linear(encoder_hidden_size + decoder_hidden_size, decoder_hidden_size)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

    def score(self, target, source):
        target_batch_size, target_length, target_dims = target.size()

        target = self.linear1(target.view(target_batch_size * target_length, target_dims))

        return torch.bmm(target.view(target_batch_size, target_length, target_dims), source.transpose(1, 2))

    def forward(self, inputs, encoder_outputs, encoder_output_lengths):
        mask = (torch.arange(0, encoder_output_lengths.max()).type_as(encoder_output_lengths)
                .repeat(encoder_output_lengths.numel(), 1).lt(encoder_output_lengths.unsqueeze(1))).unsqueeze(1)

        if next(self.parameters()).is_cuda: mask = mask.cuda()

        scores = self.score(inputs, encoder_outputs)
        batch_size, target_seq_length, source_seq_length = scores.size()

        scores.data.masked_fill_(1 - mask, -float('inf'))
        scores = self.softmax(scores.view(batch_size * target_seq_length, source_seq_length))
        scores = scores.view(batch_size, target_seq_length, source_seq_length)

        attention_outputs = torch.cat([torch.bmm(scores, encoder_outputs), inputs], 2) \
            .view(batch_size * target_seq_length, self.encoder_hidden_size + self.decoder_hidden_size)

        attention_outputs = self.tanh(self.linear_out(attention_outputs).view(batch_size, target_seq_length,
                                                                              self.decoder_hidden_size))

        return attention_outputs.transpose(0, 1).contiguous()
