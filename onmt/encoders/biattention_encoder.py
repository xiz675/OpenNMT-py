import torch.nn as nn
import torch
from onmt.encoders.encoder import EncoderBase
from onmt.utils.misc import sequence_mask
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class BiAttEncoder(EncoderBase):
    def __init__(self, rnn_type, bidirectional, num_layers, hidden_size,
                 dropout, embeddings):
        super(BiAttEncoder, self).__init__()

        assert bidirectional
        assert num_layers == 2
        self.rnn_type = rnn_type
        hidden_size = hidden_size // 2
        self.real_hidden_size = hidden_size
        self.no_pack_padded_seq = False
        self.bidirectional = bidirectional

        self.embeddings = embeddings
        input_size_list = [embeddings.embedding_size] + [4 * hidden_size] * (num_layers - 2)
        self.src_rnn_list = nn.ModuleList([getattr(nn, rnn_type)(  # getattr(nn, rnn_type) = torch.nn.modules.rnn.LSTM
            input_size=input_sizei,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=self.bidirectional) for input_sizei in input_size_list])
        self.merge_rnn = getattr(nn, rnn_type)(  # getattr(nn, rnn_type) = torch.nn.modules.rnn.LSTM
            input_size=4 * hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            bidirectional=self.bidirectional)
        self.combine_output = nn.Linear(4 * hidden_size, 2 * hidden_size, bias=True)
        self.combine_hidden = nn.Linear(2 * hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(p=dropout)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings)

    def forward(self, src_input, ans_input, lengths=None, hidden=None):
        """
        a reading comprehension style framework for encoder
        (src->BiRNN->src_outputs, ans->BiRNN->ans_outputs)->(match->src_attn_output, ans_attn_output)
        -> (src_outputs+src_attn_output, ans_outputs+ans_attn_output)->Decoder
        src_input: post
        ans_input: conversation
        lengths: tuple (src_lengths, ans_lengths)
        """

        assert isinstance(lengths, tuple)
        src_lengths, ans_lengths = lengths

        self._check_args(src_input, src_lengths, hidden)
        self._check_args(ans_input, ans_lengths, hidden)

        src_input = self.embeddings(src_input)  # [src_seq_len, batch_size, emb_dim]
        # src_len, batch, emb_dim = src_input.size()

        ans_input = self.embeddings(ans_input)  # [ans_seq_len, batch_size, emb_dim]

        # ans_len, batch, emb_dim = ans_input.size()

        # match layer

        def ans_match(src_seq, ans_seq):
            import torch.nn.functional as F
            BF_ans_mask = sequence_mask(ans_lengths)  # [batch, ans_seq_len]
            BF_src_mask = sequence_mask(src_lengths)  # [batch, src_seq_len]
            BF_src_outputs = src_seq.transpose(0, 1)  # [batch, src_seq_len, 2*hidden_size]
            BF_ans_outputs = ans_seq.transpose(0, 1)  # [batch, ans_seq_len, 2*hidden_size]

            # compute bi-att scores
            src_scores = BF_src_outputs.bmm(BF_ans_outputs.transpose(2, 1))  # [batch, src_seq_len, ans_seq_len]
            ans_scores = BF_ans_outputs.bmm(BF_src_outputs.transpose(2, 1))  # [batch, ans_seq_len, src_seq_len]

            # mask padding
            Expand_BF_ans_mask = BF_ans_mask.unsqueeze(1).expand(src_scores.size())  # [batch, src_seq_len, ans_seq_len]
            src_scores.data.masked_fill_(~(Expand_BF_ans_mask).bool(), -float('inf'))
            # src_scores = torch.ones(src_scores.shape).to(ans_seq.device)

            Expand_BF_src_mask = BF_src_mask.unsqueeze(1).expand(ans_scores.size())  # [batch, ans_seq_len, src_seq_len]
            ans_scores.data.masked_fill_(~(Expand_BF_src_mask).bool(), -float('inf'))

            # normalize with softmax
            src_alpha = F.softmax(src_scores, dim=2)  # [batch, src_seq_len, ans_seq_len]
            ans_alpha = F.softmax(ans_scores, dim=2)  # [batch, ans_seq_len, src_seq_len]

            # take the weighted average
            BF_src_matched_seq = src_alpha.bmm(BF_ans_outputs)  # [batch, src_seq_len, 2*hidden_size]
            src_matched_seq = BF_src_matched_seq.transpose(0, 1)  # [src_seq_len, batch, 2*hidden_size]

            BF_ans_matched_seq = ans_alpha.bmm(BF_src_outputs)  # [batch, ans_seq_len, 2*hidden_size]
            ans_matched_seq = BF_ans_matched_seq.transpose(0, 1)  # [src_seq_len, batch, 2*hidden_size]

            return src_matched_seq, ans_matched_seq

        # sort ans w.r.t lengths
        sorted_ans_lengths, idx_sort = torch.sort(ans_lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        idx_sort = Variable(idx_sort)  # [batch_size]
        idx_unsort = Variable(idx_unsort)  # [batch_size]

        # for src_rnn, ans_rnn in zip(self.src_rnn_list, self.ans_rnn_list):
        for layer_idx in range(len(self.src_rnn_list)):
            src_rnn = self.src_rnn_list[layer_idx]
            packed_input = src_input
            if src_lengths is not None and not self.no_pack_padded_seq:
                # Lengths data is wrapped inside a Variable.
                packed_input = pack(src_input, src_lengths.view(-1).tolist())
            # forward
            src_outputs, src_hidden = src_rnn(packed_input, hidden)
            if src_lengths is not None and not self.no_pack_padded_seq:
                # output
                src_outputs = unpack(src_outputs)[0]

            packed_ans_input = ans_input.index_select(1, idx_sort)
            if sorted_ans_lengths is not None and not self.no_pack_padded_seq:
                packed_ans_input = pack(packed_ans_input, sorted_ans_lengths.view(-1).tolist())

            ans_outputs, ans_hidden = src_rnn(packed_ans_input, hidden)
            if sorted_ans_lengths is not None and not self.no_pack_padded_seq:
                ans_outputs = unpack(ans_outputs)[0]
                ans_outputs = ans_outputs.index_select(1, idx_unsort)
                # h, c
                if self.rnn_type == 'LSTM':
                    ans_hidden = tuple([ans_hidden[i].index_select(1, idx_unsort) for i in range(2)])
                elif self.rnn_type == 'GRU':
                    ans_hidden = ans_hidden.index_select(1, idx_unsort)  # [2, batch, hidden_size]

            src_matched_seq, ans_matched_seq = ans_match(src_outputs, ans_outputs)

            src_outputs = torch.cat([src_outputs, src_matched_seq], dim=-1)  # [src_seq_len, batch_size, 4 * hidden]
            ans_outputs = torch.cat([ans_outputs, ans_matched_seq], dim=-1)  # [ans_seq_len, batch_size, 4 * hidden]

            outputs = self.combine_output(
                torch.cat([src_outputs, ans_outputs], dim=0))  # [src_seq_len+ans_seq_len, batch_size, 2 * hidden]
            hidden = self.combine_hidden(torch.cat([src_hidden, ans_hidden], dim=-1))  # [2, batch_size, hidden]
        return hidden, outputs, lengths