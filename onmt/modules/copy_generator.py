import torch
import torch.nn as nn

from onmt.utils.misc import aeq
from onmt.utils.loss import NMTLossCompute


def collapse_copy_scores(scores, batch, tgt_vocab, src_vocabs=None, conv_vocabs=None,
                         batch_dim=1, batch_offset=None):
    """
    Given scores from an expanded dictionary
    corresponeding to a batch, sums together copies,
    with a dictionary word when it is ambiguous.
    """
    offset1 = len(tgt_vocab)
    offset2 = batch.src_map.size()[-1] + offset1

    for b in range(scores.size(batch_dim)):
        blank = []
        fill = []
        blank_copy = []
        fill_copy = []

        if src_vocabs is None:
            src_vocab = batch.src_ex_vocab[b]
        else:
            batch_id = batch_offset[b] if batch_offset is not None else b
            index = batch.indices.data[batch_id]
            src_vocab = src_vocabs[index]

        if conv_vocabs is None:
            conv_vocab = batch.conv_ex_vocab[b]
        else:
            batch_id = batch_offset[b] if batch_offset is not None else b
            index = batch.indices.data[batch_id]
            conv_vocab = conv_vocabs[index]

        for i in range(1, len(src_vocab)):
            sw = src_vocab.itos[i]
            ti = tgt_vocab.stoi[sw]
            if ti != 0:
                blank.append(offset1 + i)
                fill.append(ti)

        for j in range(1, len(conv_vocab)):
            sw = conv_vocab.itos[j]
            si = src_vocab.stoi[sw]
            ti = tgt_vocab.stoi[sw]
            if ti != 0:
                blank.append(offset2 + j)
                fill.append(ti)
            if ti == 0 and si != 0:
                blank_copy.append(offset2 + j)
                fill_copy.append(offset1 + si)

        score = scores[:, b] if batch_dim == 1 else scores[b]

        if blank:
            blank = torch.Tensor(blank).type_as(batch.indices.data)
            fill = torch.Tensor(fill).type_as(batch.indices.data)
            score.index_add_(1, fill, score.index_select(1, blank))
            score.index_fill_(1, blank, 1e-10)

        if blank_copy:
            blank_copy = torch.Tensor(blank_copy).type_as(batch.indices.data)
            fill_copy = torch.Tensor(fill_copy).type_as(batch.indices.data)
            score.index_add_(1, fill_copy, score.index_select(1, blank_copy))
            score.index_fill_(1, blank_copy, 1e-10)

    return scores


class CopyGenerator(nn.Module):
    """An implementation of pointer-generator networks
    :cite:`DBLP:journals/corr/SeeLM17`.

    These networks consider copying words
    directly from the source sequence.

    The copy generator is an extended version of the standard
    generator that computes three values.

    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.

    The model returns a distribution over the extend dictionary,
    computed as

    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`


    .. mermaid::

       graph BT
          A[input]
          S[src_map]
          B[softmax]
          BB[switch]
          C[attn]
          D[copy]
          O[output]
          A --> B
          A --> BB
          S --> D
          C --> D
          D --> O
          B --> O
          BB --> O


    Args:
       input_size (int): size of input representation
       output_size (int): size of output vocabulary
       pad_idx (int)
    """

    def __init__(self, input_size, output_size, pad_idx):
        super(CopyGenerator, self).__init__()
        self.linear_bm25 = nn.Linear(1, 1)
        self.linear = nn.Linear(input_size, output_size)
        self.linear_copy = nn.Linear(input_size, 1)
        self.linear_conv_copy = nn.Linear(input_size, 1)
        self.linear_generator = nn.Linear(input_size, 1)
        self.pad_idx = pad_idx

    def forward(self, hidden, attn, conv_attn, src_map, conv_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying
        source words.

        Args:
           hidden (FloatTensor): hidden outputs ``(batch x tlen, input_size)``
           attn (FloatTensor): attn for each ``(batch x tlen, input_size)``
           src_map (FloatTensor):
               A sparse indicator matrix mapping each source word to
               its index in the "extended" vocab containing.
               ``(src_len, batch, extra_words)``
        """

        # CHECKS
        batch_by_tlen, _ = hidden.size()
        batch_by_tlen_, slen = attn.size()
        batch_by_tlen__, conv_slen = conv_attn.size()
        slen_, batch, cvocab = src_map.size()
        conv_slen_, batch_, conv_cvocab = conv_map.size()
        aeq(batch_by_tlen, batch_by_tlen_, batch_by_tlen__)
        aeq(slen, slen_)
        aeq(conv_slen, conv_slen_)

        # Original probabilities.
        logits = self.linear(hidden)
        logits[:, self.pad_idx] = -float('inf')
        prob = torch.softmax(logits, 1)

        # Probability of copying p(z=1) batch.
        p_tweets_copy = self.linear_copy(hidden)

        # bm25_normalized = self.linear_bm25(bm25.view(-1, 1)).view(bm25.size()[0])
        # bias = torch.sigmoid(bm25_normalized)
        # p_conv_copy = bias * self.linear_conv_copy(hidden)
        p_conv_copy = self.linear_conv_copy(hidden)
        p_gen = self.linear_generator(hidden)

        temp = torch.softmax(torch.cat((p_tweets_copy, p_conv_copy, p_gen), dim=1), dim=1)
        p_tweets_copy = temp[:, 0].unsqueeze(dim=1)
        p_conv_copy = temp[:, 1].unsqueeze(dim=1)
        temp_out_prob = temp[:, 2].unsqueeze(dim=1)

        mul_attn = torch.mul(attn, p_tweets_copy)
        mul_conv_attn = torch.mul(conv_attn, p_conv_copy)

        copy_prob = torch.bmm(
            mul_attn.view(-1, batch, slen).transpose(0, 1),
            src_map.transpose(0, 1)
        ).transpose(0, 1)
        copy_prob = copy_prob.contiguous().view(-1, cvocab)
        conv_copy_prob = torch.bmm(
            mul_conv_attn.view(-1, batch_, conv_slen).transpose(0, 1),
            conv_map.transpose(0, 1)
        ).transpose(0, 1)
        conv_copy_prob = conv_copy_prob.contiguous().view(-1, conv_cvocab)

        # Probability of not copying: p_{word}(w) * (1 - p(z))

        # out_prob = torch.mul(prob, 1 - p_tweets_copy - p_conv_copy)
        out_prob = torch.mul(prob, temp_out_prob)

        # return torch.cat([out_prob, copy_prob, conv_copy_prob], 1)
        return out_prob, copy_prob, conv_copy_prob


class CopyGeneratorLoss(nn.Module):
    """Copy generator criterion."""

    def __init__(self, vocab_size, force_copy, unk_index=0,
                 ignore_index=-100, eps=1e-20):
        super(CopyGeneratorLoss, self).__init__()
        self.force_copy = force_copy
        self.eps = eps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.unk_index = unk_index

    def forward(self, scores, align_src, align_conv, target):
        """
        Args:
            scores (FloatTensor): ``(batch_size*tgt_len)`` x dynamic vocab size
                whose sum along dim 1 is less than or equal to 1, i.e. cols
                softmaxed.
            align (LongTensor): ``(batch_size x tgt_len)``
            target (LongTensor): ``(batch_size x tgt_len)``
        """
        # probabilities assigned by the model to the gold targets
        vocab_probs = scores[0].gather(1, target.unsqueeze(1)).squeeze(1)

        # probability of tokens copied from source
        # copy_ix = align_src.unsqueeze(1) + self.vocab_size
        copy_ix = align_src.unsqueeze(1)
        copy_tok_probs = scores[1].gather(1, copy_ix).squeeze(1)
        # probability of tokens copied from conversation
        copy_conv_ix = align_conv.unsqueeze(1)
        copy_conv_tok_probs = scores[2].gather(1, copy_conv_ix).squeeze(1)

        # Set scores for unk to 0 and add eps
        copy_tok_probs[align_src == self.unk_index] = 0
        copy_tok_probs += self.eps  # to avoid -inf logs
        copy_conv_tok_probs[align_conv == self.unk_index] = 0
        copy_conv_tok_probs += self.eps  # to avoid -inf logs

        # find the indices in which you do not use the copy mechanism
        non_copy = torch.ones(align_src.size()).bool()
        for i in range(align_src.size()[0]):
            if align_src[i] != self.unk_index or align_conv[i] != self.unk_index:
                non_copy[i] = False
        # non_copy = align_src == self.unk_index and align_conv == self.unk_index
        # print(type(non_copy), type(target), type(self.unk_index), non_copy.shape, target.shape, non_copy.get_device(), target.get_device())
        if not self.force_copy:
            non_copy = non_copy | (target.cpu() != self.unk_index)

        probs = torch.where(
            non_copy.cuda(), copy_tok_probs + copy_conv_tok_probs + vocab_probs, copy_tok_probs + copy_conv_tok_probs
        )

        loss = -probs.log()  # just NLLLoss; can the module be incorporated?
        # Drop padding.
        loss[target == self.ignore_index] = 0
        return loss


class CopyGeneratorLossCompute(NMTLossCompute):
    """Copy Generator Loss Computation."""

    def __init__(self, criterion, generator, tgt_vocab, normalize_by_length,
                 lambda_coverage=0.0):
        super(CopyGeneratorLossCompute, self).__init__(
            criterion, generator, lambda_coverage=lambda_coverage)
        self.tgt_vocab = tgt_vocab
        self.normalize_by_length = normalize_by_length

    def _make_shard_state(self, batch, output, range_, attns):
        """See base class for args description."""
        if getattr(batch, "alignment", None) is None:
            raise AssertionError("using -copy_attn you need to pass in "
                                 "-dynamic_dict during preprocess stage.")

        shard_state = super(CopyGeneratorLossCompute, self)._make_shard_state(
            batch, output, range_, attns)

        shard_state.update({
            "copy_attn": attns.get("copy"),
            "conv_copy_attn": attns.get("conv_copy"),
            "align_src": batch.alignment[range_[0] + 1: range_[1]],
            "align_conv": batch.conv_alignment[range_[0] + 1: range_[1]],
        })
        return shard_state

    def _compute_loss(self, batch, output, target, copy_attn, conv_copy_attn, align_src, align_conv,
                      std_attn=None, coverage_attn=None):
        """Compute the loss.

        The args must match :func:`self._make_shard_state()`.

        Args:
            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            copy_attn: the copy attention value.
            align: the align info.
        """
        target = target.view(-1)
        align_src = align_src.view(-1)
        align_conv = align_conv.view(-1)

        scores = self.generator(
            self._bottle(output), self._bottle(copy_attn), self._bottle(conv_copy_attn),
            batch.src_map, batch.conv_map
        )

        # scores = self.generator(
        #     self._bottle(output), self._bottle(copy_attn), self._bottle(conv_copy_attn), batch.bm25.repeat(output.shape[0]), batch.src_map, batch.conv_map
        # )
        loss = self.criterion(scores, align_src, align_conv, target)

        if self.lambda_coverage != 0.0:
            coverage_loss = self._compute_coverage_loss(std_attn,
                                                        coverage_attn)
            loss += coverage_loss

        # this block does not depend on the loss value computed above
        # and is used only for stats
        scores_data = collapse_copy_scores(
            self._unbottle(torch.cat([scores[0], scores[1], scores[2]], 1).clone(), batch.batch_size),
            batch, self.tgt_vocab, None)

        # scores_data = collapse_copy_scores(
        #     self._unbottle(scores[0].clone(), batch.batch_size), self._unbottle(scores[1].clone(), batch.batch_size),
        #     self._unbottle(scores[2].clone(), batch.batch_size),
        #     batch, self.tgt_vocab, None)

        scores_data = self._bottle(scores_data)

        # this block does not depend on the loss value computed above
        # and is used only for stats
        # Correct target copy token instead of <unk>
        # tgt[i] = align[i] + len(tgt_vocab)
        # for i such that tgt[i] == 0 and align[i] != 0

        target_data = target.clone()
        unk = self.criterion.unk_index
        correct_mask1 = (target_data == unk) & (align_src != unk)
        offset_align1 = align_src[correct_mask1] + len(self.tgt_vocab)
        correct_mask2 = (target_data == unk) & (align_conv != unk)
        offset_align2 = align_conv[correct_mask2] + len(self.tgt_vocab) + batch.src_map.size()[-1]
        target_data[correct_mask1] += offset_align1
        target_data[correct_mask2] += offset_align2

        # Compute sum of perplexities for stats
        stats = self._stats(loss.sum().clone(), scores_data, target_data)

        # this part looks like it belongs in CopyGeneratorLoss
        if self.normalize_by_length:
            # Compute Loss as NLL divided by seq length
            tgt_lens = batch.tgt[:, :, 0].ne(self.padding_idx).sum(0).float()
            # Compute Total Loss per sequence in batch
            loss = loss.view(-1, batch.batch_size).sum(0)
            # Divide by length of each sequence and sum
            loss = torch.div(loss, tgt_lens).sum()
        else:
            loss = loss.sum()

        return loss, stats
