import torch
import torch.nn as nn

from onmt.Utils import aeq, sequence_mask


class GlobalAttention(nn.Module):
    """
    Global attention takes a matrix and a query vector. It
    then computes a parameterized convex combination of the matrix
    based on the input query.

    Constructs a unit mapping a query `q` of size `dim`
    and a source matrix `H` of size `n x dim`, to an output
    of size `dim`.


    .. mermaid::

       graph BT
          A[Query]
          subgraph RNN
            C[H 1]
            D[H 2]
            E[H N]
          end
          F[Attn]
          G[Output]
          A --> F
          C --> F
          D --> F
          E --> F
          C -.-> G
          D -.-> G
          E -.-> G
          F --> G

    All models compute the output as
    :math:`c = \sum_{j=1}^{SeqLength} a_j H_j` where
    :math:`a_j` is the softmax of a score function.
    Then then apply a projection layer to [q, c].

    However they
    differ on how they compute the attention score.

    * Luong Attention (dot, general):
       * dot: :math:`score(H_j,q) = H_j^T q`
       * general: :math:`score(H_j, q) = H_j^T W_a q`


    * Bahdanau Attention (mlp):
       * :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`


    Args:
       dim (int): dimensionality of query and key
       coverage (bool): use coverage term
       attn_type (str): type of attention to use, options [dot,general,mlp]

    """
    def __init__(self, dim, coverage=False, attn_type="dot", hier_mix_attn=False):
        super(GlobalAttention, self).__init__()

        self.dim = dim
        self.attn_type = attn_type
        assert (self.attn_type in ["dot", "general", "mlp"]), (
                "Please select a valid attention type.")

        self.hier_mix_attn = hier_mix_attn
        if hier_mix_attn:
            assert self.attn_type == "mlp"

        if self.attn_type == "general":
            self.linear_in = nn.Linear(dim, dim, bias=False)
            self.ref_linear_in = nn.Linear(dim, dim, bias=False)
        elif self.attn_type == "mlp":
            self.linear_context = nn.Linear(dim, dim, bias=False)
            self.ref_linear_context = nn.Linear(dim, dim, bias=False)
            if hier_mix_attn:
                self.linear_query = nn.Linear(dim*2, dim, bias=True)
                self.ref_linear_query = nn.Linear(dim*2, dim, bias=True)
            else:
                self.linear_query = nn.Linear(dim, dim, bias=True)
                self.ref_linear_query = nn.Linear(dim, dim, bias=True)
            self.v = nn.Linear(dim, 1, bias=False)
            self.ref_v = nn.Linear(dim, 1, bias=False)
        # mlp wants it with bias
        out_bias = self.attn_type == "mlp"
        self.linear_out = nn.Linear(dim*2, dim, bias=out_bias)
        self.ref_linear_out = nn.Linear(dim*3, dim, bias=out_bias)

        self.sm = nn.Softmax()
        self.tanh = nn.Tanh()

        if coverage:
            self.linear_cover = nn.Linear(1, dim, bias=False)
            self.ref_linear_cover = nn.Linear(1, dim, bias=False)

    def score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        if self.hier_mix_attn:
            aeq(src_dim*2, tgt_dim)
        else:
            aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch*tgt_len, tgt_dim)
                h_t_ = self.linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            target_input_dim = dim * 2 if self.hier_mix_attn else dim
            wq = self.linear_query(h_t.view(-1, target_input_dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = self.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)
        
    def ref_score(self, h_t, h_s):
        """
        Args:
          h_t (`FloatTensor`): sequence of queries `[batch x tgt_len x dim]`
          h_s (`FloatTensor`): sequence of sources `[batch x src_len x dim]`

        Returns:
          :obj:`FloatTensor`:
           raw attention scores (unnormalized) for each src index
          `[batch x tgt_len x src_len]`

        """

        # Check input sizes
        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        aeq(src_batch, tgt_batch)
        aeq(src_dim, tgt_dim)
        aeq(self.dim, src_dim)

        if self.attn_type in ["general", "dot"]:
            if self.attn_type == "general":
                h_t_ = h_t.view(tgt_batch*tgt_len, tgt_dim)
                h_t_ = self.ref_linear_in(h_t_)
                h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
            h_s_ = h_s.transpose(1, 2)
            # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
            return torch.bmm(h_t, h_s_)
        else:
            dim = self.dim
            target_input_dim = dim * 2 if self.hier_mix_attn else dim
            wq = self.ref_linear_query(h_t.view(-1, target_input_dim))
            wq = wq.view(tgt_batch, tgt_len, 1, dim)
            wq = wq.expand(tgt_batch, tgt_len, src_len, dim)

            uh = self.ref_linear_context(h_s.contiguous().view(-1, dim))
            uh = uh.view(src_batch, 1, src_len, dim)
            uh = uh.expand(src_batch, tgt_len, src_len, dim)

            # (batch, t_len, s_len, d)
            wquh = self.tanh(wq + uh)

            return self.v(wquh.view(-1, dim)).view(tgt_batch, tgt_len, src_len)

    def forward(self, input, memory_bank, ref_memory_bank=None, memory_lengths=None, ref_memory_lengths=None, coverage=None, align_score=None, cohan18=False, reweight_score=None):
        """

        Args:
          input (`FloatTensor`): query vectors `[batch x tgt_len x dim]`
          memory_bank (`FloatTensor`): source vectors `[batch x src_len x dim]`
          memory_lengths (`LongTensor`): the source context lengths `[batch]`
          coverage (`FloatTensor`): None (not supported yet)

        Returns:
          (`FloatTensor`, `FloatTensor`):

          * Computed vector `[tgt_len x batch x dim]`
          * Attention distribtutions for each query
             `[tgt_len x batch x src_len]`
        """

        # one step input
        if input.dim() == 2:
            one_step = True
            input = input.unsqueeze(1)
        else:
            one_step = False

        batch, sourceL, dim = memory_bank.size()
        batch_, targetL, dim_ = input.size()
        if ref_memory_bank is not None:
            ref_batch, ref_sourceL, ref_dim = ref_memory_bank.size()
            aeq(dim, ref_dim)
            aeq(batch, ref_batch)
        aeq(batch, batch_)
        if self.hier_mix_attn:
            aeq(dim*2, dim_)
        else:
            aeq(dim, dim_)
        aeq(self.dim, dim)
        if coverage is not None:
            batch_, sourceL_ = coverage.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)

        if coverage is not None:
            cover = coverage.view(-1).unsqueeze(1)
            memory_bank += self.linear_cover(cover).view_as(memory_bank)
            memory_bank = self.tanh(memory_bank)

        # compute attention scores, as in Luong et al.
        align = self.score(input, memory_bank)
        if ref_memory_bank is not None:
            ref_align = self.ref_score(input, ref_memory_bank)
        if cohan18 and reweight_score is not None:
            align = align * reweight_score

        if memory_lengths is not None:
            mask = sequence_mask(memory_lengths)
            mask = mask.unsqueeze(1)  # Make it broadcastable.
            align.data.masked_fill_(1 - mask, -float('inf'))
        
        if ref_memory_lengths is not None:
            ref_mask = sequence_mask(ref_memory_lengths)
            ref_mask = ref_mask.unsqueeze(1)  # Make it broadcastable.
            ref_align.data.masked_fill_(1 - ref_mask, -float('inf'))

        # Softmax to normalize attention weights
        align_vectors = self.sm(align.view(batch*targetL, sourceL))
        align_vectors = align_vectors.view(batch, targetL, sourceL)
        if ref_memory_bank is not None:
            ref_align_vectors = self.sm(ref_align.view(batch*targetL, ref_sourceL))
            ref_align_vectors = ref_align_vectors.view(batch, targetL, ref_sourceL)

        if align_score is not None:
            align_vectors = align_score.transpose(0, 1).contiguous()

        # each context vector c_t is the weighted average
        # over all the source hidden states
        c = torch.bmm(align_vectors, memory_bank)
        if ref_memory_bank is not None:
            ref_c = torch.bmm(ref_align_vectors, ref_memory_bank)

        # concatenate
        if ref_memory_bank is not None:
            concat_c = torch.cat([c, ref_c, input.narrow(2, 0, input.size(2) // 2) if self.hier_mix_attn else input], 2).view(batch*targetL, dim*3)
        else:    
            concat_c = torch.cat([c, input.narrow(2, 0, input.size(2) // 2) if self.hier_mix_attn else input], 2).view(batch*targetL, dim*2)
        if ref_memory_bank is not None:
            attn_h = self.ref_linear_out(concat_c).view(batch, targetL, dim)
        else:
            attn_h = self.linear_out(concat_c).view(batch, targetL, dim)
        if self.attn_type in ["general", "dot"] and not cohan18:
            attn_h = self.tanh(attn_h)

        if one_step:
            attn_h = attn_h.squeeze(1)
            align_vectors = align_vectors.squeeze(1)

            # Check output sizes
            batch_, dim_ = attn_h.size()
            aeq(batch, batch_)
            aeq(dim, dim_)
            batch_, sourceL_ = align_vectors.size()
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)
        else:
            attn_h = attn_h.transpose(0, 1).contiguous()
            align_vectors = align_vectors.transpose(0, 1).contiguous()

            # Check output sizes
            targetL_, batch_, dim_ = attn_h.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(dim, dim_)
            targetL_, batch_, sourceL_ = align_vectors.size()
            aeq(targetL, targetL_)
            aeq(batch, batch_)
            aeq(sourceL, sourceL_)

        if cohan18:
            return (attn_h, c.transpose(0, 1).contiguous()), (align_vectors, align.transpose(0, 1).contiguous())
        else:
            return attn_h, align_vectors
