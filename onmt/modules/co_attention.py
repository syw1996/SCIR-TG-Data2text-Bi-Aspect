""" Co-attention modules """
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory

from onmt.models.stacked_rnn import StackedLSTM, StackedGRU

class CoAttention(nn.Module):
    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0):
        super(CoAttention, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.linear_ref = nn.Linear(hidden_size, hidden_size)
        input_size = 3 * hidden_size
        self.fusion_encoder = FusionEncoder(rnn_type, bidirectional, num_layers,
                                            hidden_size, input_size, dropout)
    
    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert not rnn_type == "SRU", "SRU doesn't support input feed! " \
            "Please set -input_feed 0!"
        if rnn_type == "LSTM":
            stacked_cell = StackedLSTM
        else:
            stacked_cell = StackedGRU
        return stacked_cell(num_layers, input_size,
                            hidden_size, dropout)
    
    def forward(self, src_enc_state, src_memory_bank, ref_memory_bank, src_length=None, ref_length=None):
        """
        Args:
            src_memory_bank (`FloatTensor`): vectors from the encoder
                 `[src_len x batch x hidden]`.
            ref_memory_bank (`FloatTensor`): vectors from the encoder
                 `[ref_len x batch x hidden]`.
        Returns:
        """
        # [batch x src_len x hidden]
        h_s = src_memory_bank.transpose(0, 1)
        # [batch x ref_len x hidden]
        h_r = ref_memory_bank.transpose(0, 1)
        
        ref_memory_bank = self.alternating_forward(h_s, h_r, src_length, ref_length)
        
        return ref_memory_bank
    
    def alternating_forward(self, src_memory_bank, ref_memory_bank, src_length=None, ref_length=None):
        """
        Args:
            src_memory_bank (`FloatTensor`): vectors from the encoder
                 `[batch x src_len x hidden]`.
            ref_memory_bank (`FloatTensor`): vectors from the encoder
                 `[batch x ref_len x hidden]`.
        """
        h_r = ref_memory_bank
        # MLP
        h_s = torch.tanh(self.linear_ref(src_memory_bank.contiguous().view(-1, self.hidden_size))).view(src_memory_bank.size())
        
        # [batch x hidden x ref_len]
        r_t = torch.transpose(h_r, 1, 2)
        # [batch x src_len x ref_len]
        l = torch.bmm(h_s, r_t)
        
        # [batch x src_len x ref_len]
        a_s_ = F.softmax(l, dim=1) 
        # [batch x ref_len x src_len]
        a_s = torch.transpose(a_s_, 1, 2)
        # [batch x hidden x src_len]
        c_s = torch.bmm(r_t, a_s) 

        # [batch x hidden x src_len]
        s_t = torch.transpose(h_s, 1, 2)
        # [batch x src_len x ref_len]
        a_r = F.softmax(l, dim=2)
        # [batch x 2 * hidden x src_len] * [batch x src_len x ref_len] => [batch x 2 * hidden x ref_len]
        c_r = torch.bmm(torch.cat((s_t, c_s), 1), a_r)
        
        # [batch x ref_len x 2 * hidden]
        c_r_t = torch.transpose(c_r, 1, 2)

        # fusion BiLSTM
        # [ref_len x batch x 3 * hidden]
        bilstm_in = torch.cat((c_r_t, h_r), 2).transpose(0, 1)
        encoder_final, memory_bank = self.fusion_encoder(bilstm_in, ref_length)

        return memory_bank

class FusionEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (:obj:`str`):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :obj:`nn.Dropout`
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, input_size, dropout=0.0):
        super(FusionEncoder, self).__init__()

        self.sort_data = True

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)
    
    def sort_input(self, input_data, input_length):
        assert len(input_length.size()) == 1
        sorted_input_length, sorted_idx = torch.sort(input_length, dim=0, descending=True)

        _, reverse_idx = torch.sort(sorted_idx, dim=0, descending=False)

        real_input_data = input_data.index_select(1, sorted_idx)
        return real_input_data, sorted_input_length, reverse_idx

    def reverse_sort_result(self, output, reverse_idx):
        return output.index_select(1, reverse_idx)
    
    def forward(self, memory_bank, lengths=None):
        "See :obj:`EncoderBase.forward()`"
        self._check_args(memory_bank, lengths)

        packed_memory_bank = memory_bank
        
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            if self.sort_data:
                memory_bank, lengths, reverse_idx = self.sort_input(memory_bank, lengths)
            lengths_list = lengths.view(-1).tolist()
            packed_memory_bank = pack(memory_bank, lengths_list)

        fusion_memory_bank, encoder_final = self.rnn(packed_memory_bank)

        if lengths is not None and not self.no_pack_padded_seq:
            fusion_memory_bank = unpack(fusion_memory_bank)[0]
            if self.sort_data:
                fusion_memory_bank = self.reverse_sort_result(fusion_memory_bank, reverse_idx)
                encoder_final = tuple([self.reverse_sort_result(enc_hidden, reverse_idx) 
                  for enc_hidden in encoder_final])

        return encoder_final, fusion_memory_bank