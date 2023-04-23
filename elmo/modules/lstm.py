import mindspore
import mindspore.nn as nn
import mindspore.ops as P
import numpy as np
from elmo.nn.rnn import DynamicRNN
from elmo.nn.rnn_cells import LSTMCellWithProjection
from elmo.nn.rnn_cell_wrapper import ResidualWrapper, DropoutWrapper
from mindspore.ops.primitive import constexpr
from mindspore import Tensor, ms_function

@constexpr
def _init_state(hidden_num, batch_size, hidden_size, proj_size, dtype):
    hx = Tensor(np.zeros((hidden_num, batch_size, proj_size)), dtype)
    cx = Tensor(np.zeros((hidden_num, batch_size, hidden_size)), dtype)
    return (hx, cx)
    
class ELMoLSTM(nn.Cell):
    def __init__(
                self, 
                input_size, 
                hidden_size, 
                proj_size,
                num_layers, 
                keep_prob:float=0.0,
                cell_clip:float=0.0,
                proj_clip:float=0.0,
                skip_connections:bool=False,
                is_training:bool=True,
                batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.proj_size = proj_size
        self.num_directions = 2
        self.batch_first = batch_first
        self.support_non_tensor_inputs = True

        layers = nn.CellList()
        lstm_input_size = input_size
        for i in range(num_layers):
            forward_cell = LSTMCellWithProjection(lstm_input_size, hidden_size, cell_clip=cell_clip, proj_size=proj_size, proj_clip=proj_clip).to_float(mindspore.float16)            
            backward_cell = LSTMCellWithProjection(lstm_input_size, hidden_size, cell_clip=cell_clip, proj_size=proj_size, proj_clip=proj_clip).to_float(mindspore.float16)
            
            if skip_connections:
                if i == 0:
                    pass
                else:
                    forward_cell = ResidualWrapper(forward_cell).to_float(mindspore.float16)  
                    backward_cell = ResidualWrapper(backward_cell).to_float(mindspore.float16)  
            
            if is_training:
                forward_cell = DropoutWrapper(forward_cell, input_keep_prob=keep_prob).to_float(mindspore.float16)  
                backward_cell = DropoutWrapper(backward_cell, input_keep_prob=keep_prob).to_float(mindspore.float16)  

            forward_layer = DynamicRNN(forward_cell).to_float(mindspore.float16)  
            backward_layer = DynamicRNN(backward_cell).to_float(mindspore.float16)  
            
            lstm_input_size = proj_size
            layers.append(forward_layer)
            layers.append(backward_layer)

        self.layers = layers
        self.dropout = nn.Dropout(keep_prob=keep_prob)
        self.cast = P.Cast()

    def construct(self, x, xr, h=None, seq_length=None):
        max_batch_size = x.shape[0] if self.batch_first else x.shape[1]
        if h is None:
            h = _init_state(self.num_layers * self.num_directions, max_batch_size, self.hidden_size, self.proj_size, x.dtype)
        if self.batch_first:
            x = P.Transpose()(x, (1, 0, 2))
            xr = P.Transpose()(xr, (1, 0, 2)) 
        x_f, x_b = self._stacked_bi_dynamic_rnn(x, xr, h, seq_length)
        if self.batch_first:
           x_f = P.Transpose()(x_f, (1, 0, 2))
           x_b = P.Transpose()(x_b, (1, 0, 2))
        return x_f, x_b

    def _stacked_bi_dynamic_rnn(self, x, xr, h, seq_length=None):
        """stacked bidirectional dynamic_rnn"""
        input_forward = x
        input_backward = xr
        outputs_f = ()
        outputs_b = ()

        for i in range(self.num_layers):
            offset = i * 2
            h_f_i = (P.Squeeze(0)(h[0][offset : offset+1]), P.Squeeze(0)(h[1][offset : offset+1]))
            h_b_i = (P.Squeeze(0)(h[0][offset + 1 : offset + 2]), P.Squeeze(0)(h[1][offset+1 : offset + 2]))
            forward_cell = self.layers[offset]
            backward_cell = self.layers[offset + 1]
            output_f, _ = forward_cell(input_forward, h_f_i, seq_length)
            output_b, _ = backward_cell(input_backward, h_b_i, seq_length)
            if seq_length is None:
                output_b = P.ReverseV2([0])(output_b)
            else:
                output_b = P.ReverseSequence(0, 1)(output_b, seq_length)

            outputs_f += (self.dropout(output_f),)
            outputs_b += (self.dropout(output_b),)
            
            input_forward = output_f
            input_backward = output_b
        return outputs_f[-1], outputs_b[-1]
