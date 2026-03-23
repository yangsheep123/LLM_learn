import torch
import torch.nn as nn

# 单向RNN网络
def forward_RNN(input,weight_ih,weight_hh,bias_ih,bias_hh,h_prev):
    bs,T,_ = input.shape
    hid_dim = weight_ih.shape[0]
    h_out = torch.zeros(bs,T,hid_dim)
    for t in range(T):
        x = input[:,t,:].unsqueeze(2) # x是三维向量 bs*input_dim*1
        weight_ih = weight_ih.unsqueeze(0).tile(bs,1,1) # bs*hid_dim*input_dim
        weight_hh = weight_hh.unsqueeze(0).tile(bs,1,1) # bs*hid_dim*hid_dim
        w_times_x = torch.bmm(weight_ih,x).squeeze(-1) # torch.bmm:和批次无关的矩阵乘法,bs*hid_dim
        w_times_h = torch.bmm(weight_hh,h_prev.unsqueeze(2)).squeeze(-1) # bs*hid_dim
        h_prev = torch.tanh(w_times_x + bias_ih + w_times_h + bias_hh) # bs*hid_dim
        h_out[:,t,:] = h_prev
    return h_out,h_prev.unsqueeze(0)
        
# 双向RNN网络
def bi_forward_RNN(input,weight_ih,weight_hh,bias_ih,bias_hh,h_prev,\
            weight_ih_reverse,weight_hh_reverse,bias_ih_reverse,bias_hh_reverse,h_prev_reverse):
    bs,T,_ = input.shape
    # h_prev:bs*T*h_dim
    h_dim = h_prev.shape[2]
    h_out = torch.zeros(bs,T,h_dim*2)
    forward_output,_ = forward_RNN(input,weight_ih,weight_hh,bias_ih,bias_hh,h_prev)
    input_reverse = torch.flip(input,[1])
    backward_output,_ = forward_RNN(input_reverse,weight_ih_reverse,weight_hh_reverse, \
                                    bias_ih_reverse,bias_hh_reverse,h_prev_reverse)
    h_out[:,:,:h_dim] = forward_output
    h_out[:,:,h_dim:] = torch.flip(backward_output,[1])
    return h_out,h_out[:,-1,:].reshape((bs,2,h_dim)).transpose(0,1)
