import torch
import torch.nn as nn
import numpy
import torch.nn.functional as F

batch_size = 3

# 词典大小
max_num_src_words = 8
max_num_trg_words = 20
max_position_len = max_num_src_words

# embedding的特征维度
model_dim = 10

src_lens = torch.Tensor([2,4]).to(torch.int32)
trg_lens = torch.Tensor([5,3]).to(torch.int32)

# 生成源序列和目标序列
# step1 单词索引构成源句子和目标句子，构建batch，并做padding
src_seq = [torch.randint(1,max_num_src_words,(L,)) for L in src_lens]
trg_seq = [torch.randint(1,max_num_trg_words,(L,)) for L in trg_lens]

# 生成padding
src_seq = [F.pad(src,(0,max(src_lens)-L)) for src,L in zip(src_seq,src_lens)]
trg_seq = [F.pad(trg,(0,max(trg_lens)-L)) for trg,L in zip(trg_seq,trg_lens)]

# 再将列表中的多个tensor拼成一个tensor
src_seq = torch.cat([src.unsqueeze(0) for src in src_seq],dim=0)
trg_seq = torch.cat([trg.unsqueeze(0) for trg in trg_seq],dim=0)

# step2 构造word embedding
# +1表示还有一个pad符号
src_embedding_table = nn.Embedding(max_num_src_words+1,model_dim)
trg_embedding_table = nn.Embedding(max_num_trg_words+1,model_dim)
src_embedding = src_embedding_table(src_seq)
trg_embedding = trg_embedding_table(trg_seq)

# step3 构造position embedding(这里tgr和src的位置编码矩阵不一样，我只写了src的)
pos_mt = torch.arange(start=0,end=max_position_len).reshape((-1,1))
i_mt = torch.pow(10000,torch.arange(start=0,end=model_dim,step=2).reshape((1,-1)))
pe_embedding_table = torch.zeros(max_position_len,model_dim)
pe_embedding_table[:,0::2] = torch.sin(pos_mt / i_mt)
pe_embedding_table[:,1::2] = torch.cos(pos_mt / i_mt)

pe_embedding = nn.Embedding(max_position_len,model_dim)
pe_embedding.weight = nn.Parameter(pe_embedding_table,requires_grad=False)
# pad可以区分开来，注意位置编码只是区分了位置，和是不是有实际语义无关
src_pos = [torch.arange(max(src_lens)).unsqueeze(0) for _ in src_lens]
src_pos = torch.cat(src_pos,dim=0)
src_pos_embedding = pe_embedding(src_pos)

# 注意：每个mask的形状和Q*K^T的形状都是一样的

# step4 构造encoder的self-attention mask
# 这个mask是来遮挡Q*K^T的，先根据输入构造mask矩阵，
# 然后根据输入再计算Q*K^T，这时valid_encoder_self_attention就可以把pad的地方换成很小的数
valid_encoder_pos = [torch.ones((L,)) for L in src_lens]
# [batch_size,src_len]
valid_encoder_pos = [pos.unsqueeze(0) for pos in valid_encoder_pos]
valid_encoder_pos = [F.pad(pos,(0,max(src_lens)-L)) for pos,L in \
                         zip(valid_encoder_pos,src_lens)]
# [batch_size,src_len,1]
valid_encoder_pos = torch.cat(valid_encoder_pos,dim=0).unsqueeze(2)
# mask的shape：[batch_size,src_len,src_len]
valid_encoder_self_attention = torch.bmm(valid_encoder_pos,valid_encoder_pos.transpose(-1,-2))
mask_encoder_self_attention = (1-valid_encoder_self_attention).to(torch.bool)

# step5 构造decoder和encoder之间的mask
valid_decoder_pos = [F.pad(torch.ones((L,)),(0,max(trg_lens)-L)).unsqueeze(0) for L in trg_lens]
# [batch_size,trg_len,1]
valid_decoder_pos = torch.cat(valid_decoder_pos,dim=0).unsqueeze(2)
# [batch_size,trg_len,src_len]
valid_cross_pos = torch.bmm(valid_decoder_pos,valid_encoder_pos.transpose(1,2))
invalid_cross_pos = 1-valid_cross_pos
mask_cross_self_attention = invalid_cross_pos.to(torch.bool)


# step6 构造decoder的self-attention mask
# [trg_len,trg_len]的一个下三角形
valid_decoder_tri_pos = [F.pad(torch.tril(torch.ones(L,L)),\
                               (0,max(trg_lens)-L,0,max(trg_lens)-L)) for L in trg_lens]
# [batch_size,trg_len,trg_len]
valid_decoder_tri_pos = [pos.unsqueeze(0) for pos in valid_decoder_tri_pos]
valid_decoder_tri_pos = torch.cat(valid_decoder_tri_pos,dim=0)
# print(valid_decoder_tri_pos.shape)
mask_decoder_self_attention = (1-valid_decoder_tri_pos).to(torch.bool)
# print(mask_decoder_self_attention.shape)

