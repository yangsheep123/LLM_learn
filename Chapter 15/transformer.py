import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    def __init__(self,features:int,eps: float = 10 ** -6):
        super().__init__()
        self.eps = eps
        # 可学习权重
        self.alpha = nn.Parameter(torch.ones(features))
        # 可学习偏差
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self,x):
        mean = torch.mean(x,dim=-1,keepdim=True) #[batch_size,seq_len,1]
        std = torch.std(x,dim=-1)
        out = ((x-mean)/std)*self.alpha+self.bias
        return out

class FeedForwardBlock(nn.Module):
    def __init__(self,d_model,d_ff,Dropout):
        super().__init__()
        self.Dropout = Dropout
        self.l1 = nn.Linear(d_model,d_ff)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(d_ff,d_model)

    def forward(self,x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return (self.l2(self.Dropout(self.relu(self.l1(x)))))
    
class InputsEmbedding(nn.Module):
    def __init__(self,vocal_size,d_model):
        super.__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocal_size,d_model)
    def forward(self,x):
        # x:[batch_size,seq_len]
        # 要返回embedding里面的x
        # * math.sqrt(self.d_model)是原论文的要求
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalCode(nn.Module):
    def __init__(self,seq_len,d_model,dropout:float):
        super.__init__()
        self.Dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
        self.pe = torch.zeros((self.seq_len,d_model))
        pos_mt = torch.arange(self.seq_len).reshape((-1,1))
        i_mt = torch.arange(d_model,step=2).reshape((1,-1))
        i_mt = torch.pow(10000,2*i_mt/d_model)
        self.pe[:,0::2] = torch.sin(pos_mt / i_mt)
        self.pe[:,1::2] = torch.cos(pos_mt / i_mt)
        # 增加batch维度
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        # 注册位置编码为一个buffer，这样就不会更新位置矩阵里面的参数
        self.register_buffer('pe', self.pe)

    def forward(self,x):
        # x是word_embedding，形状是[batch_size,seq_len,d_model]
        x = x + self.pe[:,:x.shape[1],:]
        return self.Dropout(x)
    
class ResidualConnection(nn.Module):
    def __init__(self,d_model,dropout:float):
        super.__init__()
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self,d_model,h,dropout):
        super.__init__()
        self.Dropout = nn.Dropout(dropout)
        self.d_model = d_model #特征个数
        self.h = h #头的个数
        assert d_model%h==0
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model,d_model,bias=False)
        self.w_k = nn.Linear(d_model,d_model,bias=False)
        self.w_v = nn.Linear(d_model,d_model,bias=False)
        self.w_o = nn.Linear(d_model,d_model,bias=False)

    def attention(self,query,key,value,mask,dropout:nn.Dropout):
        # query:[batch_size,seq_len,d_k] (d_k是拆分后的大小)
        # key:[batch_size,seq_len,d_k]
        # value:[batch_size,seq_len,d_k]
        # attention_score:[batch_size,seq_len,seq_len]
        d_k = query.shape[-1]
        attention_score = query@key.transpose(-1,-2) / math.sqrt(d_k)
        attention_score.masked_fill_(mask,-1e9)
        attention_score = torch.softmax(attention_score,dim=-1)
        if dropout is not None:
            attention_score = dropout(attention_score)
        # attention_score@value:[batch_size,seq_len,d_k]
        return (attention_score@value),attention_score

    def forward(self,q,k,v,mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        # query:[batch_size,seq_len,d_model] --> [batch_size,seq_len,h,d_k] -->[batch_size,h,seq_len,d_k]
        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h,self.d_k)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k)
        # x:[batch_size,h,seq_len,d_k]
        x,self.attention_score = self.attention(query,key,value,mask,self.Dropout)
        # 多个头合并
        # x = x.view(x.shape[0],x.shape[2],self.h*self.d_k) 这样写不行，会让数据混
        x = x.transpose(1,2).contiguous().view(x.shape[0],x.shape[1],self.h*self.d_k)

        return self.w_o(x)

class EncoderBlock(nn.Module):
    def __init__(self,features,self_attention_block: MultiHeadAttentionBlock,\
                  feed_forward_block: FeedForwardBlock, dropout: float):
        super.__init__()
        self.norm = LayerNormalization(features)
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.ResidualConnection = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(2)])


    def forward(self,x,mask):
        x = self.ResidualConnection[0](x,lambda x:self.self_attention_block(x,x,x,mask))
        x = self.ResidualConnection[1](x,self.feed_forward_block)
        return self.norm(x)

class Encoder(nn.Module):
    def __init__(self,features,layers:nn.ModuleList):
        super.__init__()
        self.norm = LayerNormalization(features)
        # EncoderBlock的ModuleList
        self.layers = layers
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self,features,self_attention_block:MultiHeadAttentionBlock,\
                 feed_forward_block: FeedForwardBlock, \
                    cross_attention_block:MultiHeadAttentionBlock,dropout: float):
        super.__init__()
        self.norm = LayerNormalization(features)
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.cross_attention_block = cross_attention_block
        self.ResidualConnection = nn.ModuleList([ResidualConnection(features,dropout) for _ in range(3)])

    def forward(self,src_mask,trg_mask,x,encoder_output):
        x = self.ResidualConnection[0](x,lambda x:self.self_attention_block(x,x,x,trg_mask))
        x = self.ResidualConnection[1](x,lambda x:self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.ResidualConnection[2](x,self.feed_forward_block)
        return self.norm(x)
        
class Decoder(nn.Module):
    def __init__(self,layers:nn.ModuleList,features):
        super.__init__()
        # DecoderBlock的ModuleList
        self.layers = layers
        self.norm = LayerNormalization(features)
    def forward(self,x,src_mask,trg_mask,encoder_output):
        for layer in self.layers:
            x = layer(src_mask,trg_mask,x,encoder_output)
        return self.norm(x)


class ProjectLayer(nn.Module):
    def __init__(self,voca_size,d_model):
        super.__init__()
        self.l = nn.Linear(d_model,voca_size)
    def forward(self,x):
        x = self.l(x)
        return x

class Transformer(nn.Module):
    def __init__(self,encoder:Encoder,decoder:Decoder,\
                 src_emb:InputsEmbedding,trg_emb:InputsEmbedding,\
                    src_pos:PositionalCode,trg_pos:PositionalCode,projection_layer: ProjectLayer):
        super.__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.trg_emb = trg_emb
        self.src_pos = src_pos
        self.trg_pos = trg_pos
        self.projection_layer = projection_layer
    def encode(self,src,src_mask):
        src = self.src_emb(src)
        src = self.src_pos(src)
        return self.encode(src,src_mask)
    def decode(self,trg,trg_mask,src_mask,encoder_output):
        trg = self.trg_emb(trg)
        trg = self.trg_pos(trg)
        return self.decoder(src_mask,trg,trg_mask,encoder_output)
    def project(self,x):
        return self.projection_layer(x)
    
class build_transformer(nn.Module):
    def __init__(self,src_voca_size,trg_voca_size,src_seq_len,trg_seq_len,\
                 d_model=512,dropout=0.1,N=6,h=8,d_ff=2048):
        # 创建Embedding层
        src_emb = InputsEmbedding(src_voca_size,d_model)
        trg_emb = InputsEmbedding(trg_voca_size,d_model)

        # 创建位置编码层
        src_pos = PositionalCode(src_seq_len,d_model,dropout)
        trg_pos = PositionalCode(trg_seq_len,d_model,dropout)

        # 创建编码模块,N=6表示编码器有6个block
        encoder_block = []
        for _ in range(N):
            attention = MultiHeadAttentionBlock(d_model,h,dropout)
            feed_forward = FeedForwardBlock(d_model,d_ff,dropout)
            x = EncoderBlock(d_model,attention,feed_forward,dropout)
            encoder_block.append(x)
        
        # 创建解码模块
        decoder_block = []
        for _ in range(N):
            attention = MultiHeadAttentionBlock(d_model,h,dropout)
            feed_forward = FeedForwardBlock(d_model,d_ff,dropout)
            cross_attention = MultiHeadAttentionBlock(d_model,h,dropout)
            x = DecoderBlock(d_model,attention,feed_forward,cross_attention,dropout)
            decoder_block.append(x)
        
        # 创建解码器和编码器
        encoder = Encoder(d_model,nn.ModuleList(encoder_block))
        decoder = Decoder(nn.ModuleList(decoder_block),d_model)

        prj = ProjectLayer(trg_voca_size,d_model)

        transformer = Transformer(encoder,decoder,src_emb,trg_emb,src_pos,trg_pos,prj)

        # 初始化参数
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return transformer


