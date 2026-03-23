import os
import sentencepiece as spm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# # 先运行以下命令训练 BPE 模型（只需运行一次）：
# spm.SentencePieceTrainer.Train('--input="D:\\study\\data\\en2cn\\train_en.txt" --model_prefix=en_bpe --vocab_size=16000 --model_type=bpe --character_coverage=1.0 --unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3')
# spm.SentencePieceTrainer.Train('--input="D:\\study\\data\\en2cn\\train_zh.txt" --model_prefix=zh_bpe --vocab_size=16000 --model_type=bpe --character_coverage=0.9995 --unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3')

sp_en = spm.SentencePieceProcessor()
sp_en.Load('en_bpe.model')
sp_zh = spm.SentencePieceProcessor()
sp_zh.Load('zh_bpe.model')

UNK_ID = sp_en.unk_id #0
PAD_ID = sp_en.pad_id #1
BOS_ID = sp_en.bos_id #2
EOS_ID = sp_en.eos_id #3

class TranslationDataset(Dataset):
    def __init__(self,src_file, trg_file, src_tokenizer, trg_tokenizer,max_len=100):
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.max_len = max_len

        with open(src_file,encoding='utf-8') as f:
            # 对读取到的完整字符串进行按行分割，返回一个列表，
            # 列表中每个元素是「一行文本」（自动剔除每行末尾的换行符）。
            src_lines = f.read().splitlines()
        with open(trg_file,encoding='utf-8') as f:
            # 对读取到的完整字符串进行按行分割，返回一个列表，
            # 列表中每个元素是「一行文本」（自动剔除每行末尾的换行符）。
            trg_lines = f.read().splitlines()
        assert len(src_lines)==len(trg_lines)
        self.pairs = []
        for src_ids,trg_ids in zip(src_lines,trg_lines):
            #给句子前后加上开始token和结束token
            src_ids = [BOS_ID] + src_tokenizer(src_ids) + [EOS_ID]
            trg_ids = [BOS_ID] + trg_tokenizer(trg_ids) + [EOS_ID]
            if len(src_ids)<=self.max_len and len(trg_ids)<=self.max_len:
                self.pairs.append((src_ids,trg_ids))
        
        
    def __len__(self):
        return len(self.src)
    def __getitem__(self, idx):
        src_ids, trg_ids = self.pairs[idx]
        return torch.LongTensor(src_ids),torch.LongTensor(trg_ids)
    
## 对一个batch的输入和输出token序列，依照最长的序列长度，
# 用<pad> token进行填充，确保一个batch的数据形状一致，组成一个tensor。
@staticmethod
def collate_fn(batch):
    src_lines,trg_lines = zip(*batch)
    src_lens = [len(x) for x in src_lines]
    trg_lens = [len(y) for y in trg_lines]
    src_pad = nn.utils.rnn.pad_sequence(src_lines,padding_value=PAD_ID)
    trg_pad = nn.utils.rnn.pad_sequence(trg_lines,padding_value=PAD_ID)
    return src_pad, trg_pad, src_lens, trg_lens