import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from transformer import build_transformer
import torch.nn.functional as F

sp_en = spm.SentencePieceProcessor()
sp_en.Load('en_bpe.model')
sp_cn = spm.SentencePieceProcessor()
sp_cn.Load('zh_bpe.model')

UNK_ID = sp_en.unk_id #0
PAD_ID = sp_en.pad_id #1
BOS_ID = sp_en.bos_id #2
EOS_ID = sp_en.eos_id #3

def tokenize_en(text):
    return sp_en.encode(text, out_type=int)


def tokenize_cn(text):
    return sp_cn.encode(text, out_type=int)

class TranslationDataset(Dataset):
    def __init__(self,src_file,trg_file,\
                 src_tokenizer,trg_tokenizer,max_len=100):
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        with open(src_file,encoding='utf-8') as f:
            src_lines = f.read().splitlines()
        with open(trg_file,encoding='utf-8') as f:
            trg_lines = f.read().splitlines()
        assert len(src_lines)==len(trg_lines)
        self.paris = []
        for src_ids,trg_ids in zip(src_lines,trg_lines):
            src_ids = [BOS_ID] + self.src_tokenizer(src_ids) +[EOS_ID]
            trg_ids = [BOS_ID] + self.src_tokenizer(trg_ids) +[EOS_ID]
            if len(src_ids)<=max_len and len(trg_ids)<=max_len:
                self.paris.append((src_ids,trg_ids))
    def __len__(self):
        return len(self.paris)
    def __getitem__(self,idx):
        src,trg = self.paris[idx]
        return torch.LongTensor(src),torch.LongTensor(trg)
    
    @staticmethod
    def collate_fn(batch):
        # batch = [ (src1, trg1), (src2, trg2), ..., (srcN, trgN) ]
        # 每个元组 = (源语言 token 列表，目标语言 token 列表)
        # 将src和trg分开
        # [batch_size,seq_len]
        src_batch,trg_batch = zip(*batch)
        src_lens = [len(x) for x in src_batch]
        trg_lens = [len(x) for x in trg_batch]
        src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=PAD_ID)
        trg_batch = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=PAD_ID)
        return src_batch,src_lens,trg_batch,trg_lens
    
# 定义掩码
def create_mask(src,trg,pad_idx):
    # 经过pad后的src:[batch_size,src_len]
    src_mask = (src==pad_idx).unsqueeze(1).unsqueeze(2) #[batch_size,1,1,src_len]
    trg_mask = (trg==pad_idx).unsqueeze(1).unsqueeze(2) #[batch_size,1,1,trg_len]
    # 因为是经过padding的，所有句子的长度都一样
    trg_len = trg.shape[1]
    # [trg_len,trg_len]的一个下三角形
    trg_tri_mask = torch.tril(torch.ones((trg_len,trg_len))).to(torch.bool)
    # & 是 按位与运算
    trg_sub_mask = trg_mask & trg_tri_mask #[batch_size,1,trg_len,trg_len]
    return src_mask,trg_sub_mask
def train(model, dataloader, optimizer, criterion, pad_idx):
    model.train()
    total_loss = 0
    step = 0
    log_loss = 0  # 用于每100步统计

    for src, tgt, src_lens, tgt_lens in dataloader:
        step += 1

        src = src
        tgt = tgt

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask, tgt_mask = create_mask(src, tgt_input, pad_idx)

        optimizer.zero_grad()
        encoder_output = model.encode(src, src_mask)
        decoder_output = model.decode(encoder_output, src_mask, tgt_input, tgt_mask)
        output = model.project(decoder_output)

        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        log_loss += loss.item()

        if step % 100 == 0:
            avg_log_loss = log_loss / 100
            print(f"Step {step}: Avg Loss = {avg_log_loss:.4f}")
            log_loss = 0  # 重置每100步的loss计数

    return total_loss / len(dataloader)

def main():
    # 超参数
    SRC_VOCAB_SIZE = 16000
    TGT_VOCAB_SIZE = 16000
    SRC_SEQ_LEN = 128
    TGT_SEQ_LEN = 128
    BATCH_SIZE = 2
    NUM_EPOCHS = 10
    LR = 1e-4

    # 数据集加载
    train_dataset = TranslationDataset('D:/study/data/en2cn/train_en.txt', 'D:/study/data/en2cn/train_zh.txt',tokenize_en, tokenize_cn)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate_fn)

    # 构建模型
    model = build_transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    for epoch in range(NUM_EPOCHS):
        loss = train(model, train_dataloader, optimizer, criterion, PAD_ID)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {loss:.4f}")

        torch.save(model.state_dict(), "transformer.pt")

if __name__ == "__main__":
    main()




