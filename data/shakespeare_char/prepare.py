import os
import pickle
import requests
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# 下载莎士比亚数据集
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# 训练 BPE 分词器
tokenizer_file = os.path.join(os.path.dirname(__file__), 'bpe_tokenizer.json')
if not os.path.exists(tokenizer_file):
    print("Training BPE tokenizer...")
    # 定义 BPE 分词器
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    
    # 定义训练器，设置词汇表大小和特殊token
    trainer = BpeTrainer(vocab_size=50304, special_tokens=["<unk>", "<s>", "</s>"])
    
    # 训练 BPE tokenizer
    tokenizer.train([input_file_path], trainer)
    
    # 保存训练好的 tokenizer
    tokenizer.save(tokenizer_file)
else:
    print("Loading existing BPE tokenizer...")
    tokenizer = Tokenizer.from_file(tokenizer_file)

# 使用 BPE tokenizer 对数据进行编码
def encode(s):
    return tokenizer.encode(s).ids  # 编码：将字符串转换为 token ids
def decode(l):
    return tokenizer.decode(l)  # 解码：将 token ids 转换为字符串

# 创建训练集和验证集的切分
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# 将训练集和验证集进行编码
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# 导出为二进制文件
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# 保存 tokenizer 的元信息以供后续使用
meta = {
    'vocab_size': tokenizer.get_vocab_size(),
    'tokenizer': tokenizer_file,  # 保存 tokenizer 文件路径
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# 输出数据集信息
print(f"length of dataset in characters: {len(data):,}")
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")
print(f"vocab size: {tokenizer.get_vocab_size():,}")

#打印vocab
print(tokenizer.get_vocab())
