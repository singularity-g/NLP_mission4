import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torchtext.datasets import Multi30k
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import pandas as pd
# import spacy
import numpy as np
import random
import math
import jieba
import os
import pickle
from gensim.models import Word2Vec
import time
import matplotlib.pyplot as plt


class LSTM(torch.nn.Module):
    def __init__(self, hidden_size1, hidden_size2, vocab_size, input_size, num_layers):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, input_size, max_norm=1)
        self.max_len = 30
        self.lstm1 = torch.nn.LSTM(input_size, hidden_size1, num_layers, batch_first=True, bidirectional=False)
        self.lstm2 = torch.nn.LSTM(hidden_size1, hidden_size2, num_layers, batch_first=True, bidirectional=False)
        self.dropout = torch.nn.Dropout(0.1)
        self.line = torch.nn.Linear(hidden_size2 * self.max_len, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embed(x)
        output1, _ = self.lstm1(x)
        output, _ = self.lstm2(output1)
        out_d_reshaped = output.reshape(output.shape[0], (output.shape[1] * output.shape[2]))
        line_o = self.line(out_d_reshaped)
        pred = self.softmax(line_o)
        # print(pred.shape)
        return pred

class TextDataset(Dataset):
    def __init__(self, my_dict, vocab):
        self.para = []
        self.tmp = []
        self.max_len = 30
        self.source = []
        for sentence in my_dict:
            for i in range(0, len(sentence) - self.max_len):
                self.para.append(sentence[i: i + self.max_len])
                self.tmp.append(sentence[i + self.max_len])
        self.source = [
            ([vocab.get(char, vocab['<unk>']) for char in para], vocab[next_char])
            for para, next_char in zip(self.para, self.tmp)
            if next_char in vocab
        ]
    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        sample = self.source[idx]
        source = np.copy(sample[0])
        target = np.copy(sample[1])
        return source, target
def content_deal(content):
    ad = '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com'  #去除无意义的广告词
    content = content.replace(ad, '')
    content = content.replace("\u3000", '')
    content = content.replace("=", '')
    return content


def read_novel(path,stop_word_list):
    file_list = os.listdir(path)
    word_list = {}
    char_list = {}
    for file in file_list:
            novel_path = r"data/" + file
            char = []
            word = []

            with open(novel_path, 'r', encoding='gb18030') as f:
                content = f.read()
                word_list0 = content_deal(content)
                # 大于500词的段落
                for para in word_list0.split('\n'):
                    if len(para) < 1:
                        continue
                    char.append([char for char in para if char not in stop_word_list and char != ' '])
                    word.append([word for word in jieba.lcut(para) if word not in stop_word_list and word != ' '])
                    file_name = os.path.splitext(file)[0]
                    f.close()
            char_list[file_name] = char
            word_list[file_name] = word

    return char_list, word_list
def train(my_dict, vocab,id):
    train_dataset = TextDataset(my_dict, vocab)
    train_dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=True)

    # 设置gpu
    cuda_device = 1  # 这里假设第二块GPU的索引是1
    # cuda_device = 0
    torch.cuda.set_device(cuda_device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LSTM(128,128,len(vocab),64,1).to(device)
    print(model)
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # model.load_state_dict(torch.load(f"model_{id}.pth"))
    # for src in train_dataloader:
    #     print("训练集一个batch的句子索引:", src)
    #     break
    n_epochs = 200
    # criterion = nn.CrossEntropyLoss()
    loss_function = torch.nn.NLLLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    epoch_losses = []
    best_validation_loss = np.inf
    patience = 20
    for epoch in range(n_epochs):
        model.train()
        # 将梯度归零
        # 前向传播
        start_time = time.time()
        avg_loss = 0
        loss_epoch = 0
        for batch_idx, sample in enumerate(train_dataloader):
            source, target = sample
            source = source.to(device)  # .unsqueeze(-1)
            target = target.to(device)
            # target_one_hot_encoded = torch.nn.functional.one_hot(target, num_classes=len(vocab)).to(device)
            outputs = model(source)
            # outputs_flattened = outputs.permute(1, 0, 2).reshape(-1, len(vocab))
            # outputs_flattened = outputs.squeeze(0)
            # target_flattened = target.reshape(-1)
            # loss = criterion(outputs_flattened, target_flattened)
            loss = loss_function(outputs, target)
            loss.backward()
            loss_epoch += loss.detach().cpu()

            optimizer.step()  # 参数更新

        epoch_loss = loss_epoch / len(train_dataloader)
        epoch_losses.append(epoch_loss)

        # 打印当前 epoch 的信息
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.4f}, Time: {time.time() - start_time:.2f} seconds")

        if epoch_loss < best_validation_loss:
            best_validation_loss = epoch_loss
            # 保存模型参数
            torch.save(model.state_dict(), f"model_{id}.pth")
            patience_counter = 0
        else:
            # 如果验证损失没有降低，则增加等待计数器
            patience_counter += 1

            # 如果等待计数器超过指定的耐心值，则停止训练
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}.')
                break

    # plt.plot(range(1, n_epochs + 1), epoch_losses, label='Training Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training Loss')
    # plt.legend()
    # plt.show()
def test(my_dict, vocab,id):
    dataset = TextDataset(my_dict, vocab)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
    batch_iterator = iter(dataloader)
    first_batch = next(batch_iterator)
    gg = first_batch[0][:1]
    tar = first_batch[1][:1]
    words = []
    inv_vocab = {v: k for k, v in vocab.items()}
    for num in gg[0]:
        word = inv_vocab[num.item()]
        words.append(word)
    output_sentence = ''.join(words)
    print(output_sentence)
    # 设置gpu
    cuda_device = 0  # 这里假设第二块GPU的索引是1
    # cuda_device = 0
    torch.cuda.set_device(cuda_device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    model = LSTM(128,128,len(vocab),64,1).to(device)
    print(model)
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.load_state_dict(torch.load(f'model_{id}.pth'))
    model.eval()
    for _ in range(200):
        preds = model(gg.to(device))
        next_word = torch.argmax(preds, dim=-1)
        gg = torch.cat((gg, next_word.unsqueeze(0).cpu()),dim=-1)
        gg = gg[:,1:]
        print(inv_vocab[next_word.item()],end='')


if __name__ == '__main__':
    path = r"data/"
    stop_word_file = r"cn_stopwords.txt"
    punctuation_file = r"cn_punctuation.txt"
    ll = r"data/"

    """语料库预处理，第一次运行，之后可省略"""
    # # 读取停词列表
    stop_word_list = []
    with open(stop_word_file, 'r', encoding='utf-8') as f:
        for line in f:
            stop_word_list.append(line.strip())
    stop_word_list.extend("\u3000")
    stop_word_list.extend(['～', ' ', '没', '听', '一声', '道', '见', '中', '便', '说', '一个', '说道'])
    # 读取段落
    # 处理前删除文件夹内inf.txt
    # char_dict, word_dict = read_novel(ll,stop_word_list)
    # with open('word_dict.pkl', 'wb') as f:
    #     pickle.dump(word_dict, f)
    # with open('char_dict.pkl', 'wb') as f:
    #     pickle.dump(char_dict, f)
    """语料库预处理，第一次运行，之后可省略"""

    # """直接读取保存的数据"""
    with open('word_dict.pkl', 'rb') as f:
        word_dict = pickle.load(f)
    with open('char_dict.pkl', 'rb') as f:
        char_dict = pickle.load(f)
    data_source = ['倚天屠龙记']
    for id,wd in enumerate(word_dict):
        if wd in data_source:
            my_dict = word_dict[wd]

            """词汇表生成"""
            #第一次运行时取消以下注释#########
            # word2vec_model = Word2Vec(my_dict, hs=1, min_count=3, window=5, vector_size=200, sg=0, epochs=100)
            # word2vec_model.save(f"{wd}_word2vec_model.model")
            ######
            # word2vec_model = Word2Vec.load(f"{wd}_word2vec_model.model")
            # vocab = {word: idx for idx, word in enumerate(word2vec_model.wv.index_to_key)}
            # vocab['<pad'] = len(vocab)
            # vocab['<unk>'] = len(vocab)

            # with open(f'vocab_{wd}.pkl', 'wb') as f:
            #     pickle.dump(vocab, f)

            # 加载保存的 vocab 文件
            with open(f'vocab_{wd}.pkl', 'rb') as f:
                vocab = pickle.load(f)
            #
            model = 'test'
            if model == 'train':
                train(my_dict,vocab,id)
            elif model == 'test':
                test(my_dict, vocab,id)



