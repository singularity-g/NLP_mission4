import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset, DataLoader
import pickle
import time

def pos_sinusoid_embedding(seq_len,d_model) -> torch.Tensor:
    embeddings = torch.zeros((seq_len,d_model))
    for i in range(d_model):
        f = torch.sin if i%2==0 else torch.cos
        embeddings[:,i] = f(torch.arange(0,seq_len)/np.power(1e4,2*(i//2)/d_model))
    return embeddings.float()

#mask
def get_len_mask(b:int, max_len:int,feat_lens:torch.Tensor,device:torch.device):
    attn_mask = torch.ones((b,max_len,max_len),device=device)
    for i in range(b):
        attn_mask[i,:,:feat_lens[i]] = 0
    return attn_mask.to(torch.bool)

#casual mask
def get_subsequent_mask(b:int,max_len:int,device:torch.device)->torch.Tensor:
    return torch.triu(torch.ones((b,max_len,max_len),device=device), diagonal=1).to(torch.bool)

def get_enc_dec_mask(b:int,max_feat_len:int,feat_lens:torch.Tensor,max_label_len:int,device:torch.device)->torch.Tensor:
    attn_mask = torch.zeros((b, max_label_len, max_feat_len), device=device)
    for i in range(b):
        attn_mask[i,:,feat_lens[i]:] = 1
    return attn_mask.to(torch.bool)

class MultiHeadAttention(nn.Module):
    def __init__(self,d_k,d_v,d_model,num_heads,p=0.):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout = nn.Dropout(p)

        self.W_Q = nn.Linear(d_model,d_k*num_heads)
        self.W_K = nn.Linear(d_model,d_k*num_heads)
        self.W_V = nn.Linear(d_model,d_v*num_heads)
        self.W_out = nn.Linear(d_v*num_heads,d_model)

        nn.init.normal_(self.W_Q.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.W_K.weight, mean=0, std=np.sqrt(2.0/ (d_model + d_k)))
        nn.init.normal_(self.W_V.weight, mean=0, std=np.sqrt(2.0/ (d_model + d_v)))
        nn.init.normal_(self.W_out.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

    def forward(self,Q,K,V,attn_mask,**kwargs):
        N = Q.size(0)
        q_len, k_len = Q.size(1),K.size(1)
        d_k, d_v =self.d_k,self.d_v
        num_heads = self.num_heads

        # multi_head split
        Q = self.W_Q(Q).view(N,-1,num_heads,d_k).transpose(1,2)
        K = self.W_K(K).view(N,-1,num_heads,d_k).transpose(1,2)
        V = self.W_V(V).view(N,-1,num_heads,d_v).transpose(1,2)

        # pre-process mask
        if attn_mask is not None:
            assert attn_mask.size() == (N, q_len,k_len)
            attn_mask = attn_mask.unsqueeze(1).repeat(1,num_heads,1,1)
            attn_mask = attn_mask.bool()


        # calculate attention weight
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e4)
        attns = torch.softmax(scores, dim=-1)
        attns = self.dropout(attns)

        output = torch.matmul(attns,V)
        output = output.transpose(1,2).contiguous().reshape(N,-1,d_v*num_heads)

        output = self.W_out(output)

        return output

class PoswiseFFN(nn.Module):
    def __init__(self,d_model,d_ff,p=0.):
        super(PoswiseFFN,self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.conv1 = nn.Conv1d(d_model,d_ff,1,1,0)
        self.conv2 = nn.Conv1d(d_ff,d_model,1,1,0)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p)

    def forward(self,x):
        out = self.conv1(x.transpose(1,2))
        out = self.relu(out)
        out = self.conv2(out).transpose(1,2)
        out = self.dropout(out)
        return out

class EncoderLayer(nn.Module):
    def __init__(self,dim,n,dff,dropout_posffn,dropout_attn):
        assert dim%n == 0
        hdim = dim//n
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.multi_head_attn = MultiHeadAttention(hdim,hdim,dim,n,dropout_attn)
        self.poswise_ffn = PoswiseFFN(dim, dff,dropout_posffn)

    def forward(self,enc_in,attn_mask):
        residual = enc_in
        context = self.multi_head_attn(enc_in,enc_in,enc_in,attn_mask)
        out = self.norm1(residual + context)
        residual = out
        out = self.poswise_ffn(out)
        out = self.norm2(residual + out)

        return out

class Encoder(nn.Module):
    def __init__(
            self,dropout_emb,dropout_posffn,dropout_attn,
            num_layers,enc_dim,num_heads,dff,tgt_len
    ):
        super(Encoder, self).__init__()
        self.tgt_len = tgt_len
        self.enc_dim = enc_dim
        self.pos_emb = nn.Embedding.from_pretrained(pos_sinusoid_embedding(tgt_len,enc_dim),freeze=True)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.layers = nn.ModuleList(
            [EncoderLayer(enc_dim,num_heads,dff,dropout_posffn,dropout_attn) for _ in range(num_layers)]
        )
    def forward(self,X,X_lens,mask=None):
        batch_size, seq_len,d_model = X.shape
        a = self.pos_emb(torch.arange(seq_len,device=X.device))
        out = X + a
        out = self.emb_dropout(out)

        for layer in self.layers:
            out = layer(out,mask)
        return out

class DecoderLayer(nn.Module):
    def __init__(self,dim,n,dff,dropout_posffn,dropout_attn):
        super(DecoderLayer, self).__init__()
        assert dim%n == 0
        hdim = dim//n
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.poswise_ffn = PoswiseFFN(dim,dff,p=dropout_posffn)
        self.dec_attn = MultiHeadAttention(hdim,hdim,dim,n,dropout_attn)
        self.enc_dec_attn = MultiHeadAttention(hdim,hdim,dim,n,dropout_attn)

    def forward(self,dec_in,enc_out,dec_mask,dec_enc_mask,cache=None,freqs_cis=None):
        residual = dec_in
        context = self.dec_attn(dec_in,dec_in,dec_in,dec_mask)
        dec_out = self.norm1(context+residual)
        residual = dec_out
        context = self.enc_dec_attn(dec_out,enc_out,enc_out,dec_enc_mask)
        dec_out = self.norm2(context+residual)
        residual = dec_out
        out = self.poswise_ffn(dec_out)
        dec_out = self.norm3(out+residual)
        return dec_out

class Decoder(nn.Module):
    def __init__(
            self,dropout_emb,dropout_posffn,dropout_attn,
            num_layers,dec_dim,num_heads,dff,tgt_len,tgt_vocab_size
    ):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size,dec_dim)
        self.dropout_emb = nn.Dropout(p=dropout_emb)
        self.pos_emb = nn.Embedding.from_pretrained(pos_sinusoid_embedding(tgt_len,dec_dim),freeze=True)
        self.layers = nn.ModuleList([
            DecoderLayer(dec_dim,num_heads,dff,dropout_posffn,dropout_attn) for _ in range(num_layers)
            ]
        )
    def forward(self,labels,enc_out,dec_mask,dec_enc_mask,cache=None):
        tgt_emb = self.tgt_emb(labels)
        pos_emb = self.pos_emb(torch.arange(labels.size(1),device=labels.device))
        dec_out = self.dropout_emb(tgt_emb+pos_emb)
        for layer in self.layers:
            dec_out = layer(dec_out,enc_out,dec_mask,dec_enc_mask)
        return dec_out

class Transformer(nn.Module):
    def __init__(
            self,frontend:nn.Module,encoder:nn.Module,decoder:nn.Module,dec_out_dim:int,vocab:int
    )->None:
        super().__init__()
        self.frontend = frontend
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(dec_out_dim,vocab)
        self.vocab = vocab
        self.embedding = nn.Embedding(self.vocab,256)
        # self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self,X:torch.Tensor,X_lens:torch.Tensor,labels:torch.Tensor):
        X_lens, labels = X_lens.long(), labels.long()
        b = X.size(0)
        device = X.device
        out = self.embedding(X).squeeze(2)
        # frontend
        # out = self.frontend(out)
        # out = X
        max_feat_len = out.size(1)
        max_label_len = labels.size(1)
        # encoder
        enc_mask = get_len_mask(b,max_feat_len,X_lens,device)
        enc_out = self.encoder(out,X_lens,enc_mask)
        # decoder
        dec_mask = get_subsequent_mask(b,max_label_len,device)
        dec_enc_mask = get_enc_dec_mask(b,max_feat_len,X_lens,max_label_len,device)
        dec_out = self.decoder(labels, enc_out, dec_mask, dec_enc_mask)
        logits = self.linear(dec_out)
        # logits = self.softmax(logits)

        return logits

class TextDataset(Dataset):
    def __init__(self, my_dict, vocab):
        self.para = []
        self.tmp = []
        self.max_len = 30
        self.source = []
        for sentence in my_dict:
            for i in range(0, len(sentence) - self.max_len *2):
                self.para.append(sentence[i: i + self.max_len])
                self.tmp.append(['sos']+sentence[i + self.max_len:i+self.max_len*2])
        self.source = [
            ([vocab.get(char, vocab['<unk>']) for char in para], [vocab.get(char, vocab['<unk>']) for char in next_char])
            for para, next_char in zip(self.para, self.tmp)
            # if next_char[-1] in vocab
        ]
    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        sample = self.source[idx]
        source = np.copy(sample[0])
        target = np.copy(sample[1])
        return source, len(source),target,len(target)

def train(my_dict, vocab,id):
    train_dataset = TextDataset(my_dict, vocab)
    train_dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=True)

    # 设置gpu
    cuda_device = 0  # 这里假设第二块GPU的索引是1
    # cuda_device = 0
    torch.cuda.set_device(cuda_device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    max_feat_len = 30  # the maximum length of input sequence
    max_label_len = 31  # the maximum length of output sequence
    fbank_dim = 64  # the dimension of input feature
    hidden_dim = 256  # the dimension of hidden layer
    vocab_size = len(vocab)
    # Model
    feature_extractor = nn.Linear(fbank_dim, hidden_dim)  # A linear layer to simulate the audio feature extractor
    encoder = Encoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.0,
        num_layers=1, enc_dim=hidden_dim, num_heads=1, dff=256, tgt_len=max_feat_len
    )
    decoder = Decoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.0,
        num_layers=1, dec_dim=hidden_dim, num_heads=1, dff=256, tgt_len=max_label_len, tgt_vocab_size=vocab_size
    )
    model = Transformer(feature_extractor, encoder, decoder, hidden_dim, vocab_size).to(device)

    # Loss and Optimizer
    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    print(model)
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # model.load_state_dict(torch.load(f"model_{id}.pth"))
    # for src in train_dataloader:
    #     print("训练集一个batch的句子索引:", src)
    #     break
    n_epochs = 50
    # criterion = torch.nn.NLLLoss().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
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
            source, src_lens,target,tgt_lens = sample
            source = source.to(device).unsqueeze(-1)
            target = target.to(device)
            # target_one_hot_encoded = torch.nn.functional.one_hot(target, num_classes=len(vocab)).to(device)
            outputs = model(source, src_lens, target)
            next_word = torch.argmax(outputs, dim=-1)
            out = outputs[:,1:,:].reshape(-1, len(vocab))
            tar = target[:,1:].reshape(-1)
            loss = criterion(out, tar)
            loss.backward()
            loss_epoch += loss.detach().cpu()

            optimizer.step()  # 参数更新

        epoch_loss = loss_epoch / len(train_dataloader)
        epoch_losses.append(epoch_loss)

        # 打印当前 epoch 的信息
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.4f}, Time: {time.time() - start_time:.2f} seconds")

        torch.save(model.state_dict(), f"model_transformer_{id}.pth")
        if epoch_loss < best_validation_loss:
            best_validation_loss = epoch_loss
            # 保存模型参数
            torch.save(model.state_dict(), f"model_transformer_{id}.pth")
            patience_counter = 0
        else:
            # 如果验证损失没有降低，则增加等待计数器
            patience_counter += 1

            # 如果等待计数器超过指定的耐心值，则停止训练
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}.')
                break
def test(my_dict, vocab,id):
    dataset = TextDataset(my_dict, vocab)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
    batch_iterator = iter(dataloader)
    first_batch = next(batch_iterator)
    gg = first_batch[0][:1]

    gg_len = first_batch[1][:1]
    tar = first_batch[2][:1]
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

    max_feat_len = 30  # the maximum length of input sequence
    max_label_len = 31  # the maximum length of output sequence
    fbank_dim = 64  # the dimension of input feature
    hidden_dim = 256  # the dimension of hidden layer
    vocab_size = len(vocab)
    # Model
    feature_extractor = nn.Linear(fbank_dim, hidden_dim)  # A linear layer to simulate the audio feature extractor
    encoder = Encoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.0,
        num_layers=1, enc_dim=hidden_dim, num_heads=1, dff=256, tgt_len=max_feat_len
    )
    decoder = Decoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.0,
        num_layers=1, dec_dim=hidden_dim, num_heads=1, dff=256, tgt_len=max_label_len, tgt_vocab_size=vocab_size
    )
    model = Transformer(feature_extractor, encoder, decoder, hidden_dim, vocab_size).to(device)
    print(model)
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.load_state_dict(torch.load(f"model_transformer_{id}.pth"))
    model.eval()

    preds = model(gg.unsqueeze(-1).to(device), gg_len.to(device), tar.to(device))
    next_word = torch.argmax(preds, dim=-1)
    for i in range(1,len(next_word[0])):
        print(inv_vocab[next_word[:,i].item()],end='')
    # for _ in range(200):
    #     preds = model(gg.unsqueeze(-1).to(device), gg_len.to(device), tar.to(device))
    #     next_word = torch.argmax(preds, dim=-1)
    #     # gg = torch.cat((gg, next_word[:,:1].cpu()),dim=-1)
    #     # gg = gg[:,1:]
    #     print(inv_vocab[next_word[:,:1].item()],end='')
if __name__ == "__main__":
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
    for id, wd in enumerate(word_dict):
        if wd in data_source:
            my_dict = word_dict[wd]

            """词汇表生成"""
            # 第一次运行时取消以下注释#########
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
            vocab['sos'] = len(vocab)
            model = 'test'
            if model == 'train':
                train(my_dict, vocab, id)
            elif model == 'test':
                test(my_dict, vocab, id)