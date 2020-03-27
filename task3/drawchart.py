import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import csv
import time
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
class Dataset:
    def __init__(self, file, train=True):
        self.train = train
        with open(file, encoding='utf-8') as f:
            self.data = f.readlines()
        with open('vocab.csv',encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            self.vocab = next(reader)
        self.len = len(self.data)
    def __getitem__(self, index):
        if self.train:
            try:
                text,target= self.data[index].replace('\n','').split(',,')
            except:
                print(self.data[index])
            text = text.split()
            target = target.split()
            text_ids = [self.vocab.index(i) for i in text]
            target_ids = [self.vocab.index(i) for i in target]
            token_tensor = torch.tensor(text_ids)
            target_tensor = torch.tensor(target_ids)
            return (token_tensor,target_tensor)
        else:
            text = self.data[index].split()
            text_ids = []
            for i in text:
                try:
                    text_ids.append(self.vocab.index(i))
                except:
                    print('UNK: ',i)
                    text_ids.append(4)
            token_tensor = torch.tensor(text_ids)
            return (token_tensor,token_tensor)
    def __len__(self):
        return self.len
def pad_batch(batch):
    (tokens,targets) = zip(*batch)
    tokens_pad = pad_sequence(tokens, batch_first=True)
    targets_pad = pad_sequence(targets, batch_first=True)
    return tokens_pad,targets_pad
def embedding(w):
  return words.iloc[w].as_matrix()
class EncoderRNN(nn.Module):
    def __init__(self, input_size,embedding_size, hidden_size,batch_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size,num_layers=1,batch_first=True,bidirectional=False)
        self.hidden = torch.zeros(1, batch_size, self.hidden_size, device=device)
    def forward(self, input,hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded,hidden)

        gi = F.linear(embedded.view(embedded.shape[0],-1), w_ih, b_ih)
        gh = F.linear(hidden[0], w_hh, b_hh)
        i_r, i_i, i_n = gi.chunk(3, 1)
        h_r, h_i, h_n = gh.chunk(3, 1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        # newgate = torch.tanh(i_n + resetgate * h_n)
        # hy = newgate + inputgate * (hidden - newgate)
        mean = torch.mean(inputgate,dim=1)
        return output, hidden ,mean
class DecoderRNN(nn.Module):
    def __init__(self, output_size,hidden_size ):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size,num_layers=1,batch_first=True,bidirectional=False)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output,hidden)
        output = self.out(output)
        output = self.softmax(output+1e-7)
        return output,hidden
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(0))

    batch_size = 256
    embedding_size = 256
    hidden_size = 128
    learning_rate = 0.0008

    torch.manual_seed(0)
    dataset = Dataset('train4.txt')
    vocab = dataset.vocab
    trainset, valset  = data.random_split(dataset, (int(len(dataset)*0.8) ,dataset.len-int(len(dataset)*0.8)))
    train_loader = DataLoader(valset, batch_size=batch_size, shuffle=True,collate_fn=pad_batch,drop_last=True)
    encoder = EncoderRNN(len(vocab),embedding_size, hidden_size,batch_size)
    decoder = DecoderRNN(len(vocab), hidden_size)
    for m in encoder.modules():
        if isinstance(m, (nn.Linear)):
            nn.init.orthogonal_(m.weight)
    for m in decoder.modules():
        if isinstance(m, (nn.Linear)):
            nn.init.orthogonal_(m.weight)
    encoder.load_state_dict(torch.load('task2_encoder_v4.pkl'))
    decoder.load_state_dict(torch.load('task2_decoder_v4.pkl'))
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate,weight_decay=1e-5)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate,weight_decay=1e-5)
    print('encoder: ',str(sum(p.numel() for p in filter(lambda p: p.requires_grad, encoder.parameters()))))
    print('decoder: ',str(sum(p.numel() for p in filter(lambda p: p.requires_grad, decoder.parameters()))))
    print('-'*30)
    teacher_forcing_ratio = 0
    # testset = Dataset('hw2.1-1_testing_data.txt',train=False)
    # trainset, valset  = data.random_split(dataset, (int(len(dataset)*0.99) ,dataset.len-int(len(dataset)*0.99)))
    # v_acc = get_acc(testset,'e',encoder,decoder)
    # v_acc = evaluate('val_ans.csv','val_set.csv')
    # print(v_acc)
    # exit()
    for epoch in range(1):
        train_ls = 0
        encoder.eval()
        decoder.eval()
        plt.figure(2)
        plt.rcParams['figure.figsize'] = (12.0, 6.0)
        plt.rcParams['savefig.dpi'] = 200
        plt.rcParams['figure.dpi'] = 200
        chart_data=[]
        for i in range(25):
            chart_data.append([0 for j in range(i)])
        count = [0 for i in range(25)]
        w_ih = 0
        b_ih = 0
        w_hh = 0
        b_hh = 0
        for name in encoder.named_parameters():
            # print(name[0])
            if name[0] == 'gru.weight_ih_l0':
                w_ih = name[1]
            if name[0] == 'gru.bias_ih_l0':
                b_ih = name[1]
            if name[0] == 'gru.weight_hh_l0':
                w_hh = name[1]
            if name[0] == 'gru.bias_hh_l0':
                b_hh = name[1]
        # print(time.asctime( time.localtime(time.time()) ))
        for step, (batch) in enumerate(train_loader):
            print(step)
            input_tensor,target_tensor=[t.to(device) for t in batch]
            # print(input_tensor.shape)
            shape = input_tensor.shape
            encoder_hidden = encoder.hidden
            encoder_outputs = None
            tmp = [[] for i in range(batch_size)]
            for i in range(len(input_tensor.transpose(0,1))):
                encoder_output, encoder_hidden ,mean= encoder(input_tensor.transpose(0,1)[i].view(batch_size,-1),encoder_hidden)
                for j in range(batch_size):
                    tmp[j].append(float(mean[j]))
            for i in range(len(input_tensor)):
                try:
                    length = input_tensor[i].tolist().index(0)
                except:
                    length = len(input_tensor[i])
                for j in range(length):
                    chart_data[length][j] += tmp[i][j]
                count[length] += 1
            decoder_hidden = encoder_hidden
            decoder_input = torch.tensor([[1]]*batch_size, device=device)
            criterion = nn.CrossEntropyLoss(reduction='sum',ignore_index=0)
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            for di in range(1,len(target_tensor[0])):
                decoder_output, decoder_hidden = decoder(decoder_input,decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.view(batch_size,-1)
        print(count)
        for i in range(25):
            if count[i] == 0 :
                continue
            else:
                plt.plot([j for j in range(1,i+1) ],[chart_data[i][j]/count[i] for j in range(i)],label=str(i))
        plt.xlabel('timestep',fontsize=20)
        plt.xticks(np.linspace(0,20,21))
        plt.grid(True,axis="x",ls='--')
        plt.savefig('gate.jpg')
        plt.close('all')
        # plt.show()
