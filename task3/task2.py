import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json
import random
import csv
import time
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
def evaluate(predict_file,true_file):
    all_assign_cnt = 0
    correct_cnt = 0
    with open(true_file, 'r', encoding='utf-8') as f:
        train_data = f.read().split('\n')[:-1]
    with open(predict_file, 'r', encoding='utf-8') as f:
        result_data = f.read().split('\n')[:-1]
    for i, train_line in enumerate(train_data):
        try:
            result_line = result_data[i].strip().split()
        except:
            result_line = []

        # train_line = train_line.split(',')[0].strip()
        # print(train_line)
        control_signal = train_line.split('<SOS>')[0].strip().split()
        control_cnt = len(control_signal) // 2
        all_assign_cnt += control_cnt
        for j in range(control_cnt):
            position, word = control_signal[j*2: j*2+2]
            position = int(position)
            if position < len(result_line) and result_line[position] == word:
                correct_cnt += 1
    return (correct_cnt / all_assign_cnt)
def get_acc(val_set,name,encoder,decoder):
    batch_size=256
    val_loader = DataLoader(val_set, batch_size=batch_size,collate_fn=pad_batch,drop_last=True)
    val_set = open(name+'_set.csv', 'w', newline='', encoding='utf-8')
    val_ans = open(name+'_ans.csv', 'w', newline='', encoding='utf-8')
    v_writer = csv.writer(val_set,delimiter=' ')
    a_writer = csv.writer(val_ans,delimiter=' ')
    encoder.eval()
    decoder.eval()
    for step, (batch) in enumerate(val_loader):
        input_tensor,target_tensor=[t.to(device) for t in batch]
        shape=input_tensor.shape
        encoder_hidden = encoder.hidden
        for i in range(len(input_tensor.transpose(0,1))):
            encoder_output, encoder_hidden = encoder(input_tensor.transpose(0,1)[i].view(shape[0],-1),encoder_hidden)
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([[1]]*shape[0], device=device)
        file_output = decoder_input.detach()
        for di in range(1,50):
            decoder_output, decoder_hidden = decoder(decoder_input,decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.view(shape[0],-1)
            file_output = torch.cat((file_output,topi.view(shape[0],-1).detach()),dim=1)
        for line in input_tensor.tolist():
            try:
                length=line.index(0)
            except:
                length=len(line)
            line=line[:length]
            line = [vocab[index] for index in line]
            v_writer.writerow(line)
        for line in file_output.tolist():
            line = [vocab[index] for index in line]
            try:
                eos = line.index('<EOS>')+1
            except:
                eos = len(line)
            line = line[:eos]
            a_writer.writerow(line)
        # break
        # print(step)
    val_set.close()
    val_ans.close()
    v_acc = evaluate(name+'_ans.csv',name+'_set.csv')
    return v_acc
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
            # print(self.data[index].replace('\n',''))
            try:
                text,target= self.data[index].replace('\n','').split(',,')
            except:
                print(self.data[index])
                # print('hi')
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
        # print(output.shape)
        # print(hidden.shape)
        # w_ih = 0
        # b_ih = 0
        # w_hh = 0
        # b_hh = 0
        # for name in encoder.named_parameters():
        #     print(name[0])
        #     if name[0] == 'gru.weight_ih_l0':
        #         w_ih = name[1]
        #     if name[0] == 'gru.bias_ih_l0':
        #         b_ih = name[1]
        #     if name[0] == 'gru.weight_hh_l0':
        #         w_hh = name[1]
        #     if name[0] == 'gru.bias_hh_l0':
        #         b_hh = name[1]
        # print(embedded.shape)
        # print(w_ih.shape)
        # print(b_ih.shape)
        # gi = F.linear(embedded[0], w_ih, b_ih)
        # gh = F.linear(hidden[0], w_hh, b_hh)
        # print(gi.shape)
        # i_r, i_i, i_n = gi.chunk(3, 1)
        # h_r, h_i, h_n = gh.chunk(3, 1)
        # resetgate = F.sigmoid(i_r + h_r)
        # inputgate = F.sigmoid(i_i + h_i)
        # newgate = F.tanh(i_n + resetgate * h_n)
        # print(resetgate.shape)
        return output, hidden
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
        # print(output)
        # print('-'*30)
        return output,hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
def draw_chart(chart_data,outfile_name):
    plt.figure()
    plt.rcParams['figure.figsize'] = (12.0, 6.0)
    plt.rcParams['savefig.dpi'] = 200
    plt.rcParams['figure.dpi'] = 200
    plt.plot(chart_data['epoch'],chart_data['tarin_loss'],label='tarin_loss')
    plt.grid(True,axis="y",ls='--')
    plt.legend(loc= 'best')
    plt.xlabel('epoch',fontsize=20)
    # plt.yticks(np.linspace(0,1,11))
    plt.savefig(outfile_name+'_tarin_loss.jpg')
    plt.close('all')
    plt.figure()
    plt.rcParams['figure.figsize'] = (12.0, 6.0)
    plt.rcParams['savefig.dpi'] = 200
    plt.rcParams['figure.dpi'] = 200
    plt.plot(chart_data['epoch'],chart_data['val_acc'],label='val_acc')
    plt.grid(True,axis="y",ls='--')
    plt.legend(loc= 'best')
    plt.xlabel('epoch',fontsize=20)
    # plt.yticks(np.linspace(0,1,11))
    plt.savefig(outfile_name+'_val_acc.jpg')
    plt.close('all')
    with open(outfile_name+'.json','w') as file_object:
        json.dump(chart_data,file_object)
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
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,collate_fn=pad_batch,drop_last=True)
    encoder = EncoderRNN(len(vocab),embedding_size, hidden_size,batch_size)
    decoder = DecoderRNN(len(vocab), hidden_size)
    for m in encoder.modules():
        if isinstance(m, (nn.Linear)):
            nn.init.orthogonal_(m.weight)
    for m in decoder.modules():
        if isinstance(m, (nn.Linear)):
            nn.init.orthogonal_(m.weight)
            # nn.init.kaiming_normal_(m.weight, mode='fan_in')
    # encoder.load_state_dict(torch.load('task2_encoder_v4.pkl'))
    # decoder.load_state_dict(torch.load('task2_decoder_v4.pkl'))
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate,weight_decay=1e-5)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate,weight_decay=1e-5)
    print('encoder: ',str(sum(p.numel() for p in filter(lambda p: p.requires_grad, encoder.parameters()))))
    print('decoder: ',str(sum(p.numel() for p in filter(lambda p: p.requires_grad, decoder.parameters()))))
    print('-'*30)
    teacher_forcing_ratio = 0.75
    # testset = Dataset('hw2.1-1_testing_data.txt',train=False)
    # trainset, valset  = data.random_split(dataset, (int(len(dataset)*0.99) ,dataset.len-int(len(dataset)*0.99)))
    # v_acc = get_acc(testset,'e',encoder,decoder)
    # v_acc = evaluate('val_ans.csv','val_set.csv')
    # print(v_acc)
    # exit()
    chart_data={"tarin_loss":[],"val_acc":[],"epoch":[]}
    for epoch in range(15000):
        train_ls = 0
        encoder.train()
        decoder.train()
        # print(time.asctime( time.localtime(time.time()) ))
        for step, (batch) in enumerate(train_loader):
            input_tensor,target_tensor=[t.to(device) for t in batch]
            shape = input_tensor.shape
            encoder_hidden = encoder.hidden
            encoder_outputs = None
            for i in range(len(input_tensor.transpose(0,1))):
                encoder_output, encoder_hidden = encoder(input_tensor.transpose(0,1)[i].view(batch_size,-1),encoder_hidden)
                if encoder_outputs is None:
                    encoder_outputs = encoder_output
                else:
                    encoder_outputs = torch.cat((encoder_outputs,encoder_output),dim=1)
            decoder_hidden = encoder_hidden
            decoder_input = torch.tensor([[1]]*batch_size, device=device)
            file_output = decoder_input
            criterion = nn.CrossEntropyLoss(reduction='sum',ignore_index=0)
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            # use_teacher_forcing = True
            loss=0
            for di in range(1,len(target_tensor[0])):
                decoder_output, decoder_hidden = decoder(decoder_input,decoder_hidden)
                loss += criterion(decoder_output.view(batch_size,-1), target_tensor.transpose(0,1)[di])
                topv, topi = decoder_output.topk(1)
                if use_teacher_forcing:
                    decoder_input = target_tensor.transpose(0,1)[di].view(batch_size,-1)  # Teacher forcing
                else:
                    decoder_input = topi.view(batch_size,-1)
                file_output = torch.cat((file_output,topi.view(batch_size,-1).detach()),dim=1)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            encoder.hidden = encoder_hidden.detach()
            train_ls += loss.item()
        torch.save(encoder.state_dict(), 'task2_encoder_v4'+'.pkl')
        torch.save(decoder.state_dict(), 'task2_decoder_v4'+'.pkl')
        v_acc = get_acc(valset,'val',encoder,decoder)
        print('Epoch: ' + str(epoch) + ' train loss: ' + str(train_ls/((step+1)*batch_size)) +'    val acc: ' + str(v_acc))
        if v_acc > 0.80:
            t_acc = get_acc(trainset,'train',encoder,decoder)
            print('train acc: ' + str(t_acc) +' val acc: ' + str(v_acc))
        chart_data['epoch'].append(epoch)
        chart_data['tarin_loss'].append(train_ls/((step+1)*batch_size))
        chart_data['val_acc'].append(v_acc)
        draw_chart(chart_data,'model')