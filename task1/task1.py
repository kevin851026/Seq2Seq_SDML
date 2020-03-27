import pandas as pd
import torch
import torch.nn.functional as F
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
    with open(predict_file,'rb') as f_i:
        predict = f_i.readlines()
    with open(true_file,'rb') as f_o:
        true = f_o.readlines()

    correct = 0
    assert len(predict) == len(true), "Predict file and true file should have same length."
    for i in range(len(predict)):
        if predict[i] == true[i]:
            correct += 1
    return (correct/len(predict))
def get_acc(val_set,name,encoder,decoder):
    batch_size=512
    val_loader = DataLoader(val_set, batch_size=batch_size,collate_fn=pad_batch)
    val_set = open(name+'_set.csv', 'w', newline='', encoding='utf-8')
    val_ans = open(name+'_ans.csv', 'w', newline='', encoding='utf-8')
    v_writer = csv.writer(val_set,delimiter=' ')
    a_writer = csv.writer(val_ans,delimiter=' ')
    encoder.eval()
    decoder.eval()
    print(name)
    # [1, 343, 3, 302, 2]
    for step, (batch) in enumerate(val_loader):
        input_tensor,target_tensor=[t.to(device) for t in batch]
        shape=input_tensor.shape
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        # decoder_hidden = encoder_hidden.view(1,batch_size,-1)
        decoder_hidden = encoder_hidden
        decoder_input = torch.tensor([[1]]*shape[0], device=device)
        file_output = decoder_input.detach()
        for di in range(1,30):
            decoder_output, decoder_hidden = decoder(decoder_input,decoder_hidden)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.view(shape[0],-1)
            file_output = torch.cat((file_output,topi.view(shape[0],-1).detach()),dim=1)
        tensor_len=[]
        for line in input_tensor.tolist():
            try:
                length=line.index(0)
            except:
                length=len(line)
            line=line[:length]
            line = [vocab[index] for index in line]
            tensor_len.append(line.index('<EOS>')+1)
            v_writer.writerow(line)
        for line,length in zip(file_output.tolist(),tensor_len):
            line = [vocab[index] for index in line]
            try:
                eos = line.index('<EOS>')+1
            except:
                eos = len(line)
            line = line[:eos]
            a_writer.writerow(line)
        l = input_tensor.tolist()
        for i in range(0,len(l)):
            x=l[i].copy()[:l[i].index(2)+1]
            # print(x)
            if x == [1, 3, 53, 50, 2]:
                print(l[i])
                print(file_output.tolist()[i])
                break
    val_set.close()
    val_ans.close()
    v_acc = 0
    # v_acc = evaluate(name+'_ans.csv',name+'_set.csv')
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
        self.x=0
    def __getitem__(self, index):
        # print(self.data[index])
        text = self.data[index].split()
        # print(text)
        text_ids = []
        for i in text:
            try:
                x=self.vocab.index(i)
                text_ids.append(x)
            except:
                # print('UNK: ',i)
                self.x+=1
                text_ids.append(1111)
        if self.data[index].replace("\n",'') == '<SOS> tom looked confused <EOS>':
            print(text)
            print(text_ids)
            # exit()
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
    def __init__(self, input_size,embedding_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.Sequential(
                        nn.GRU(embedding_size, hidden_size,batch_first=True,num_layers=1,bidirectional=True)
                    )
    def forward(self, input):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded)
        # print(output.shape)
        # print(hidden.shape)
        return output, hidden
class DecoderRNN(nn.Module):
    def __init__(self, output_size,hidden_size ):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size,num_layers=1,batch_first=True,bidirectional=True)
        self.out = nn.Linear(hidden_size*2, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input,hidden):
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
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(0))

    batch_size = 256
    embedding_size = 256+128
    hidden_size = 256
    learning_rate = 0.0008

    torch.manual_seed(123456789)
    dataset = Dataset('train.txt')
    vocab = dataset.vocab
    testset = Dataset('hw2.0_testing_data.txt')
    trainset, valset  = data.random_split(dataset, (int(len(dataset)*0.8) ,dataset.len-int(len(dataset)*0.8)))
    train_loader = DataLoader(trainset, batch_size=batch_size,collate_fn=pad_batch,drop_last=True)
    encoder = EncoderRNN(len(vocab),embedding_size, hidden_size)
    decoder = DecoderRNN(len(vocab), hidden_size)
    for m in encoder.modules():
        if isinstance(m, (nn.Linear)):
            nn.init.orthogonal_(m.weight)
    for m in decoder.modules():
        if isinstance(m, (nn.Linear)):
            nn.init.orthogonal_(m.weight)
            # nn.init.kaiming_normal_(m.weight, mode='fan_in')
    encoder.load_state_dict(torch.load('task1_encoder_v1.pkl'))
    decoder.load_state_dict(torch.load('task1_decoder_v1.pkl'))
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate,weight_decay=1e-5)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate,weight_decay=1e-5)
    print('encoder: ',str(sum(p.numel() for p in filter(lambda p: p.requires_grad, encoder.parameters()))))
    print('decoder: ',str(sum(p.numel() for p in filter(lambda p: p.requires_grad, decoder.parameters()))))
    print('-'*30)
    teacher_forcing_ratio = 0.7
    # test_acc = get_acc(testset,'test',encoder,decoder)
    v_acc = get_acc(valset,'val',encoder,decoder)
    t_acc = get_acc(dataset,'train',encoder,decoder)
    # print('    val acc: ' + str(v_acc),'    test acc: ' + str(test_acc))
    exit()
    for epoch in range(150):
        train_ls = 0
        encoder.train()
        decoder.train()
        print(time.asctime( time.localtime(time.time()) ))
        for step, (batch) in enumerate(train_loader):
            input_tensor,target_tensor=[t.to(device) for t in batch]
            encoder_outputs, encoder_hidden = encoder(input_tensor)
            # decoder_hidden = encoder_hidden.view(1,batch_size,-1)
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
            train_ls += loss.item()
            # if step % 100 == 0:
            #     print('      ' + str(step) + ' train loss: ' + str(train_ls/((step+1)*batch_size)))
        torch.save(encoder.state_dict(), 'task1_encoder_v1'+'.pkl')
        torch.save(decoder.state_dict(), 'task1_decoder_v1'+'.pkl')
        print('Epoch: ' + str(epoch) + ' train loss: ' + str(train_ls/((step+1)*batch_size)))
        v_acc = get_acc(valset,'val',encoder,decoder)
        test_acc = get_acc(testset,'test',encoder,decoder)
        if v_acc > 0.80:
            t_acc = get_acc(dataset,'train',encoder,decoder)
            print('train acc: ' + str(t_acc) +' val acc: ' + str(v_acc),'    test acc: ' + str(test_acc))
        else:
            print('    val acc: ' + str(v_acc),'    test acc: ' + str(test_acc))