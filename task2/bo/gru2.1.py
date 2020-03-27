import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import torch.nn.functional as F
import random
import csv
from progressbar import *
MAX_LENGTH = 30
BATCH_SIZE = 512
EPOCH = 81
VOC_SIZE = 6593
EMBEDDING_SIZE = 300
FORCING_RATIO = 0.9
SPECIAL_TOKENS = {"<SOS>": 6589, "<EOS>": 6590, "<UNK>": 6591, "<PAD>": 6592}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class Hw211Dataset(Dataset):
    def __init__(self, fileX, fileY, vocab_ids, train = True):
        self.train = train
        with open(fileX, 'r', encoding='utf-8') as f:
            self.data = f.read().split('\n')[:-1]
        if self.train:
            with open(fileY, 'r', encoding='utf-8') as f:
                self.label = f.read().split('\n')[:-1]
        with open(vocab_ids, 'r', encoding='utf-8') as f:
            self.vocab_ids = json.load(f)
        
        self.len = len(self.data)

    def __getitem__(self, index):
        wordsx = self.data[index].split()
        # print(wordsx)
        if len(wordsx)>30:
            print(wordsx)
        sentence_encode = []
        for word in wordsx:
            try:
                sentence_encode.append(self.vocab_ids[word])
            except:
                sentence_encode.append(self.vocab_ids[SPECIAL_TOKENS['<UNK>']])
        if self.train:
            wordsy = self.label[index].split()
            # print(wordsy)
            label_encode = []
            for word in wordsy:
                try:
                    label_encode.append(self.vocab_ids[word])
                except:
                    label_encode.append(self.vocab_ids[SPECIAL_TOKENS['<UNK>']])
        else:
            label_encode = None

        token_tensor = torch.tensor(sentence_encode)
        label_tensor = torch.tensor(label_encode)

        return (token_tensor, label_tensor)

    def __len__(self):
        return self.len

def pad_batch(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    (tokens, labels) = zip(*batch)
    tokens_length = [len(t) for t in tokens]

    if batch[0][1] is None:
        labels = None
    else:
        labels = pad_sequence(labels, batch_first=True, padding_value=SPECIAL_TOKENS['<PAD>'])
        labels_trans = labels.permute(1, 0)
        labels_trans = torch.LongTensor(labels_trans)
    tokens_pad = pad_sequence(tokens, batch_first=True, padding_value=SPECIAL_TOKENS['<PAD>'])
    tokens_pad = torch.LongTensor(tokens_pad)

    return tokens_pad, labels_trans, tokens_length

class Encoder(nn.Module):
    def __init__(self, embedding, hidden_size, dropout):
        super(Encoder, self).__init__()

        self.embedding = embedding
        self.gru = nn.GRU(input_size=EMBEDDING_SIZE, hidden_size=hidden_size, batch_first=True, num_layers=1, bidirectional=True, dropout=dropout)

    def forward(self, input_ids, tokens_length):
        embeds = self.embedding(input_ids)
        embeds_packed = pack_padded_sequence(embeds, tokens_length, batch_first=True)
        encode, h_n = self.gru(embeds_packed.float())
        encode_pad, encode_len = pad_packed_sequence(encode, batch_first=True)

        return encode_pad, h_n#, encode_len

class Decoder(nn.Module):
    def __init__(self, embedding, hidden_size, output_size, dropout):
        super(Decoder, self).__init__()

        self.embedding = embedding
        self.gru = nn.GRU(input_size=EMBEDDING_SIZE, hidden_size=hidden_size, num_layers=1, bidirectional=True, dropout=dropout)
        self.out = nn.Linear(hidden_size*2, output_size)
        nn.init.orthogonal_(self.out.weight)
        # self.out.weight = nn.Parameter()

    def forward(self, input_ids, h_n):
        embeds = self.embedding(input_ids)
        gru_output, h_n = self.gru(embeds.float(), h_n)
        gru_output = gru_output.squeeze(0)
        output = self.out(gru_output)

        return output, h_n

class Model(nn.Module):
    def __init__(self, embedding, hidden_size, voc_size, dropout=0):
        super(Model, self).__init__()
        self.embedding_en = nn.Embedding(voc_size, EMBEDDING_SIZE)
        # self.embedding.weight = nn.Parameter(torch.from_numpy(embedding.astype(float)))
        self.encoder = Encoder(embedding=self.embedding_en, hidden_size=hidden_size, dropout=dropout)
        self.embedding_de = nn.Embedding(voc_size, EMBEDDING_SIZE)
        self.decoder = Decoder(embedding=self.embedding_de, hidden_size=hidden_size, output_size=voc_size, dropout=dropout)

    def forward(self, input_ids, tokens_length, batch_size, teacher_forcing=False, labels=None):
        encoder_output, encoder_hidden = self.encoder(input_ids, tokens_length)
        encoder_output = encoder_output.permute(1, 0, 2)

        decoder_input = Variable(torch.LongTensor([[SPECIAL_TOKENS['<SOS>']]*batch_size]))
        decoder_input = decoder_input.to(device)
        decoder_hidden = encoder_hidden

        loss = 0
        predict = torch.zeros(MAX_LENGTH, batch_size)
        # print(predict[0].shape)
        for i in range(MAX_LENGTH-2):
            decoder_output_i, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            _, predict_index = decoder_output_i.topk(1)
            predict_index = predict_index.view(-1)

            # print(labels.shape)
            if teacher_forcing and random.random() < FORCING_RATIO:
                decoder_input = Variable(torch.LongTensor([[labels[i][j] for j in range(batch_size)]]))
            else:
                decoder_input = Variable(torch.LongTensor([[predict_index[j] for j in range(batch_size)]]))
            decoder_input = decoder_input.to(device)

            predict[i] = predict_index

            if labels is not None:
                loss += F.cross_entropy(decoder_output_i, labels[i], ignore_index=SPECIAL_TOKENS['<PAD>'], reduction='sum') #, reduction='sum'
                if i == len(labels)-1:
                    break
            

        predict = predict.type(torch.LongTensor)
        if labels is not None:
            return loss, predict
        # print(predict)
        return predict

def evaluate_predict(model, loader, vocab, datalen, testing=False):
    print('evaluate:　' + torch.cuda.get_device_name(0))
    correct = 0
    total = 0
    predict = []
    model.eval()
    val_set = open('val_set.csv', 'w', newline='', encoding='utf-8')
    val_ans = open('val_ans.csv', 'w', newline='', encoding='utf-8')
    v_writer = csv.writer(val_set,delimiter=' ')
    a_writer = csv.writer(val_ans,delimiter=' ')
    with torch.no_grad():
        pbar = ProgressBar().start()
        for step, (batch) in enumerate(loader):
            if testing:
                tokens, labels, length = [t for t in batch]
                tokens = tokens.to(device)
            else:
                tokens, labels, length = [t for t in batch]
                tokens = tokens.to(device)
                labels = labels.to(device)
                labels = labels.permute(1, 0)
            output = model(input_ids=tokens, tokens_length=length, batch_size=len(tokens))
            output = output.to(device)
            output = output.permute(1, 0)
            for line in tokens.tolist():
                try:
                    length=line.index(SPECIAL_TOKENS['<PAD>'])
                except:
                    length=len(line)
                line=line[:length]
                line = [vocab[index] for index in line]
                v_writer.writerow(line)
            for line in output.tolist():
                # print(line)
                line = [vocab[index] for index in line]
                # print(line)
                try:
                    eos = line.index('<EOS>')+1
                except:
                    eos = len(line)
                line = line[:eos]
                a_writer.writerow(line)
            if not testing:
                for i in range(len(output)):
                    try:
                        out = output[i].tolist()
                        out = out[:out.index(SPECIAL_TOKENS['<EOS>'])+1]
                        lab = labels[i].tolist()
                        lab = lab[:lab.index(SPECIAL_TOKENS['<EOS>'])+1]
                    #     print(out)
                    except:
                        out = output[i].tolist()
                        lab = labels[i].tolist()
                    if out == lab:
                        correct += 1

                    sentence = ''
                    for j in range(len(out)-1):
                        sentence += vocab[out[j]] + ' '
                    sentence += vocab[out[len(out)-1]]
                    predict.append(sentence)
                total += len(output)

            else:
                for i in range(len(output)):
                    try:
                        out = output[i].tolist()
                        out = out[:out.index(SPECIAL_TOKENS['<EOS>'])+1]
                    except:
                        out = output[i].tolist()

                    sentence = ''
                    for j in range(len(out)-1):
                        sentence += vocab[out[j]] + ' '
                    sentence += vocab[out[len(out)-1]]
                    predict.append(sentence)
                total += len(output)

            pbar.update(int((step+1)/(datalen/BATCH_SIZE)*100))
        pbar.finish()
    val_set.close()
    val_ans.close()
    if testing:
        acc = None
    else:
        acc = correct/total
    return predict, acc

def evaluate(train, output):
    all_assign_cnt = 0
    correct_cnt = 0
    with open(train, 'r') as f:
        train_data = f.read().split('\n')[:-1]
    with open(output, 'r') as f:
        result_data = f.read().split('\n')[:-1]

    for i, train_line in enumerate(train_data):
        try:
            result_line = result_data[i].strip().split()
        except:
            result_line = []

        train_line = train_line.split(',')[0].strip()
        control_signal = train_line.split('<EOS>')[1].strip().split()
        control_cnt = len(control_signal) // 2
        all_assign_cnt += control_cnt
        for j in range(control_cnt):
            position, word = control_signal[j*2: j*2+2]
            position = int(position)
            if position < len(result_line) and result_line[position] == word:
                correct_cnt += 1

    print('accuracy: ', correct_cnt / all_assign_cnt)

if __name__ == '__main__':
    print(torch.cuda.get_device_name(0))
    torch.manual_seed(0)
    with open('vocab.txt', 'r', encoding='utf-8') as f:
        vocab = f.read().split()
    dataset = Hw211Dataset('train4.txt', 'label4.txt', 'vocab.json')
    trainset, valset = data.random_split(dataset, (int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)))
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_batch)
    val_loader = DataLoader(valset, batch_size=BATCH_SIZE, collate_fn=pad_batch)
    # word_embedding = np.load('word_embedding.npy')
    word_embedding = []

    model = Model(word_embedding, hidden_size=300, voc_size=VOC_SIZE)
    model = torch.load('model_redsum_0epo.pkl')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=pad_batch)
    vpred, vacc = evaluate_predict(model, val_loader, vocab, len(valset))
    # print(vacc)
    with open('out.txt', 'w', encoding='utf-8') as f:
        for line in vpred:
            f.write(line+'\n')
    evaluate('val_set.csv','val_ans.csv')
    # exit()
    for epoch in range(EPOCH):
        train_ls = 0
        train_step = 0
        model.train()
        # pbar = ProgressBar().start()
        for step, (batch) in enumerate(train_loader):
            tokens, labels, length = [t for t in batch]
            tokens = tokens.to(device)
            labels = labels.to(device)
            # print(tokens)
            # print(labels)
            output = model(input_ids=tokens, tokens_length=length, batch_size=len(tokens), teacher_forcing=True, labels=labels)
            optimizer.zero_grad()
            loss = output[0]
            loss.backward()
            optimizer.step()
            # exit()
            train_ls += loss.item()
            train_step = step
            # if step%300 == 0:
            # print('step: ' + str(step))
            # print(step)
            if step % 100 == 0:
                print('      ' + str(step) + ' train loss: ' + str(train_ls/((step+1)*BATCH_SIZE)))
            # pbar.update(int((step+1)/(len(trainset)/BATCH_SIZE)*100))
        # pbar.finish()

        # tpred, tacc = evaluate_predict(model, train_loader, vocab, len(trainset))
        vpred, vacc = evaluate_predict(model, val_loader, vocab, len(valset))
        print('Epoch: ' + str(epoch) + ' train loss: ' + str(train_ls/(train_step+1)*BATCH_SIZE))
        evaluate('val_set.csv','val_ans.csv')
        # print(' val acc: ' + str(vacc)) #'train acc: ' + str(tacc) +
        # print()
        torch.save(model, 'model_redsum_{}epo.pkl'.format(epoch))
        # if epoch%10 == 0:
        #     torch.save(model, 'model_redsum_{}epo.pkl'.format(epoch))

        #     test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=pad_batch)
        #     testpred, testacc = evaluate_predict(model, test_loader, vocab, len(dataset))
        #     print(testacc)
        #     with open('out.txt', 'w', encoding='utf-8') as f:
        #         for line in testpred:
        #             f.write(line+'\n')
        #     evaluate('train4.txt', 'out.txt')

