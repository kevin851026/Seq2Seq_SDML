from pytorch_transformers import BertTokenizer
import csv
import random
x=[]
t=0
#-----------creat vocab------------------
#
# with open('200_vocab_dataset.txt', encoding='utf-8') as f:
#     for line in f.readlines():
#         # print(line)
#         # t+=1
#         # if t %10000 == 0:
#         #     print(t)
#         for word in line.split():
#             if word != "\n":
#                 if word.replace("\n", "") not in x:
#                     x.append(word.replace("\n", ""))
#         # if t==10:
#         #     break
# with open('vocab.csv', 'w', newline='', encoding='utf-8') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(x)
# with open('vocab.csv', newline='', encoding='utf-8') as csvfile:
#     reader = csv.reader(csvfile)
#     vocab = next(reader)
#     print(len(vocab))
#     print(vocab[0])
#-----------creat vocab------------------
with open('200_vocab_dataset.txt', newline='', encoding='utf-8') as file,\
        open('train.txt','w', newline='', encoding='utf-8') as out:
    lines=file.readlines()
    pre=lines[0].replace("\n", "")
    del(lines[0])
    t=0
    for line in lines:
        word_list=[]
        for word in line.split():
            word_list.append(word)
        word_list = word_list[1:-1]
        output_line = pre
        index = random.randint(0,len(word_list)-1)
        index2 = random.randint(0,len(word_list)-1)
        while index == index2 and len(word_list)>1:
            index2 = random.randint(0,len(word_list)-1)
        if index > index2:
            index, index2 = index2, index
        if len(word_list) == 1:
            output_line = output_line + " " + str(index+1) + " " + word_list[index] + " ,, "
        else:
            if random.random()>=0.7:
                output_line = output_line + " " + str(index+1) + " " + word_list[index] + " ,, "
            else:
                output_line = output_line + " " + str(index+1) + " " + word_list[index]+ " " + str(index2+1) + " " + word_list[index2] + " ,, "
        output_line += line.replace("\n", "")
        # print(output_line)
        out.write(output_line+"\n")
        pre = line.replace("\n", "")
        # print(output_line)
        # t+=1

        # if t %10000 == 0:
        #     print(t)

