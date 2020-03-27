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
with open('task2/hw2.1_corpus.txt', newline='', encoding='utf-8') as file,\
        open('task2/train3.txt','w', newline='', encoding='utf-8') as out:
    lines=file.readlines()
    pre=lines[0].replace("\n", "")
    del(lines[0])
    t=0
    for line in lines:
        if len(line) > 100:
            continue
        line=line.replace("\n", "")
        output_line = '<SOS> '
        index = random.randint(0,len(line)-1) % 30
        index2 = random.randint(0,len(line)-1) % 30
        while index == index2 and len(line)>1:
            index2 = random.randint(0,len(line)-1) % 30
        if index > index2:
            index, index2 = index2, index
        target = line[index]
        target2 = line[index2]
        for word in pre:
            output_line=output_line + word + " "
        if len(line) == 1:
            output_line = output_line +'<EOS> '+ str(index+1) + " " + target+ ","
        else:
            if random.random()>=0.7:
                output_line = output_line +'<EOS> '+ str(index+1) + " " + target + ","
            else:
                output_line = output_line +'<EOS> '+ str(index+1) + " " + target + " " + str(index2+1) + " " + target2 + ","
        output_line+='<SOS> '
        for word in line:
            output_line=output_line + word + " "
        output_line+='<EOS>'
        pre = line
        out.write(output_line+'\n')
        # print(output_line)
        t+=1

        if t %10000 == 0:
            print(t)

