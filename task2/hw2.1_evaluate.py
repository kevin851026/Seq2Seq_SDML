import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--training_file', type=str, required=True)
parser.add_argument('--result_file', type=str, required=True)
args = parser.parse_args()

all_assign_cnt = 0
correct_cnt = 0
with open(args.training_file, 'r') as f:
    train_data = f.read().split('\n')[:-1]
with open(args.result_file, 'r') as f:
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
