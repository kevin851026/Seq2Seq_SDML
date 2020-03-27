import random

if __name__ == '__main__':
	train = []
	label = []
	with open('hw2.1_corpus.txt', 'r', encoding='utf-8') as f:
		lines = f.read().split('\n')
		lines.remove('')
		for l in range(len(lines[:-1])):
			if len(lines[l]) > 26 or len(lines[l+1]) > 26:
				continue
			tmpT = '<SOS>'
			for w in lines[l]:
				tmpT += ' ' + w
			tmpT += ' <EOS>'
			idx = random.randint(1, len(lines[l+1]))

			if random.random()>0.2 and len(lines[l]) <= 24:
				idx2 = random.randint(1, len(lines[l+1]))
				if idx2	!=	idx:
					if idx2 > idx:
						tmpT += ' ' + str(idx)
						tmpT += ' ' + lines[l+1][idx-1]
						tmpT += ' ' + str(idx2)
						tmpT += ' ' + lines[l+1][idx2-1]
					else:
						tmpT += ' ' + str(idx2)
						tmpT += ' ' + lines[l+1][idx2-1]
						tmpT += ' ' + str(idx)
						tmpT += ' ' + lines[l+1][idx-1]
				else:
					tmpT += ' ' + str(idx)
					tmpT += ' ' + lines[l+1][idx-1]
			else:
				tmpT += ' ' + str(idx)
				tmpT += ' ' + lines[l+1][idx-1]

			tmpL = '<SOS>'
			for w in lines[l+1]:
				tmpL += ' ' + w
			tmpL += ' <EOS>'
			train.append(tmpT)
			label.append(tmpL)

	with open('train4.txt', 'w', encoding='utf-8') as f:
		for l in train:
			f.write(l + '\n')

	with open('label4.txt', 'w', encoding='utf-8') as f:
		for l in label:
			f.write(l + '\n')