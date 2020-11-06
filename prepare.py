import re
import glob
from subprocess import Popen, PIPE, STDOUT
from os import walk
import os
import glob
import preprocess as pp
import pdb

def generate_training_data(text):
	maxlen = 150
	sentences = []
	next_chars = []
	# for i in range(0, len(text) - maxlen - 1):
	for i in range(0, len(text) - maxlen):
	    sentences.append(text[i: i + maxlen])
	    next_chars.append(text[i + maxlen])
	    sentences[i] = re.sub(r'[\n\t]',' ', sentences[i])		
	    next_chars[i] = re.sub(r'[\n\t]',' ', next_chars[i])
	    print(sentences[i] + "\t" + next_chars[i])
	    FILE_OUT.write(sentences[i] + "\t" + next_chars[i] + "\n")

# path = './gcc/gcc/testsuite'
# path = './html_seeds_learn'
path ='./training'
DATA_PATH = 'pair_len_150.txt'
FILE_OUT = open(os.path.join(DATA_PATH), 'w')


files = []
valid_count = 0
for root, d_names, f_names in os.walk(path):
	for f in f_names:
		files.append(os.path.join(root, f))

for file in files:
	if ('nocomment' in file or 'nospace' in file or 'nomacro' in file or 'raw' in file):
		command = 'rm ' + file
		p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)
	# if (not file.endswith('.c') and not file.endswith('.h') and not file.endswith('.C')):
	if (not file.endswith('.html') and not file.endswith('.htm')):
		continue
	# text = open(file, 'r', encoding='iso-8859-1').read()
	text = open(file, 'r', encoding='utf-8').read()
	# lines = text.split('\n')
	# for line in lines:
	# 	line = pp.remove_space(line)
	# text = pp.remove_comment(text)
	# text = pp.replace_macro(text, file)
	text = pp.remove_space(text)
	# is_valid = pp.verify_correctness(text, file, 'nospace')
	is_valid = 1
	if (is_valid):
		valid_count += 1
		# generate_training_data(line)	
		generate_training_data(text)
	print(valid_count)
