from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.models import model_from_json
import json
import numpy as np
import pdb
import os
import preprocess as pp
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed_dir', default='./seed', help='html files seed path')
parser.add_argument('--save_dir', default='./seed', help='path to store new html files')
parser.add_argument('--max_num_line', type=int, default=5, help='number of html tags to add by deepfuzz')
parser.add_argument('--num_seed', type=int, default=100, help='number of html files (seed) for deepfuzz')

FLAGS = parser.parse_args()


latent_dim = 512  # Latent dimensionality of the encoding space.
# num_samples = 2000000  # Number of samples to train on.
num_samples = 400000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'pair_len_150.txt'
# maxlen = 50
maxlen = 150
MAXLEN = maxlen

seed_path = FLAGS.seed_dir
save_dir = FLAGS.save_dir
max_num_line = FLAGS.max_num_line
num_seed = FLAGS.num_seed


# # Vectorize the data.
# input_texts = []
# target_texts = []
# input_characters = set()
# target_characters = set()
# # with open(data_path, 'r', encoding='iso-8859-1') as f:
# with open(data_path, 'r', encoding='utf-8') as f:
#     lines = f.read().split('\n')
# for line in lines[: min(num_samples, len(lines) - 1)]:
#     input_text, target_text = line.split('\t')
#     target_text = '\t' + target_text + '\n'
#     input_texts.append(input_text)
#     target_texts.append(target_text)
#     for char in input_text:
#         if char not in input_characters:
#             input_characters.add(char)
#     for char in target_text:
#         if char not in target_characters:
#             target_characters.add(char)

input_characters = [' ', '!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '}', '§', '°', '´', 'Ä', 'Ö', 'Ü', 'ä', 'ö', 'ü', '€']
target_characters = ['\t', '\n', ' ', '!', '"', '#', '$', '%', '&', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '}', '§', '°', '´', 'Ä', 'Ö', 'Ü', 'ä', 'ö', 'ü', '€']

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
NUM_ENCODER_TOKENS = num_encoder_tokens
# max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_encoder_seq_length = 150
# max_decoder_seq_length = max([len(txt) for txt in target_texts])
max_decoder_seq_length = 3

# print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

# print('input_characters:',input_characters)
# print('target_characters:',target_characters)


# encode the data into tensor
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
# print(input_token_index)
INPUT_TOKEN_INDEX = input_token_index
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])
# print(target_token_index )

# encoder_input_data = np.zeros(
#     (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
#     dtype=np.bool)
# decoder_input_data = np.zeros(
#     (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
#     dtype=np.bool)
# decoder_target_data = np.zeros(
#     (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
#     dtype=np.bool)

# for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
#     for t, char in enumerate(input_text):
#         encoder_input_data[i, t, input_token_index[char]] = 1.
#     for t, char in enumerate(target_text):
#         # decoder_target_data is ahead of decoder_input_data by one timestep
#         decoder_input_data[i, t, target_token_index[char]] = 1.
#         if t > 0:
#             # decoder_target_data will be ahead by one timestep
#             # and will not include the start character.
#             decoder_target_data[i, t - 1, target_token_index[char]] = 1.

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# load weights into new model
# model.load_weights("4model.h5")
# model.load_weights("1model.h5")
model.load_weights("2model.h5")
print('Model restroed!!!!!!')

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

def decode_sequence(input_seq, diversity):
    # Encode the input as state vectors.
    # tk = Tokenizer()
    # tk.fit_on_texts(input_seq)
    # index_list = tk.texts_to_sequences(input_seq)
    # input_seq = pad_sequences(index_list, maxlen=MAXLEN)

    encoder_input_data = np.zeros(
        (1, MAXLEN, NUM_ENCODER_TOKENS),
        dtype=np.bool)

    for t, char in enumerate(input_seq):
        # print(t)
        # print(char)
        encoder_input_data[0, t, INPUT_TOKEN_INDEX[char]] = 1.

    # states_value = encoder_model.predict(input_seq)
    states_value = encoder_model.predict(encoder_input_data)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token

        sampled_token_index = sample(output_tokens[0, -1, :], diversity)
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def synthesis(text, gmode='g1', smode='nosample'):
    length = len(text)
    random.seed()

    if (length < maxlen):
        return text

    if (gmode is 'g1'):
        print('gmode is g1')
        prefix_start = random.randrange(length - maxlen)
        prefix = text[prefix_start:prefix_start + maxlen]
        generated = prefix
        head = text[0 : prefix_start]
        tail = text[prefix_start + maxlen:]
        # cut_index = tail.find(';') + 1
        # tail = tail[cut_index:]
        # num_line = 0
        # k = 0
        cut_index_1 = tail.find('</') + 1
        tail = tail[cut_index_1:]
        cut_index_2 = tail.find('>') + 1
        head = text[0:prefix_start + cut_index_1 + cut_index_2 ]
        prefix = text[prefix_start + cut_index_1 + cut_index_2:prefix_start + cut_index_1 + cut_index_2+ maxlen]
        generated = prefix
        tail = tail[cut_index_2:]
        # prefix
        # tail = tail[cut_index_2:]
        num_line = 0
        k = 0
        end_pre_1 = 0
        end_pre_2 = 0
        # print('-' * 50)s
        while (num_line < max_num_line and k < 1000):
            print('add_line:',num_line)
            k = k + 1
            if (smode is 'nosample'):
                next_char = decode_sequence(prefix, 1)
            if (smode is 'sample'):
                next_char = decode_sequence(prefix, 1.2)
            if (smode is 'samplespace'):
                # if (generated[-1] == ' ' or generated[-1] == ';'):
                if (generated[-1] == ' ' or generated[-1] == '>'):
                    next_char = decode_sequence(prefix, 1.2)[0]
                    print('decode_sequence', next_char)
                else:
                    next_char = decode_sequence(prefix, 1)[0]
                    print('decode_sequence', next_char)
            # if (next_char == ';'):
            #     num_line += 1
            if (next_char == '<'):
                end_pre_1 = 1
            if (next_char == '/'):
                end_pre_2 = 1
            # if (next_char == ';'):
            if (end_pre_1 == 1 and end_pre_2 ==1 and next_char == '>'):
                num_line += 1
                end_pre_1 = 0
                end_pre_2 = 0
            generated += next_char
            # print('lenth of next char',len(next_char))
            # print('char_0',next_char[0])
            # print('char_1',next_char[1])
            # print('char_2',next_char[2])
            # print('char_3',next_char[3])
            prefix = prefix[1:] + next_char
            generated = pp.remove_space(generated)
            print(generated)
            prefix = pp.remove_space(prefix)
        text = head + generated + tail

    if (gmode is 'g2'):
        for i in range(2):
            length = len(text)
            prefix_start = random.randrange(length - maxlen)
            prefix = text[prefix_start:prefix_start + maxlen]
            generated = prefix
            head = text[0 : prefix_start]
            tail = text[prefix_start + maxlen:]
            # cut_index = tail.find(';') + 1
            cut_index_1 = tail.find('</') + 1
            tail = tail[cut_index_1:]
            cut_index_2 = tail.find('>') + 1
            tail = tail[cut_index_2:]
            num_line = 0
            k = 0
            end_pre_1 = 0
            end_pre_2 = 0
            while (num_line < max_num_line/2 and k < 150):
                k = k + 1
                if (smode is 'nosample'):
                    next_char = decode_sequence(prefix, 1)
                if (smode is 'sample'):
                    next_char = decode_sequence(prefix, 1.2)
                if (smode is 'samplespace'):
                    if (generated[-1] == ' ' or generated[-1] == ';'):
                        next_char = decode_sequence(prefix, 1.2)
                    else:
                        next_char = decode_sequence(prefix, 1)
                if (next_char == '<'):
                    end_pre_1 = 1
                if (next_char == '/'):
                    end_pre_2 = 1
                # if (next_char == ';'):
                if (end_pre_1 == 1 and end_pre_2 ==1 and next_char == '>'):
                    num_line += 1
                    end_pre_1 = 0
                    end_pre_2 = 0
                generated += next_char
                prefix = prefix[1:] + next_char
            text = head + generated + tail

    if (gmode is 'g3'):
        prefix_start = random.randrange(length - maxlen)
        prefix = text[prefix_start:prefix_start + maxlen]
        generated = prefix
        head = text[0 : prefix_start]
        tail = text[prefix_start + maxlen:]
        num_chop_line = 0
        while (num_chop_line < max_num_line):
            cut_index = tail.find(';') + 1
            tail = tail[cut_index:]
            num_chop_line += 1
        num_line = 0
        k = 0
        while (num_line < max_num_line and k < 150):
            k = k + 1
            if (smode is 'nosample'):
                next_char = decode_sequence(prefix, 1)
            if (smode is 'sample'):
                next_char = decode_sequence(prefix, 1.2)
            if (smode is 'samplespace'):
                if (generated[-1] == ' ' or generated[-1] == ';'):
                    next_char = decode_sequence(prefix, 1.2)
                else:
                    next_char = decode_sequence(prefix, 1)
            if (next_char == ';'):
                num_line += 1
            generated += next_char
            prefix = prefix[1:] + next_char
        text = head + generated + tail

    # print('-' * 50)
    # print('head: ')
    # print(head)
    print('generated: ')
    print(generated)
    # print('tail: ')
    # print(tail)
    return text
        
def generate():
    total_count = 0
    syntax_valid_count = 0
    files = []
    num = 1
    for root, d_names, f_names in os.walk(seed_path):
        for f in f_names:
            files.append(os.path.join(root, f))
    for file in files:
        if ('new' in file):
            continue
        # if (not file.endswith('.c') and not file.endswith('.h') and not file.endswith('.C')):
        if (not file.endswith('.html') and not file.endswith('.htm')):
            continue
        # try:
        #     # text = open(file, 'r').read()
        #     text = open(file, 'r',encoding='utf-8').read() 
        #     print(file)
        #     # text = pp.replace_macro(text, file)
        #     # text = pp.remove_comment(text)
        #     text = pp.remove_space(text)
        #     # is_valid = pp.verify_correctness(text, file, 'deepfuzz_original')
        #     # if (not is_valid):
        #     #     continue
        #     total_count += 1
        #     # text = synthesis(text, 'g1', 'nosample')
        #     # print(file)
        #     # assert 0==1
        #     text = synthesis(text, 'g1', 'samplespace')
        #     print(file)
        #     # is_valid = verify_correctness(text, file, 'deepfuzz_g1_nosample')
        #     # if (is_valid):
        #     syntax_valid_count += 1
        # text = open(file, 'r').read()
        try:
            text = open(file, 'r',encoding='utf-8').read() 
            print('='*50)
            print('processing file:',file)
            # text = pp.replace_macro(text, file)
            # text = pp.remove_comment(text)
            text = pp.remove_space(text)
            # is_valid = pp.verify_correctness(text, file, 'deepfuzz_original')
            # if (not is_valid):
            #     continue
            total_count += 1
            # text = synthesis(text, 'g1', 'nosample')
            # print(file)
            # assert 0==1
            text = synthesis(text, 'g1', 'samplespace')
            k=random.randint(0,100)
            file_generate = open(save_dir +'/'+ file.split('/')[-1][:-5] + '_new_' +str(k)+ '.html','w')
            print(save_dir +'/'+ file.split('/')[-1][:-5] + '_new_' +str(k)+ '.html')
            file_generate.write(text)
            file_generate.close()
                
                # is_valid = verify_correctness(text, file, 'deepfuzz_g1_nosample')
                # if (is_valid):
            syntax_valid_count += 1
        except:
            continue
        num += 1
        if num > num_seed:
            break 
    pass_rate = syntax_valid_count / total_count
    print('syntax_valid_count: %d' % (syntax_valid_count))
    print('total_count: %d' % (total_count))
    print('pass rate: %f' % pass_rate)

generate()
