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
import string

import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--seed_dir', default='./seed', help='html files seed path')
# parser.add_argument('--save_dir', default='./seed', help='path to store new html files')
# parser.add_argument('--max_num_line', type=int, default=5, help='number of html tags to add by deepfuzz')
# parser.add_argument('--num_seed', type=int, default=100, help='number of html files (seed) for deepfuzz')
parser.add_argument('--seed_dir', default='./seed', help='html files seed path')

FLAGS = parser.parse_args()


latent_dim = 512  # Latent dimensionality of the encoding space.
data_path = 'pair_len_150.txt'
maxlen = 150
MAXLEN = maxlen

# seed_path = FLAGS.seed_dir
# save_dir = FLAGS.save_dir
# max_num_line = FLAGS.max_num_line
# num_seed = FLAGS.num_seed


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

print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


# encode the data into tensor
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
INPUT_TOKEN_INDEX = input_token_index
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

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
model.load_weights("4model_200.h5")
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

def synthesis(text, html_types, gmode='g1', smode='nosample'):
    length = len(text)
    random.seed()

    if (length < maxlen):
        return text

    if (gmode is 'g1'):
        print('gmode is g1')
        # start = text.find('<body')+1
        # prefix_start = random.randrange(start, length - maxlen)
        # prefix = text[prefix_start:prefix_start + maxlen]
        # generated = prefix
        # head = text[0 : prefix_start]
        # tail = text[prefix_start + maxlen:]
        # cut_index = tail.find(';') + 1
        # tail = tail[cut_index:]
        # num_line = 0
        # k = 0
        # cut_index_1 = tail.find('</') + 1
        # tail = tail[cut_index_1:]
        # cut_index_2 = tail.find('>') + 1
        # head = text[0:prefix_start + cut_index_1 + cut_index_2 ]
        # prefix = text[prefix_start + cut_index_1 + cut_index_2:prefix_start + cut_index_1 + cut_index_2+ maxlen]
        # generated = prefix
        # tail = tail[cut_index_2:]
        # prefix
        # tail = tail[cut_index_2:]
        prefix = text[-150:]
        generated = text[150:]
        num_line = 0
        k = 0
        end_pre_1 = 0
        end_pre_2 = 0
        stop = False
        # print('-' * 50)s
        # while (num_line < max_num_line and k < 1000):
        while(stop == False):
            # print('add_line:',num_line)
            # k = k + 1
            if (smode is 'nosample'):
                next_char = decode_sequence(prefix, 1)
            if (smode is 'sample'):
                next_char = decode_sequence(prefix, 1.2)
            if (smode is 'samplespace'):
                next_chars = ""
                next_char = generated[-1]
                for i in range(20):
                    if (next_char == ' ' or next_char == '\t' or next_char == '>'):
                        next_char = decode_sequence(prefix, 1.2)[0]
                        next_chars += next_char 
                        prefix = prefix[1:]+next_char
                    else:
                        next_char = decode_sequence(prefix, 1)[0]
                        next_chars += next_char 
                        prefix = prefix[1:]+next_char

                # if (generated[-1] == ' ' or generated[-1] == ';'):
                # if (generated[-1] == ' ' or generated[-1] == '>'):
                #     next_chars = ""
                #     for i in range(20):
                #         next_char = decode_sequence(prefix, 1)[0]
                #         # next_chars.join(next_char)
                #         next_chars += next_char 
                #         prefix = prefix[1:]+next_char
                #         # print('next_char ',next_char)
                #     print('decode_sequence',next_chars)
                # else:
                #     next_chars = ""
                #     for i in range(20):
                #         next_char = decode_sequence(prefix, 1)[0]
                #         # next_chars.join(next_char)
                #         next_chars += next_char 
                #         prefix = prefix[1:]+next_char
                #         # print('next_char',next_char)
                #     print('decode_sequence', next_chars)
            
            generated = generated + next_chars

            if ('</' in generated and '>' in generated[generated.find('</'):]):
                stop = True
                
            # generated += next_char !!!!!!!!!
            if stop == True:
                stop_1 = generated.find('</')+2
                generated = generated[:stop_1]+html_types+'>'
                # till_char = next_chars.find('/')+1
                # till_char += next_chars[till_char:].find('>')+1 
                # generated = generated + next_chars[:till_char]

            # prefix = prefix[1:] + next_char!!!!!!!!!!!
            generated = pp.remove_space(generated)
            # print(generated)
            prefix = pp.remove_space(prefix)
        # text = head + generated + tail

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
    # print('prefix: ')
    # print(prefix)
    # print('generated: ')
    # print(generated)
    # print('tail: ')
    # print(tail)
    return generated
        
def generate(previous_string_150 , html_types):

        # try:
        # text = open(file, 'r',encoding='utf-8').read() 
        # print('processing file:',file)

        # text = pp.replace_macro(text, file)
        # text = pp.remove_comment(text)
    text = previous_string_150 
    prefix = '<'+str(html_types) +' id ='
    text = text + prefix
    # text = pp.remove_space(text)
    # text = text[-150:]
    if len(text)<150:
        text = " "*(150-len(text))+text
    print('='*50)
    # is_valid = pp.verify_correctness(text, file, 'deepfuzz_original')
    # if (not is_valid):
    #     continue
    # total_count += 1
    # text = synthesis(text, 'g1', 'nosample')
    # print(file)
    # assert 0==1
    text = synthesis(text, html_types, 'g1', 'samplespace')

    # syntax_valid_count += 1
    # except:
    #     continue
    return text


if __name__ == "__main__":
    for j in range(10):
        random_prefix = ''
        for i in range(10):
            random_prefix+=''.join(random.sample(string.ascii_letters + string.digits, 15))
        gen = generate(random_prefix , 'img')
        print('generated tag:', gen)


