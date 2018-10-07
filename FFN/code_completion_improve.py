import numpy as np
import tflearn
# from tflearn.data_utils import to_categorical, pad_sequences
# import sys
from random import randint


class Code_Completion_Improve:
    def __init__(self):
        # prefix, suffix and total length of sequence
        self.prefix_length = 5
        self.suffix_length = 5
        self.total_length = self.prefix_length + self.suffix_length
        self.num_unique_tokens = 0
        self.input_size = 0

    def token_to_string(self, token):
        return token["type"] + "-@@-" + token["value"]

    def tokens_to_strings(self, tokens):
        return [self.token_to_string(token) for token in tokens]

    def string_to_token(self, string):
        splitted = string.split("-@@-")
        return {"type": splitted[0], "value": splitted[1]}

    # conver token to onehot vector of length of unique tokens
    def one_hot(self, string):
        vector = [0] * len(self.string_to_number)
        vector[self.string_to_number[string]] = 1
        return vector

    #  convert sequence of tokens to n_hot vector, suitable to feed to the network
    def n_hot(self, prev_list, next_list):
        input_vector = np.zeros((self.input_size,), dtype=np.int8)

        for i, string in enumerate(prev_list):
            target_idx = (i*self.num_unique_tokens) + self.string_to_number[string]
            input_vector[target_idx] = 1

        for i, string in enumerate(next_list):
            target_idx = (i * self.num_unique_tokens) + self.string_to_number[string]
            # target_idx = self.string_to_number[string]
            input_vector[target_idx] = 1

        return input_vector

    def prepare_data(self, token_lists):
        all_token_strings = set()
        for token_list in token_lists:
            for token in token_list:
                all_token_strings.add(self.token_to_string(token))
        all_token_strings = list(all_token_strings)
        all_token_strings.sort()
        print("Unique tokens: " + str(len(all_token_strings)))
        self.num_unique_tokens = len(all_token_strings)
        self.string_to_number = dict()
        self.number_to_string = dict()
        max_number = 0
        for token_string in all_token_strings:
            self.string_to_number[token_string] = max_number
            self.number_to_string[max_number] = token_string
            max_number += 1

        self.input_size = self.num_unique_tokens * self.total_length # unique token X consider total

        # prepare x & y pairs
        xs = []
        ys = []

        for token_list in token_lists:
            length = len(token_list)
            # 600 samples from each file
            for i in range(600):
                idx = randint(0, length-1)
                token = token_list[idx]        
                token_string = self.token_to_string(token)

                # create one hole
                prefix_min_idx = max([0, idx - self.prefix_length])
                suffix_max_idx = min([len(token_list) - 1, idx + self.suffix_length])

                prev_token_strings = self.tokens_to_strings(token_list[prefix_min_idx:idx])
                next_token_strings = self.tokens_to_strings(token_list[(idx + 1):(suffix_max_idx + 1)])

                xs.append(self.n_hot(prev_token_strings, next_token_strings))
                ys.append(self.one_hot(token_string))    
               
                # create 2 holes
                prefix_min_idx = max([0, idx - self.prefix_length])
                suffix_max_idx = min([len(token_list) - 1, idx + self.suffix_length])

                prev_token_strings = self.tokens_to_strings(token_list[prefix_min_idx:idx])
                next_token_strings = self.tokens_to_strings(token_list[(idx + 2):(suffix_max_idx + 2)])

                xs.append(self.n_hot(prev_token_strings, next_token_strings))
                ys.append(self.one_hot(token_string))           

        print("x,y pairs: " + str(len(xs)))     
        return xs, ys


    def create_network(self):
        self.net = tflearn.input_data(shape=[None, self.input_size])
      
        self.net = tflearn.fully_connected(self.net, 250, activation='relu')
        self.net = tflearn.fully_connected(self.net, 250, activation='relu')
        self.net = tflearn.fully_connected(self.net, 250, activation='relu')
        self.net = tflearn.fully_connected(self.net, len(self.string_to_number), activation='softmax')
        self.net = tflearn.regression(self.net, learning_rate = 0.01)
        self.model = tflearn.DNN(self.net)

    def load(self, token_lists, model_file):
        self.prepare_data(token_lists)
        self.create_network()
        self.model.load(model_file)

    def train(self, token_lists, model_file):
        (xs, ys) = self.prepare_data(token_lists)     
        self.create_network()
        self.model.fit(xs, ys, n_epoch=1, batch_size=2000, show_metric=True)
        self.model.save(model_file)

    # if suffix not found return single token else return tokens   
    def query(self, prefix, suffix):
        predict = []
        num_suffixes = min([len(suffix), self.suffix_length])
        num_prefixes = min([len(prefix), self.prefix_length])

        prev_token_strings = self.tokens_to_strings(prefix[-num_prefixes:])
        next_token_strings = self.tokens_to_strings(suffix[:num_suffixes])
        
        x = self.n_hot(prev_token_strings, next_token_strings)     
        y = self.model.predict([x])
        best_number = np.argmax(y)
        best_string = self.number_to_string[best_number]
        best_token = self.string_to_token(best_string)
        predict.append(best_token)

        ########
        if(len(next_token_strings)>0):
            prev_token_strings = prev_token_strings[1:]
            prev_token_strings.append(best_string)
            x = self.n_hot(prev_token_strings, next_token_strings)    
            y = self.model.predict([x])
            best_number = np.argmax(y)
            best_string = self.number_to_string[best_number]
            best_token = self.string_to_token(best_string)
            if(best_string == str(next_token_strings[0])):
                return predict
            else:
                predict.append(best_token)
                prev_token_strings = prev_token_strings[1:]
                prev_token_strings.append(best_string)
                x = self.n_hot(prev_token_strings, next_token_strings)    
                y = self.model.predict([x])
                best_number = np.argmax(y)
                best_string = self.number_to_string[best_number]
                best_token = self.string_to_token(best_string)
                if(best_string == str(next_token_strings[0])):
                    return predict
        return predict[:1]

