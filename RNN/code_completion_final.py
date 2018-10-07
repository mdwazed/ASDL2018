import tflearn
# from tflearn.data_utils import to_categorical, pad_sequences
import numpy
from random import randint


class Code_Completion_Final:
    # return token as string og type+value
    def token_to_string(self, token):
        return token["type"] + "-@@-" + token["value"]

    def string_to_token(self, string):
        splitted = string.split("-@@-")
        return {"type": splitted[0], "value": splitted[1]}

    # 0 pad to vector for equal length vector
    def zero_hot(self):
        vector = [0] * len(self.string_to_number)
        return vector
    # return one hot vector of token passed 
    # with a length of unique tokens vector
    def one_hot(self, string):
        vector = [0] * len(self.string_to_number)
        vector[self.string_to_number[string]] = 1
        return vector

    def prepare_data(self, token_lists):
        all_token_strings = set()
        for token_list in token_lists:
            for token in token_list:
                all_token_strings.add(self.token_to_string(token))
        all_token_strings = list(all_token_strings)
        all_token_strings.sort()
        print("Unique tokens: " + str(len(all_token_strings)))
        # sys.exit()
        self.string_to_number = dict()
        self.number_to_string = dict()
        max_number = 0
        for token_string in all_token_strings:
            self.string_to_number[token_string] = max_number
            self.number_to_string[max_number] = token_string
            max_number += 1

        # x, y pairs to feed to the network, length of x is 6
        # first creating 1 hole and then creating 2 holes
        # x= suffix[1]+suffix[0]+prefix[0]+prefix[1]+prefix[2]+prefix[3]
        # y=token
        xs = []
        ys = []
        for token_list in token_lists:
            length = len(token_list)
            #  take 100 samples from each file
            for i in range(100):
                idx = randint(0, length-1)
                token = token_list[idx]
                # pad 0 to all prefix if prefix length is 0
                if idx == 0:
                    token_string = self.token_to_string(token)
                    prefix_4 = self.zero_hot()
                    prefix_3 = self.zero_hot()
                    prefix_2 = self.zero_hot()
                    prefix_1 = self.zero_hot()

                # pad 0 to previous 3 position if pefix length is 1
                if idx == 1:
                    token_string = self.token_to_string(token)
                    prefix_4 = self.zero_hot()
                    prefix_3 = self.zero_hot()
                    prefix_2 = self.zero_hot()
                    prefix_1 = self.one_hot(self.token_to_string(token_list[idx - 1]))

                # If prefix length is 2 then consider previous 2 tokens and pad 2 zeros      
                if idx == 2:
                    token_string = self.token_to_string(token)
                    prefix_4 = self.zero_hot()
                    prefix_3 = self.zero_hot()
                    prefix_2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                    prefix_1 = self.one_hot(self.token_to_string(token_list[idx - 1]))

                # If prefix length is 3 then consider previous 3 tokens and pad 1 zeros 
                if idx == 3:
                    token_string = self.token_to_string(token)
                    prefix_4 = self.zero_hot()
                    prefix_3 = self.one_hot(self.token_to_string(token_list[idx - 3]))
                    prefix_2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                    prefix_1 = self.one_hot(self.token_to_string(token_list[idx - 1]))

                # add previous 4 tokens if prefix length is 4 or more 
                if idx > 3:
                    token_string = self.token_to_string(token)
                    prefix_4 = self.one_hot(self.token_to_string(token_list[idx - 4]))
                    prefix_3 = self.one_hot(self.token_to_string(token_list[idx - 3]))
                    prefix_2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                    prefix_1 = self.one_hot(self.token_to_string(token_list[idx - 1]))

           
                #  add 2 suffix if suffix length is 2 or more
                if(length - idx) > 2:
                    suffix_1 = self.one_hot(self.token_to_string(token_list[idx+1]))
                    suffix_2 = self.one_hot(self.token_to_string(token_list[idx+2]))
             
                #  add 1 suffix, and one 0 pad  if suffix length is 1
                if(length - idx) == 2:
                    suffix_1 = self.one_hot(self.token_to_string(token_list[idx+1]))
                    suffix_2 = self.zero_hot()
             
                # pad two 0 if suffix length is o
                if(length - idx) == 1:
                    suffix_1 = self.zero_hot()
                    suffix_2 = self.zero_hot()
                              
                # add 2 suffix and 4 prefix to xs
                xs.append([suffix_2, suffix_1, prefix_4, prefix_3, prefix_2,
                           prefix_1])
                ys.append(self.one_hot(token_string))

                # repeat the previous step but creating hole of 2 missing tokens
                if idx == 0:
                    token_string = self.token_to_string(token)
                    prefix_4 = self.zero_hot()
                    prefix_3 = self.zero_hot()
                    prefix_2 = self.zero_hot()
                    prefix_1 = self.zero_hot()

              
                if idx == 1:
                    token_string = self.token_to_string(token)
                    prefix_4 = self.zero_hot()
                    prefix_3 = self.zero_hot()
                    prefix_2 = self.zero_hot()
                    prefix_1 = self.one_hot(self.token_to_string(token_list[idx - 1]))

          
                if idx == 2:
                    token_string = self.token_to_string(token)
                    prefix_4 = self.zero_hot()
                    prefix_3 = self.zero_hot()
                    prefix_2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                    prefix_1 = self.one_hot(self.token_to_string(token_list[idx - 1]))

              
                if idx == 3:
                    token_string = self.token_to_string(token)
                    prefix_4 = self.zero_hot()
                    prefix_3 = self.one_hot(self.token_to_string(token_list[idx - 3]))
                    prefix_2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                    prefix_1 = self.one_hot(self.token_to_string(token_list[idx - 1]))

                if idx > 3:
                    token_string = self.token_to_string(token)
                    prefix_4 = self.one_hot(self.token_to_string(token_list[idx - 4]))
                    prefix_3 = self.one_hot(self.token_to_string(token_list[idx - 3]))
                    prefix_2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                    prefix_1 = self.one_hot(self.token_to_string(token_list[idx - 1]))

     
                if(length - idx) > 3:
                    suffix_1 = self.one_hot(self.token_to_string(token_list[idx+2]))
                    suffix_2 = self.one_hot(self.token_to_string(token_list[idx+3]))
                  

                if(length - idx) == 3:
                    suffix_1 = self.one_hot(self.token_to_string(token_list[idx+2]))
                    suffix_2 = self.zero_hot()
               

                if(length - idx) <= 2:
                    suffix_1 = self.zero_hot()
                    suffix_2 = self.zero_hot()
 

                xs.append([suffix_2, suffix_1, prefix_4, prefix_3, prefix_2,
                           prefix_1])
                ys.append(self.one_hot(token_string))

                # repeat the previous step but creating hole of 3 missing tokens
                if idx == 0:
                    token_string = self.token_to_string(token)
                    prefix_4 = self.zero_hot()
                    prefix_3 = self.zero_hot()
                    prefix_2 = self.zero_hot()
                    prefix_1 = self.zero_hot()

              
                if idx == 1:
                    token_string = self.token_to_string(token)
                    prefix_4 = self.zero_hot()
                    prefix_3 = self.zero_hot()
                    prefix_2 = self.zero_hot()
                    prefix_1 = self.one_hot(self.token_to_string(token_list[idx - 1]))

          
                if idx == 2:
                    token_string = self.token_to_string(token)
                    prefix_4 = self.zero_hot()
                    prefix_3 = self.zero_hot()
                    prefix_2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                    prefix_1 = self.one_hot(self.token_to_string(token_list[idx - 1]))

              
                if idx == 3:
                    token_string = self.token_to_string(token)
                    prefix_4 = self.zero_hot()
                    prefix_3 = self.one_hot(self.token_to_string(token_list[idx - 3]))
                    prefix_2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                    prefix_1 = self.one_hot(self.token_to_string(token_list[idx - 1]))

                if idx > 3:
                    token_string = self.token_to_string(token)
                    prefix_4 = self.one_hot(self.token_to_string(token_list[idx - 4]))
                    prefix_3 = self.one_hot(self.token_to_string(token_list[idx - 3]))
                    prefix_2 = self.one_hot(self.token_to_string(token_list[idx - 2]))
                    prefix_1 = self.one_hot(self.token_to_string(token_list[idx - 1]))

     
                if(length - idx) >= 6:
                    suffix_1 = self.one_hot(self.token_to_string(token_list[idx+4]))
                    suffix_2 = self.one_hot(self.token_to_string(token_list[idx+5]))
                  

                if(length - idx) == 5:
                    suffix_1 = self.one_hot(self.token_to_string(token_list[idx+4]))
                    suffix_2 = self.zero_hot()
               

                if(length - idx) <= 4:
                    suffix_1 = self.zero_hot()
                    suffix_2 = self.zero_hot()
 

                xs.append([suffix_2, suffix_1, prefix_4, prefix_3, prefix_2,
                           prefix_1])
                ys.append(self.one_hot(token_string))


        print("x,y pairs: " + str(len(xs)))
        return (xs, ys)

    #  create the LSTM network 
    def create_network(self):
        self.net = tflearn.input_data(shape=[None, 6, len(self.string_to_number)])
        self.net = tflearn.lstm(self.net, 128, return_seq=True)
        self.net = tflearn.lstm(self.net, 128, return_seq=True)
        self.net = tflearn.lstm(self.net, 128)
        self.net = tflearn.fully_connected(self.net, len(self.string_to_number), activation='softmax', bias=True,
                                           trainable=True)
        self.net = tflearn.regression(self.net, optimizer='adam', loss='categorical_crossentropy')
        self.model = tflearn.DNN(self.net)

    #  Load stored model for test and predict
    def load(self, token_lists, model_file):
        self.prepare_data(token_lists)
        self.create_network()
        self.model.load(model_file)

    def train(self, token_lists, model_file):
        (xs, ys) = self.prepare_data(token_lists)
        self.create_network()
        self.model.fit(xs, ys, n_epoch=1, batch_size=50, show_metric=True)
        self.model.save(model_file)

    #  predict y feeding x, seq of tokens
    def predict(self, x):
        y = self.model.predict([x])
        predicted_seq = y[0]
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist()
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.number_to_string[best_number]
        best_token = self.string_to_token(best_string)
        return best_token

    def query(self, prefix, suffix):

        # Padding all with 0 if prefix length is 0
        if len(prefix) == 0:
            prefix_4 = self.zero_hot()
            prefix_3 = self.zero_hot()
            prefix_2 = self.zero_hot()
            prefix_1 = self.zero_hot()

        # Use the available prefix tokens and pad the rest with zero vectors
        elif len(prefix) == 1:            
            prefix_4 = self.zero_hot()
            prefix_3 = self.zero_hot()
            prefix_2 = self.zero_hot()
            prefix_1 = self.one_hot(self.token_to_string(prefix[-1]))

        # Use the available prefix tokens and padd the rest with zero vectors
        elif len(prefix) == 2:            
            prefix_4 = self.zero_hot()
            prefix_3 = self.zero_hot()
            prefix_2 = self.one_hot(self.token_to_string(prefix[-2]))
            prefix_1 = self.one_hot(self.token_to_string(prefix[-1]))

        # Use the available prefix tokens and padd the rest with zero vector
        elif len(prefix) == 3:            
            prefix_4 = self.zero_hot()
            prefix_3 = self.one_hot(self.token_to_string(prefix[-3]))
            prefix_2 = self.one_hot(self.token_to_string(prefix[-2]))
            prefix_1 = self.one_hot(self.token_to_string(prefix[-1]))

        #  use closest 4 prefix if prefix length is 4 or more
        else:           
            prefix_4 = self.one_hot(self.token_to_string(prefix[-4]))
            prefix_3 = self.one_hot(self.token_to_string(prefix[-3]))
            prefix_2 = self.one_hot(self.token_to_string(prefix[-2]))
            prefix_1 = self.one_hot(self.token_to_string(prefix[-1]))


        #  add suffix according to lenght of suffix or pad 0
        if len(suffix) == 0:
            suffix_1 = self.zero_hot()
            suffix_2 = self.zero_hot()

        elif len(suffix) == 1:
            suffix_1 = self.one_hot(self.token_to_string(suffix[0]))
            suffix_2 = self.zero_hot()

        elif len(suffix) >= 2:
            suffix_1 = self.one_hot(self.token_to_string(suffix[0]))
            suffix_2 = self.one_hot(self.token_to_string(suffix[1]))
       
        # array list for storing all the predicted tokens
        predicted = []
        x = [suffix_2, suffix_1, prefix_4, prefix_3, prefix_2, prefix_1]
        best_token = self.predict(x)
        predicted.append(best_token)
        

        iter = 0
        while True:
            ## Each iter add predicted token to the end of prefix and try to match up with first token of suffix 
            prefix_4 = prefix_3
            prefix_3 = prefix_2
            prefix_2 = prefix_1
            prefix_1 = self.one_hot(self.token_to_string(best_token))

            x = [suffix_2, suffix_1, prefix_4, prefix_3, prefix_2, prefix_1]
            best_token = self.predict(x)

			# look for 1st token of suffix or boundary condition
            if (len(suffix) != 0 and (best_token['value'] == suffix[0]['value'] and best_token['type'] == suffix[0][
                'type']) or iter == 4):
                # If iter is 4 actual missing token is not found thus return first prediction             
                if (iter >= 3):
                    return [predicted[0]]
                else:
					#return all the predicted tokens
                    return predicted
            else:
                predicted.append(best_token)

            iter = iter + 1

########################################################
