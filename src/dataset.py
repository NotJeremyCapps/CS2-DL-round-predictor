import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as P
from torch.utils.data import Subset
import pandas as pd
import csv
import math



class CS2PredictionDataset(Dataset):

    def calc_len(self):
        csv_count = 0
        self.total_len = 0
        with open(self.list, 'r') as file1: #open text file to list of csv files
            words = file1.read().strip().split()
            while(csv_count < len(words)):
                with open(words[csv_count], 'r') as file2: #open first csv file as file2
                    readingcsv = csv.reader(file2)
                    current_data = []
                    for row in readingcsv: 
                        current_data.append(row) #first index is row and then col
                    self.total_len = self.total_len + math.ceil((len(current_data) - 1)/self.sequence_length) #remove first row for column titles

                    #if((len(current_data)-1)%self.sequence_length != 0):
                    #    total_len += 1
                    csv_count =csv_count + 1
        print(self.total_len)

    def load_tensors(self):
        with open(self.list, 'r') as file1: #open text file to list of csv files
            words = file1.read().strip().split()
            self.total_csv_files = len(words)
            print(words)
            with open(words[self.csvfile], 'r') as file2: #open first csv file as file2 -- need to indicate based on order of csv file
                readingcsv = csv.reader(file2)
                self.all_data = []
                for row in readingcsv: 
                    self.all_data.append(row) #first index is row and then col

        #create target tensor  
        self.target_for_round= torch.tensor((float(self.all_data[1][-1])))

        #self.target_for_round= torch.full((self.sequence_length,), (float(self.all_data[1][-1])))

        #find prim/secondary index
        data_weap_ind = 0
        for index, word in enumerate(self.all_data[0]):
            if "primary" in word:
                data_weap_ind = index
                break
            elif "secondary" in word:
                data_weap_ind = index
                break

    
        #create non prim/second data into floats
        int_data = []
    
        for i in range(1, len(self.all_data)):#increment over every row
            temp_array = []
            for m in range(data_weap_ind): #up to prim/secondary data -THIS 96 IS INCORRECT SHOULD PROB automate
                    #convert to float
                    temp_array.append(float(self.all_data[i][m])) 
                   
            int_data.append(temp_array)
            
        #change into tensor
        self.data = torch.tensor(int_data) #data index 
        #create prim/secondary weapon data as floats
        prim_data_weap = []
        sec_data_weap = []
    
        for i in range(1, len(self.all_data)):#increment over every row
            prim_array = []
            sec_array = []
            for m in range(data_weap_ind,len(self.all_data[0])-1): #from prim/secondary data to win 
                    #convert to float
                    if(0 == (m-data_weap_ind)%2 ):
                        prim_array.append(float(self.all_data[i][m])) 
                    elif(1 == (m-data_weap_ind)%2):
                        sec_array.append(float(self.all_data[i][m]))
                   
            prim_data_weap.append(prim_array)
            sec_data_weap.append(sec_array)
            
        #change into tensor
        self.prim_data_weap_t = torch.tensor(prim_data_weap) #data index 
        self.sec_data_weap_t = torch.tensor(sec_data_weap) #data index 
      

        self.offset = 0 #offset from padding
        self.csvfile += 1


    def __init__(self, list, sequence_length): #a data point is one tick/timestamp, target is win/loss, data is parameter data, sequence_length eg 30 seconds 

        #sets amount of data points/ticks for each sequence
        self.sequence_length= sequence_length
        self.starting_index_of_round = 0


        self.csvfile = 0
        self.list = list
        #read in csv
        
        self.new_round = 1

        self.load_tensors()
        self.calc_len()





    


    def __getitem__(self, prov_index): #index is beginning index of sequence, this assumes all the data for rounds and games is sequential
        #determines tensor index of round for total provided index/offset/seqence length
        self.index_of_round = prov_index - self.starting_index_of_round
        tensor_index = self.index_of_round *self.sequence_length - self.offset
        len_of_round = len(self.data)

        excess= self.sequence_length - (len_of_round%self.sequence_length) # not 0 based
        if self.offset == 0: #tick needs to be padded if there is excess frames and 0 offset
            #indicate new round
            self.new_round = 1
            self.starting_index_of_round = prov_index
            if excess!= self.sequence_length:
                x_main_data = P.pad(self.data,(0,0,excess,0), value=0)
                x_prim_data_weap = P.pad(self.prim_data_weap_t,(0,0,excess,0), value=0) 
                x_sec_data_weap = P.pad(self.sec_data_weap_t,(0,0,excess,0), value=0) 

                self.offset = excess
                x_main_data = x_main_data[0:self.sequence_length]
                x_prim_data_weap = x_prim_data_weap[0:self.sequence_length]
                x_sec_data_weap = x_sec_data_weap[0:self.sequence_length]

            



        else: 
           x_main_data = self.data[tensor_index :tensor_index + self.sequence_length]
           x_prim_data_weap = self.prim_data_weap_t[tensor_index :tensor_index + self.sequence_length]
           x_sec_data_weap = self.sec_data_weap_t[tensor_index :tensor_index + self.sequence_length]
           self.new_round = 0

        '''
        if(prov_index <= 5):
           # with open('md_tensor_output' + str(prov_index) + '.txt', 'w') as f:
            #    torch.set_printoptions(threshold=torch.inf)
            #    print(x_main_data, file = f)
            with open('prim_tensor_output' + str(prov_index) + '.txt', 'w') as f:    
                torch.set_printoptions(threshold=torch.inf)
                print(x_prim_data_weap, file = f)
            with open('sec_tensor_output' + str(prov_index) + '.txt', 'w') as f:    
                torch.set_printoptions(threshold=torch.inf)
                print(x_sec_data_weap, file = f)
        '''

        #reload
        if(tensor_index + self.sequence_length ==  len(self.data) and self.csvfile != self.total_csv_files ):      
            self.load_tensors()

        #create tensor for new round
        #self.new_round= torch.full((self.sequence_length,), (float(self.new_round)))


        return self.target_for_round, self.new_round, x_main_data, x_prim_data_weap, x_sec_data_weap
    
    def __len__(self):
        return self.total_len 

def split(percent):
    demo_data_root="../game_demos"
    round_train_txt_file = os.path.join(demo_data_root, "preprocessed", f"rounds_train.txt")
    round_test_txt_file = os.path.join(demo_data_root, "preprocessed", f"rounds_test.txt")
    with open("../game_demos/preprocessed/rounds.txt", "r") as file1:
        lines = file1.readlines()

    total_games = 0
    current_game_name_total = ''
    for line in lines:
        #print(line)
        if current_game_name_total and current_game_name_total in line:
         
            continue
        else:
            total_games +=1
            current_game_name_total = line[0:-6]

    current_game_name = ''
    current_game_count = 0
    with open(round_train_txt_file, "a") as train_file, open(round_test_txt_file, "a") as test_file:

        for line in lines:
            if current_game_name and current_game_name in line:
                pass
            else:
                current_game_count += 1
                current_game_name = line[:-6]

            
            if((current_game_count/total_games) < percent):
                train_file.write(line)
            else:
                test_file.write(line)

        


