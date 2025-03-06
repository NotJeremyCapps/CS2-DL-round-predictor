import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as P
from torch.utils.data import Subset
import pandas as pd
import csv
import math
import re
import itertools


class CS2PredictionDataset(Dataset):

    def __init__(self, list_path, sequence_length): #a data point is one tick/timestamp, target is win/loss, data is parameter data, sequence_length eg 30 seconds 

        self.overlap = 2 #eg 2 means that half of the sequences will overlap

        #sets amount of data points/ticks for each sequence
        self.sequence_length= sequence_length
        self.starting_index_of_round = 0


        self.csvfile = 0
        self.list = list_path

        self.curr_csv_pth = None
        self.plyr_11_skips = 0
        #read in csv
        

        #self.load_tensors()
        self.calc_len()

    def calc_len(self):
        csv_count = 0
        self.total_len = 0
        with open(self.list, 'r') as file1: #open text file to list of csv files
            words = file1.read().strip().split()
            while(csv_count < len(words)):
                with open(os.path.join("../..", words[self.csvfile]), 'r') as file2: #open first csv file as file2
                    readingcsv = csv.reader(file2)
                    current_data = []
                    for row in readingcsv: 
                        current_data.append(row) #first index is row and then col
                    #self.total_len = self.total_len + (math.ceil((len(current_data) - 1)/self.sequence_length))
                    self.total_len = self.total_len + (math.ceil((len(current_data) - 1)/self.sequence_length)*self.overlap -2) #remove first row for column titles
                    #*2-2 to account for the overalp of 50% except for the first index
                    #ceiling accounts fo padded data


                    #if((len(current_data)-1)%self.sequence_length != 0):
                    #    total_len += 1
                    csv_count =csv_count + 1
        print(self.total_len)

    def load_tensors(self):
        with open(self.list, 'r') as file1: #open text file to list of csv files
            words = file1.read().strip().split()
            self.total_csv_files = len(words)
            # print(words)

            valid_round = False

            while valid_round == False:
                with open(os.path.join("../..", words[self.csvfile]), 'r') as file2: #open first csv file as file2 -- need to indicate based on order of csv file
                    readingcsv = csv.reader(file2)
                    self.all_data = []
                    for row in readingcsv: 
                        self.all_data.append(row) #first index is row and then col


                ###can remove this if clean up works
                    #Check for 11th player and skip index if found
                    if len(self.all_data[0]) > 167: 

                        print(f"Player 11 found. Num Instances: {self.plyr_11_skips}, Row Len: {len(self.all_data[0])}")

                        with open(f"getitem_maindata.txt", "a") as f: 
                            f.write(f"[SKIPPED] - {len(self.all_data[0])} from file: {words[self.csvfile]}\n\n")

                        self.plyr_11_skips += 1
                        self.csvfile += 1 # Go to next csvfile
                    elif (len(self.all_data[2]) != 167):
                        print(f"Length is incorrect. Skippin CSV File: {words[self.csvfile]}")
                        self.csvfile += 1 # Go to next csvfile
             
                    else: 
                        '''for i in range(1,len(self.all_data)):
                            if (len(self.all_data[i]) != 167):
                                print(f"Length is incorrect. Skippin CSV File: {words[self.csvfile]}. Row: {i}")
                                self.csvfile += 1 # Go to next csvfile
                        '''
                        valid_round = True
                        self.curr_csv_pth = words[self.csvfile]
                        break


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
                    try:
                        temp_array.append(float(self.all_data[i][m])) 
                    except:
                        print(f"Error: {self.csv_file} Row/Column: {i} {m}")
                   
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
        
        # with open('batch_data.txt', 'a') as f:
        #     print(f"Loading tensors!! All Data: {self.all_data}, Shape: {len(self.all_data)};\n\n\n", file = f)
            
        #change into tensor
        self.prim_data_weap_t = torch.tensor(prim_data_weap) #data index 
        self.sec_data_weap_t = torch.tensor(sec_data_weap) #data index 
      

        self.offset = 0 #offset from padding
        self.csvfile += 1
        self.new_round = 2
        self.overlap_offset = 0



    def __getitem__(self, prov_index): #index is beginning index of sequence, this assumes all the data for rounds and games is sequential
        #reset and load for first index
        
        # add total skips to index that comes in to maintain indexing

        if prov_index == 0:
            self.starting_index_of_round = 0
            self.csvfile = 0
            self.load_tensors()
            self.overlap_offset = 0

        
        #determines tensor index of round for total provided index/offset/seqence length
        self.index_of_round = prov_index - self.starting_index_of_round #starting index is the provided index for a new round
        tensor_index = self.index_of_round *self.sequence_length - self.offset -self.overlap_offset

        

        len_of_round = len(self.data)

        excess= self.sequence_length - (len_of_round%self.sequence_length) # not 0 based
        if self.new_round == 2: #check for padding if in a new round
            self.new_round -= 1  #decrement to indicate next round is not a new round but current one is 
            self.starting_index_of_round = prov_index
            if excess!= self.sequence_length:
                x_main_data = P.pad(self.data,(0,0,excess,0), value=0) # Left padding
                x_prim_data_weap = P.pad(self.prim_data_weap_t,(0,0,excess,0), value=0) 
                x_sec_data_weap = P.pad(self.sec_data_weap_t,(0,0,excess,0), value=0)

                # x_main_data = P.pad(self.data,(0,0,0,excess), value=0) # Right padding 
                # x_prim_data_weap = P.pad(self.prim_data_weap_t,(0,0,0,excess), value=0) 
                # x_sec_data_weap = P.pad(self.sec_data_weap_t,(0,0,0,excess), value=0)  

                self.offset = excess
                x_main_data = x_main_data[0:self.sequence_length]
                x_prim_data_weap = x_prim_data_weap[0:self.sequence_length]
                x_sec_data_weap = x_sec_data_weap[0:self.sequence_length]
            else:
                x_main_data = self.data[0:self.sequence_length]
                x_prim_data_weap = self.prim_data_weap_t[0 : self.sequence_length]
                x_sec_data_weap = self.sec_data_weap_t[0:self.sequence_length]


        else: 
           x_main_data = self.data[tensor_index :tensor_index + self.sequence_length]
           x_prim_data_weap = self.prim_data_weap_t[tensor_index :tensor_index + self.sequence_length]
           x_sec_data_weap = self.sec_data_weap_t[tensor_index :tensor_index + self.sequence_length]
           self.new_round = 0
           self.overlap_offset = round(self.sequence_length/self.overlap) + self.overlap_offset
            
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
        if(tensor_index + self.sequence_length== len(self.data) and self.csvfile != self.total_csv_files ):      
            self.load_tensors()

        #create tensor for new round
        #self.new_round= torch.full((self.sequence_length,), (float(self.new_round)))

        if(self.new_round == 2):
            new_round_output = 0
        
        else:
            new_round_output = self.new_round

        with open(f"getitem_maindata.txt", "a") as f: 
            f.write(f"{x_main_data.size()} from file: {self.curr_csv_pth}\n\n")

        return self.target_for_round, new_round_output, x_main_data, x_prim_data_weap, x_sec_data_weap
    
    def __len__(self):
        return self.total_len 
'''
def cleanup_data(demo_root):
    round_train_txt_preclean_file = os.path.join(demo_root, f"rounds_train_preclean.txt")
    round_test_txt_preclean_file = os.path.join(demo_root, f"rounds_test_preclean.txt")   

    round_train_txt_file = os.path.join(demo_root, f"rounds_train.txt")
    round_test_txt_file = os.path.join(demo_root, f"rounds_test.txt")  


    #Training File Cleaning
    with open(round_train_txt_preclean_file, "r") as train_file_pre:
        csv_files_1 = train_file_pre.readlines()

    print("Cleaning Up Training Data:\n")

    with open(round_train_txt_file, "w") as round_train_clean:
        for i in range(len(csv_files_1)):
            #Check for 11th player and skip index if found
            with open( os.path.join("../..", csv_files_1[i].strip()), "r") as current_csv_file:
                #make array for csv
                readingcsv = csv.reader(current_csv_file)
                current_csv_file_arr = []
                for row in itertools.islice(readingcsv, 3): 
                    current_csv_file_arr.append(row) #first index is row and then col

                #complted checks
                if len(current_csv_file_arr[0]) > 167: 
                    print(f"Player 11 found. Row Len: {len(current_csv_file_arr[0])}.  SkippinG CSV File: {csv_files_1[i]}")

                elif len(current_csv_file_arr[0]) < 167: 
                    print(f"Less than 10 Players Found. Row Len: {len(current_csv_file_arr[0])}.  SkippinG CSV File: {csv_files_1[i]}")

                elif (len(current_csv_file_arr[2]) != 167):
                    print(f"Length is incorrect. SkippinG CSV File: {csv_files_1[i]}")
                    
                else:
                    round_train_clean.write(csv_files_1[i])

    #Testing File Cleaning
    with open(round_test_txt_preclean_file, "r") as test_file_pre:
        csv_files_2 = test_file_pre.readlines()

    print("Cleaning Up Testing Data:\n")

    with open(round_test_txt_file, "w") as round_test_clean:
        for i in range(len(csv_files_2)):
            
            #Check for 11th player and skip index if found
            with open(os.path.join("../..", csv_files_2[i].strip()), "r") as current_csv_file:
                readingcsv = csv.reader(current_csv_file)
                current_csv_file_arr = []
                for row in itertools.islice(readingcsv, 3):
                    current_csv_file_arr.append(row) #first index is row and then col
                if len(current_csv_file_arr[0]) > 167: 
                    print(f"Player 11 found. Row Len: {len(current_csv_file_arr[0])}.  SkippinG CSV File: {csv_files_2[i]}")

                elif len(current_csv_file_arr[0]) < 167: 
                    print(f"Less than 10 Players Found. Row Len: {len(current_csv_file_arr[0])}.  SkippinG CSV File: {csv_files_2[i]}")

                elif (len(current_csv_file_arr[2]) != 167):
                    print(f"Length is incorrect. SkippinG CSV File: {csv_files_2[i]}")
                else:
                    round_test_clean.write(csv_files_2[i])
'''


    

def cleanup_data_pre(demo_root):
    round_txt_preclean_file = os.path.join(demo_root, f"rounds.txt")

    round_txt_postclean_file = os.path.join(demo_root, f"rounds_clean.txt")


    #Training File Cleaning
    with open(round_txt_preclean_file, "r") as train_file_pre:
        csv_files_1 = train_file_pre.readlines()

    print("Cleaning Up Data:\n")

    with open(round_txt_postclean_file, "w") as round_clean:
        for i in range(len(csv_files_1)):
            #Check for 11th player and skip index if found
            with open( os.path.join("../..", csv_files_1[i].strip()), "r") as current_csv_file:
                #make array for csv
                readingcsv = csv.reader(current_csv_file)
                current_csv_file_arr = []
                for row in itertools.islice(readingcsv, 3): 
                    current_csv_file_arr.append(row) #first index is row and then col

                #complted checks
                if len(current_csv_file_arr[0]) > 167: 
                    print(f"Player 11 found. Row Len: {len(current_csv_file_arr[0])}.  SkippinG CSV File: {csv_files_1[i]}")

                elif len(current_csv_file_arr[0]) < 167: 
                    print(f"Less than 10 Players Found. Row Len: {len(current_csv_file_arr[0])}.  SkippinG CSV File: {csv_files_1[i]}")

                elif (len(current_csv_file_arr[2]) != 167):
                    print(f"Length is incorrect. SkippinG CSV File: {csv_files_1[i]}")
                    
                else:
                    round_clean.write(csv_files_1[i])


    
   




def split_dataset(percent, demo_root="../game_demos/preprocessed"):
    # demo_data_root="../game_demos"
    # round_train_txt_file = os.path.join(demo_data_root, "preprocessed", f"rounds_train.txt")
    # round_test_txt_file = os.path.join(demo_data_root, "preprocessed", f"rounds_test.txt")
    cleanup_data_pre(demo_root)

    round_train_txt_file = os.path.join(demo_root, f"rounds_train.txt")
    round_test_txt_file = os.path.join(demo_root, f"rounds_test.txt")

    with open(os.path.join(demo_root, "rounds_clean.txt"), "r") as file1: 
    # with open(os.path.join(demo_root, "rounds_147len.txt"), "r") as file1:
        lines = file1.readlines()

    total_games = 0
    current_game_name_total = ''
    # Make sure we dont split data from same game
    split_idx = round(percent * len(lines))

    print(f"Number of rounds: {len(lines)}, Split: {split_idx} @ precent: {percent}")

    try:
        split_game_num = int(lines[split_idx].split("de_mirage_")[1].split("_round")[0])

        i_game_num = split_game_num
        
        while split_game_num == i_game_num:
            i_game_num = int(lines[split_idx].split("de_mirage_")[1].split("_round")[0])
            split_idx+=1
    except Exception as e: 
        print(f"Most likely due to de_mirage not being map used, Error: {e}")


    split_idx -= 1

    with open(round_train_txt_file, "w") as train_file, open(round_test_txt_file, "w") as test_file:
        for train_line in lines[:split_idx]:
            train_file.write(train_line)
        for test_line in lines[split_idx:]:
            test_file.write(test_line)

    #cleanup_data(demo_root)
        


