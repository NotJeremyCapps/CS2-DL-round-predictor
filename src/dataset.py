import os
import torch
from torch.utils.data import Dataset
import torch.nn.functional as P
from torch.utils.data import Subset
import pandas as pd
import csv


class CS2PredictionDataset(Dataset):
    def __init__(self, list, sequence_length): #a data point is one tick/timestamp, target is win/loss, data is parameter data, sequence_length eg 30 seconds 

        #sets amount of data points/ticks for each sequence
        self.sequence_length= sequence_length
       
        self.list = list
        #read in csv
        with open(self.list, 'r') as file1: #open text file to list of csv files
            words = file1.read().strip().split()
            print(words)
            with open(words[0], 'r') as file2: #open first csv file as file2
        #with open(self.list, 'r'):
                readingcsv = csv.reader(file2)
                self.all_data = []
                for row in readingcsv: 
                    self.all_data.append(row) #first index is row and then col

        #create target array  

        self.target_for_round= float(self.all_data[1][-1]) #create target array for target

        #create data as float
        int_data = []
        for i in range(1, len(self.all_data)-1):#increment over every row
            temp_array = []
            for m in range(len(self.all_data[i])-1): #do not include win column
                    #convert to float
                    temp_array.append(float(self.all_data[i][m])) 
                   
            int_data.append(temp_array)




        #change into tensor
        self.data = torch.tensor(int_data) #data index 

        print(self.data)
        self.offset = 0 #offset from padding
        self.csvfile = 1



    


    def __getitem__(self, index): #index is beginning index of sequence, this assumes all the data for rounds and games is sequential
        
        #--- We need a tick column --- 
        index = index + self.offset
        if self.data[index][0] == 0: #tick is 0 may need to pa
            len_of_round = len(self.data)
            excess= len_of_round%self.sequence_length # not 0 based
            if(excess!= 0): #round needs to be padded
                x = P.pad(self.data,(0,0,excess,0), value=0) 
            #x = torch.cat((padding, self.data[index][0:len_of_round]), 0)
            self.offset = excess
        else: #if ticks restart -> pad
           x = self.data[index :index + self.sequence_length]

        with open('tensor_output.txt', 'w') as f:
            torch.set_printoptions(threshold=torch.inf)
            print(x, file = f)



        #reload
        if(index + self.sequence_length ==  len(self.data)):      
        
            #read in csv
            with open(self.list, 'r') as file1: #open text file to list of csv files
                words = file1.read().strip().split()
                with open(words[self.csvfile * 3], 'r') as file2: #open first csv file as file2
                    readingcsv = csv.reader(file2)
                    self.all_data = []
                    for row in readingcsv: 
                        self.all_data.append(row) #first index is row and then col

            #create target array  

            self.target_for_round= float(self.all_data[1][-1]) #create target array for target

            #create data as float
            int_data = []
            for i in range(1, len(self.all_data)-1):#increment over every row
                temp_array = []
                for m in range(len(self.all_data[i])-1): #do not include win column
                        #convert to float
                        temp_array.append(float(self.all_data[i][m])) 
                    
                int_data.append(temp_array)




            #change into tensor
            self.data = torch.tensor(int_data) #data index 

            self.offset = 0 #offset from padding
            self.csvfile += 1

        return x, self.target_for_round



        data, label = self.tensor_dataset[index], self.labels[index]
        #or self.tensor_dataset.columns will return labels of columns



        return data, label
    
    def __len__(self): #samples in dataset
        csv_count = 0
        total_len = 0
        with open(self.list, 'r') as file1: #open text file to list of csv files
            words = file1.read().strip().split()
            while(csv_count *3 < len(words)-1):
                with open(words[csv_count * 3], 'r') as file2: #open first csv file as file2
                    readingcsv = csv.reader(file2)
                    current_data = []
                    for row in readingcsv: 
                        current_data.append(row) #first index is row and then col
                    total_len = total_len + len(current_data) - 1
                    csv_count =csv_count + 1
        
        print(total_len)

        return total_len




'''
    -add tick column
    -secondary weapon needs to be enum to 0 if they have no weapon
    -replace with index of win

'''
