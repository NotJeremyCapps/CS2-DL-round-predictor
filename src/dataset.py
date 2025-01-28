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
       
        #replace this section with section from get_item

        self.list = list
        #change into tensor
        #self.data = torch.tensor(data)
        #self.target = torch.tensor(target)

      #reload
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
        target = []
        for i in range(1, len(self.all_data)-1):#increment over every row
        #split into target and data indexes 
            target.append(float(self.all_data[i][-4])) #create target array --------THIS SHOULD BE THE INDEX OF THE WIN!-----------

        #create a tensor
        self.target = torch.tensor(target)


        #create data as float
        int_data = []
        for i in range(1, len(self.all_data)-1):#increment over every row
            temp_array = []
            for m in range(len(self.all_data[i])):
                if(m == len(self.all_data[i]) - 4):# if increment is the result do not add
                    continue
                else:
                    #convert to float
                    if(self.all_data[i][m] != ''):
                        temp_array.append(float(self.all_data[i][m])) 
                    else:
                        temp_array.append(0) #do not convert empty spaces, replace with 0 SHOULD FIX THIS IN ACTUAL CODE - EG SECONDARY WEAPON - THIS IS NOT A PERMANENT FIX ----

            int_data.append(temp_array)




        #change into tensor
        self.data = torch.tensor(int_data) #data index 

        print(self.data)
        self.offset = 0 #offset from padding
        self.csvfile = 1



    


    def __getitem__(self, index): #index is beginning index of sequence, this assumes all the data for rounds and games is sequential
        
        #--- We need a tick column --- 
        index = index + self.offset
        if self.data[index][0] == 0: #tick is 0 may need to pad
            len_of_round = 0 #0 based
            while(self.data[len_of_round + 1][0] > self.data[len_of_round][0]):
                len_of_round += 1
            excess= len_of_round%self.sequence_length # 0 based
            if(excess!= 0): #round needs to be padded
                x = P.pad(self.data[0:(self.sequence_length - excess-1)], (0, excess-1), value=0) 
            #x = torch.cat((padding, self.data[index][0:len_of_round]), 0)
        else: #if ticks restart -> pad
           x = self.data[index :index + self.sequence_length]

        '''
        #reload
        if(index + self.sequence_length == self.data.shape[0]):
            #read in csv
            readingcsv = csv.reader(self.list[self.csvfile])
            self.all_data = []
            for row in readingcsv: 
                self.all_data.append(row) #first index is row and then col
        
            target = []
            data = []
            for i in range(len(self.all_data)):
            #split into target and data indexes 
                target.append(self.all_data[-3][i]) #create target array
                del self.all_data[-3][i] #remove target from all data

            #change into tensor
            self.data = torch.tensor(self.all_data) #data index 
            self.target = torch.tensor(target)

            self.offset = 0 #offset from padding
            self.csvfile +=1
        '''
        return x



        data, label = self.tensor_dataset[index], self.labels[index]
        #or self.tensor_dataset.columns will return labels of columns



        return data, label
    
    def __len__(self): #samples in dataset
        return self.data.shape[0]
    


