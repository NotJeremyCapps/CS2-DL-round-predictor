# Using Deep Learning to Predict the Outcome of a Counter-Strike 2 Round

In this project, we use deep learning to train a model to predict the outcome of a Counter-Strike 2 round at any given time during the round. Any given time is at any instance, or tick during the round given the game state during that instance of time or tick. In Counter-Strike 2 time is measured in discrete intervals of 60 times a second, each instance is called a tick, and given the game state in a tick our model should give a percent chance for each team to win the current round.

Demo Parsing

The first main stage of this project is the demo parser. Given the amount of demos, seconds in a demo, and ticks in the demo, there is a lot of data to be processed. Additionally, there are a significant amount of parameters for this model - around 200 - given the amount of moving parts in the game such as weapon selection and location of bomb. There are also 10 players requiring that data for each player individually be recorded and taken into consideration. Each demo contains 1 game with at least 13 rounds. The demos for this predictor has been constrained to the Mirage map to limit inconsistinces in the data. 

The demos are processed into csv files using demo_parser.py. The rows of the csv indicate the ticks, and the columns indicate the parameters. Each individual round is converted into its individual csv file. The csv file paths are then printed into rounds.txt. String data such as the name of the weapons are enumerated in enums.json and are processed through the embedded layer in the model that takes in the caterogrial data. Integer or float data is normalized in the rounds.py class which is called in demo_parser.py. Demo_parser has multiple classes that support the data processing including rounds.py and player.py. 

At the end of the demo parsing, a split function is called to split the data file paths into a traning and testing text file to be called seperately for training and testing. The split function is in dataset.py and can optionally be called when running training.py.

Command: Python demo_parser.py


Training

Once the data is parsed, it is ready to be used in the model for training. To train the model, call training.py. The model used for this project is an LSTM model. Training.py instantiates the model from model.py and trains/tests its. You can set the epochs, batch size, and sequence length values near the top of this file.The dataset.py file is also called when running the training. Dataset.py is what reads the csv files and converts the data into tensors. It also is the file returns the tensors for requested indexes of the data. Additionally, it overlaps data between sequences to create more data to train/test the model. The overlap percentage can be set by the overlap variable in the innit function. A overlap value of 2 represents an overlap of 50% of the data. A value of 3 would represent a overlap value of 1/3.

Command: Python training.py