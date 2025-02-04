import torch
import json
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

# Load enums for embedding layer
emun_file = open("enums.json", 'r')
enums = json.loads(emun_file.read())

class AudioLSTM(nn.Module):

    def __init__(self, n_feature=3, out_feature=3, n_hidden=256, n_layers=2, drop_prob=0.5):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_feature = n_feature

        self.prim_embed_dim = 9
        self.sec_embed_dim = 4

        self.prim_weap_embedding = nn.Embedding(len(enums['Player']['primary_weapon']), self.prim_embed_dim) # 25 vocab size
        self.sec_weap_embedding = nn.Embedding(len(enums['Player']['secondary_weapon']), self.sec_embed_dim) # 9 vocab size


        self.lstm = nn.LSTM(self.n_feature, self.n_hidden, self.n_layers, dropout=self.drop_prob, batch_first=True)

        self.layer_norm = nn.LayerNorm()

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(n_hidden, out_feature)

    def forward(self, x, cat_data, hidden):

        # Embed categorical primary and secondary weapo s
        
        prim_embed_tensors = []

        # TODO: Determine sequence length for padded data
        lengths = [seq.size(0) for seq in x]
        # x.shape (batch, seq_len, n_features)

        padded_sequences = rnn_utils.pad_sequence(x, batch_first=True)


        x = rnn_utils.pack_padded_sequence(padded_sequences, lengths, batch_first=True, enforce_sorted=False)

        l_out, l_hidden = self.lstm(x, hidden)

        # out.shape (batch, seq_len, n_hidden*direction)
        out = self.dropout(l_out)

        # out.shape (batch, out_feature)
        out = self.fc(out[:, -1, :])

        # return the final output and the hidden state
        return out, l_hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if torch.cuda.is_available():
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else: 
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden