import torch
import json
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F

ENUMS_PATH = "enums.json"

# Load enums for embedding layer
emun_file = open(ENUMS_PATH, 'r')
enums = json.loads(emun_file.read())

class CS2LSTM(nn.Module):

    def __init__(self, n_feature=None, out_feature=1, n_hidden=256, n_layers=3, drop_prob=0.5):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden 
        self.n_feature = n_feature

        self.prim_embed_dim = 9
        self.sec_embed_dim = 4

        # TODO: change to non hard coded
        # Non-Categorical features for players + (Weapon encodings for all 10 players)
        self.n_feature = 146 + (10 *(self.prim_embed_dim + self.sec_embed_dim))

        # Add 2 to embedding vocab size: 
        # https://discuss.pytorch.org/t/solved-assertion-srcindex-srcselectdimsize-failed-on-gpu-for-torch-cat/1804/13
        
        self.prim_weap_embedding = nn.Embedding(len(enums['Player']['primary_weapon'])+2, self.prim_embed_dim) # 25 vocab size
        self.sec_weap_embedding = nn.Embedding(len(enums['Player']['secondary_weapon'])+2, self.sec_embed_dim) # 9 vocab size


        self.lstm = nn.LSTM(self.n_feature, self.n_hidden, self.n_layers, dropout=self.drop_prob, batch_first=True)

        # self.layer_norm = nn.LayerNorm()

        self.dropout = nn.Dropout(drop_prob)
        # self.dropout3 = nn.Dropout(drop_prob)

        self.fc = nn.Linear(n_hidden, n_hidden)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(drop_prob)
        self.fc2 = nn.Linear(n_hidden, out_feature)

    def forward(self, x_main_data, x_prim_weap, x_sec_weap, hidden):

        # Embed categorical primary and secondary weapons
        
        # x_prim_weap (batch_size, seq_len, 10(weapon of 10 player))

        # Pass through embedding layer
        prim_weap_embed = self.prim_weap_embedding(x_prim_weap) 
        sec_weap_embed = self.sec_weap_embedding(x_sec_weap) 

        # prim_weap_embed = self.dropout1(prim_weap_embed)
        # sec_weap_embed = self.dropout2(prim_weap_embed)

        combined_embeds = torch.cat([prim_weap_embed, sec_weap_embed], dim=-1)  


        B, T, num_players, embed_dim = combined_embeds.shape
        combined_embeds = combined_embeds.view(B, T, num_players * embed_dim)


        # x.shape (batch_size, seq_len, main_features + weapon_embeddings(276))
        x = torch.cat([x_main_data, combined_embeds], dim=-1)

        # with open("prim_weap_embed.txt", "a") as f: 
        #     f.write(f"Prim embedding: {x_main_data}, Shape: {x_main_data.shape}")

    # find lengths of valid data in padded sequences
        # valid_mask = x_data_combined.ne(0).any(dim=-1)  # Shape: (batch_size, seq_length)
    
        # Sum over the sequence dimension to count valid time steps
        # lengths = valid_mask.sum(dim=1).cpu()

        # x.shape (batch, seq_len, n_features)
        # x = rnn_utils.pack_padded_sequence(x_data_combined, lengths, batch_first=True, enforce_sorted=False)

        l_out, l_hidden = self.lstm(x, hidden)

        # unpack
        # l_unpacked, _ = rnn_utils.pad_packed_sequence(l_out, batch_first=True)

        # out.shape (batch, hidden_size, hidden_size)
        out = self.dropout(l_out)

        # with open("dropoutput.txt", "a") as f: 
        #     f.write(f"Dropout out: {out}, Shape: {out.shape}")

        # out.shape (batch, out_feature)
        output = self.fc(out[:, -1, :]) # last time step

        output = self.relu(output)

        output = self.dropout2(output)

        output = self.fc2(output)

        # with open("fc_out.txt", "a") as f:
        #     f.write(f"{out}\n\n")

        # No sigmoid on output when using BCELogigitLoss : 
        # https://stackoverflow.com/questions/75979632/pytorchs-nn-bcewithlogitsloss-behaves-totaly-differently-than-nn-bceloss
        
        # output = torch.sigmoid(out)  # Convert to probabilities

        # return the final output and the hidden state
        return output, l_hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if torch.cuda.is_available():
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else: 
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden