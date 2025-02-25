import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Round:
    def __init__(self, round_title="",round_num=0, map_name=None, demo_data_root="../game_demos", enums_path:str="enums.json"):

        self.round_num=round_num
        self.round_title = round_title
        self.map_name = map_name

        self.start_tick = None
        self.end_tick = None
        self.players = []

        self.bomb_planted = []
        self.bomb_postion = []

        self.bomb_timer = []
        self.round_timer = []

        self.winner = None
        self.bomb_plant_time = None
        self.reason = None

        self.df = pd.DataFrame()
        self.other_dfs = {}

        self.preprocessed_dir_pth = os.path.join(demo_data_root, "preprocessed", map_name)
        os.makedirs(self.preprocessed_dir_pth, exist_ok=True)

        self.csv_file = os.path.join(self.preprocessed_dir_pth, f"{self.round_title}.csv")
        self.round_txt_file = os.path.join(demo_data_root, "preprocessed", f"rounds.txt")

        #self.round_train_txt_file = os.path.join(demo_data_root, "preprocessed", f"rounds_train.txt")
        #self.round_test_txt_file = os.path.join(demo_data_root, "preprocessed", f"rounds_test.txt")

        emun_file = open("enums.json", 'r')
        self.enums = json.loads(emun_file.read())

        self.input_params_num = 0



    def init_headers(self):
        self.df["game_tick"] = pd.Series(dtype=object)


    def load_round_data(self, round_dict: dict):

        info_dict = {}
        coord_dict = {}
        round_info_dict = {}

        self.start_tick = round_dict['freeze_end'][self.round_num]
        self.end_tick = round_dict['end'][self.round_num]

        # info_dict['winner'] = round_dict['winner'][self.round_num]

        # CT Win = 0
        # T Win = 1
        if round_dict['winner'][self.round_num] == "CT":
            round_info_dict['winner'] = [0]
        else:
            round_info_dict['winner'] = [1]

        # self.df.loc[0, 'reason'] = round_dict['reason'][self.round_num]
        # self.df.loc[0, 'bomb_plant'] = self.bomb_plant_time
        # self.input_params_num += 1

        # Load bomb info
        if str(round_dict['bomb_plant'][self.round_num]) != "<NA>":
            self.bomb_plant_time = int(round_dict['bomb_plant'][self.round_num]) - self.start_tick #when bomb was planted tick relative to round start
        else:
            self.bomb_plant_time = None

        for tick in range((self.end_tick - self.start_tick) + 1):

            if(tick == 0):
                self.round_timer.append(0)
            else:
                self.round_timer.append(self.round_timer[-1] + 1)

            if(self.bomb_plant_time == None):
                self.bomb_planted.append(0)
                self.bomb_timer.append(0)
            elif(tick < self.bomb_plant_time):
                self.bomb_planted.append(0)
                self.bomb_timer.append(0)
            else:
                self.bomb_planted.append(1)
                self.bomb_timer.append(self.bomb_timer[-1] + 1)


            for player in self.players:
                if(self.other_dfs['binary_df'][f"{player.player_name}_has_bomb"][tick] == 1):
                    self.bomb_postion.append((self.other_dfs['coord_df'][f"{player.player_name}_x"][tick], self.other_dfs['coord_df'][f"{player.player_name}_y"][tick], self.other_dfs['coord_df'][f"{player.player_name}_z"][tick]))
                    
            if(len(self.bomb_postion) == 0):
                self.bomb_postion.append((1136.0, 32.0, -164.78845)) #a position i found in a round where the bomb started, prob somewhere in T spawn
            elif(len(self.bomb_postion) == tick):
                self.bomb_postion.append(self.bomb_postion[-1])

        # with open("round_class_info.txt", "a") as f:
        #     f.write(str(round_dict))

        
        coord_dict["bomb_x"] = [pos[0] for pos in self.bomb_postion]
        coord_dict["bomb_y"] = [pos[1] for pos in self.bomb_postion]
        coord_dict["bomb_z"] = [pos[2] for pos in self.bomb_postion]


        self.df["game_tick"] = list(range(round_dict['freeze_end'][self.round_num], round_dict['end'][self.round_num]+1) / round_dict['end'].values[-1])

        if 1 in self.bomb_planted: #only scale timer values if bomb planted
            self.df[f"bomb_timer"] = (np.array(self.bomb_timer) - min(self.bomb_timer)) / (max(self.bomb_timer) - min(self.bomb_timer)) #scale to between 0 and 1
        else:
            self.df[f"bomb_timer"] = self.bomb_timer


        self.other_dfs['coord_df'] = pd.concat([self.other_dfs['coord_df'], pd.DataFrame(coord_dict)], axis=1)

        self.other_dfs['scaled_df'] = pd.concat([self.other_dfs['scaled_df'], pd.DataFrame(info_dict)], axis=1)

        # Want it to be last column of csv
        self.other_dfs['round_end_df'] = pd.DataFrame(round_info_dict)


    def load_player_tick_data(self, players):
       
        unscaled_dict = {}
        binary_dict = {}
        embed_dict = {}
        coord_dict = {}
        angle_dict = {}

        for player in players:

            coord_dict[f"{player.player_name}_x"] = [pos[0] for pos in player.position]
            coord_dict[f"{player.player_name}_y"] = [pos[1] for pos in player.position]
            coord_dict[f"{player.player_name}_z"] = [pos[2] for pos in player.position]

            # Extract Yaw and Pitch and sin/cos encode

            angle_dict[f"{player.player_name}_pitch_c"] = np.cos(player.pitch)
            angle_dict[f"{player.player_name}_pitch_s"] = np.sin(player.pitch)

            angle_dict[f"{player.player_name}_yaw_c"] = np.cos(player.yaw)
            angle_dict[f"{player.player_name}_yaw_s"] = np.sin(player.yaw)

            unscaled_dict[f"{player.player_name}_hp"] = player.health
            unscaled_dict[f"{player.player_name}_flash_dur"] = player.Flash

            unscaled_dict[f"{player.player_name}_grenades"] = player.grenade_count


        # Values not scaled
            # Inventory

            binary_dict[f"{player.player_name}_has_helm"] = player.HasHelmet
            binary_dict[f"{player.player_name}_has_armor"] = player.HasArmor
            binary_dict[f"{player.player_name}_has_defuse"] = player.HasDefuser
            binary_dict[f"{player.player_name}_has_bomb"] = player.HasBomb
        # Values not scaled but need to be passed through embedding layer
            embed_dict[f"{player.player_name}_primary"] = player.primary_weapon
            embed_dict[f"{player.player_name}_secondary"] = player.secondary_weapon

        self.other_dfs['scaled_df'] = pd.DataFrame(unscaled_dict)

        self.other_dfs['binary_df'] = pd.DataFrame(binary_dict) # Binary Data Df

        self.other_dfs['coord_df'] = pd.DataFrame(coord_dict)

        self.other_dfs['angle_df'] = pd.DataFrame(angle_dict)

        self.other_dfs['embed_df'] = pd.DataFrame(embed_dict)



    def write_round_to_csv(self):

        # Normalize coordinates 
        # Map: de anubis
        # [X] - Max: 1460, Min: -2660
        # [Y] - Max: 891, Min: -2606
        # [Z] - Max: 30, Min: -371
        max_x = 1460
        min_x = -2660
        max_y = 891
        min_y = -2606
        max_z = 30
        min_z = -371

        # print(f"{self.other_dfs['coord_df'].columns.astype(str)}")

        # rand_col = self.other_dfs['coord_df'].columns[-1]
        # print(rand_col)

        # Header column shows up in extraction, need to get rid of it
        # self.other_dfs['coord_df'].columns = self.other_dfs['coord_df'].columns[:-1].astype(str)

        # Header column shows up in extraction, need to get rid of it
        x_cols = [col for col in self.other_dfs['coord_df'].columns.astype(str) if col.endswith('_x')]
        y_cols = [col for col in self.other_dfs['coord_df'].columns.astype(str) if col.endswith('_y')]
        z_cols = [col for col in self.other_dfs['coord_df'].columns.astype(str) if col.endswith('_z')]

        # Nomalize coordinate columns to be from -1 to 1
        for col in x_cols:
            self.other_dfs['coord_df'][col] = (2 * ((self.other_dfs['coord_df'][col] - min_x) / (max_x - min_x))) - 1
            
        for col in y_cols:
            self.other_dfs['coord_df'][col] = (2 * ((self.other_dfs['coord_df'][col] - min_y) / (max_y - min_y))) - 1
        
        for col in z_cols:
            self.other_dfs['coord_df'][col] = (2 * ((self.other_dfs['coord_df'][col] - min_z) / (max_z - min_z))) - 1

        # Normalize Angle Values

        # print(f"Angle Columns: {self.other_dfs['angle_df'].columns}")

        # for col in self.other_dfs['angle_dict']:
        #     self.other_dfs['angle_df'][f"{col}_s"] = np.sin(self.other_dfs['angle_dict'][col])
        #     self.other_dfs['angle_df'][f"{col}_s"] = np.sin(self.other_dfs['angle_dict'][col])
        #     self.other_dfs['angle_df'].drop(f'{col}', axis=1) # Drop none encoded column

        # print(self.other_dfs['scaled_df'])
        # scale health data to between 0 and 1

        health_cols = [col for col in self.other_dfs['scaled_df'].columns.astype(str) if col.endswith('_hp')]

        for col in health_cols:
            self.other_dfs['scaled_df'][col] = self.other_dfs['scaled_df'][col] / 100 # Max healthe is 100 

        grenade_cols = [col for col in self.other_dfs['scaled_df'].columns.astype(str) if col.endswith('_grenades')]

        for col in grenade_cols:
            self.other_dfs['scaled_df'][col] = self.other_dfs['scaled_df'][col] / 4 # Max of 4 grenades

        flash_dur_cols = [col for col in self.other_dfs['scaled_df'].columns.astype(str) if col.endswith('_flash_dur')]

        for col in flash_dur_cols:
            self.other_dfs['scaled_df'][col] = self.other_dfs['scaled_df'][col] / (3 * 60) # Max of 3 sec flash dur with 60 tick per sec


        # scalar = MinMaxScaler(feature_range=(0, 1))
        # norm_data = scalar.fit_transform(self.other_dfs['scaled_df'])
        

        # scalar = StandardScaler()
        # scalar.fit()

        # scalar.transform()
        

        # self.other_dfs['scaled_df'] = pd.DataFrame(norm_data, columns=self.other_dfs['scaled_df'].columns)

        # try:
        #     # Get rid of dict from unscaled data
        #     self.other_dfs.pop('unscaled_df')

        #     # Get rid of dict for unscaled coordinates
        # except Exception as e:
        #     print(f"Error e: {e}")

        self.df = pd.concat([self.df, *self.other_dfs.values()], axis=1)

        # for col in self.df.columns:
        #     if col.endswith('_primary'): # Gets the first instance of primary for later enbedding processing
        #         categ_idx = self.df.columns.get_loc(col)
        #         break
        

        self.df.to_csv(self.csv_file, index=True)
        with open(self.round_txt_file, "a") as f:
            f.write(f"{self.csv_file}\n")
        #if(text_file == 0):
        #    with open(self.round_test_txt_file, "a") as f:
        #        f.write(f"{self.csv_file}\n")
        #else:
        #    with open(self.round_train_txt_file, "a") as f:
        #        f.write(f"{self.csv_file}\n")

        