import os
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Round:
    def __init__(self, round_title="",round_num=0, map_name=None, demo_data_root="../game_demos", enums_path:str="enums.json"):

        self.round_num=round_num
        self.round_title = round_title
        self.map_name = map_name

        self.start_tick = None
        self.end_tick = None
        self.players = []

        self.map = None
        self.bomb_planted = []
        self.bomb_postion = []

        self.tick_start = 0
        self.tick_end = 0
        self.bomb_timer_left = 0

        self.winner = None
        self.bomb_plant_time = None
        self.reason = None

        self.df = pd.DataFrame()

        self.preprocessed_dir_pth = os.path.join(demo_data_root, "preprocessed", map_name)
        os.makedirs(self.preprocessed_dir_pth, exist_ok=True)

        self.csv_file = os.path.join(self.preprocessed_dir_pth, f"{self.round_title}.csv")
        self.round_txt_file = os.path.join(self.preprocessed_dir_pth, f"rounds.txt")

        emun_file = open("enums.json", 'r')
        self.enums = json.loads(emun_file.read())

        self.input_params_num = 0

    def init_headers(self):
        self.df["game_tick"] = pd.Series(dtype=object)
        # for player in players:
        #     for attr in ['x', 'y', 'z', 'pitch', 'yaw', 'hp', 'flash_dur', 'has_helm', 'has_armor', 
        #         'has_defuse', 'grenades']:
        #         col_name = f"{player.player_name}_{attr}"
        #         if col_name not in self.scaled_df.columns:
        #             self.scaled_df[col_name] = pd.Series(dtype=float)
        #             self.input_params_num += 1

        #     for attr in ['primary', 'secondary']:
        #         col_name = f"{player.player_name}_{attr}"
        #         if col_name not in self.categorical_df.columns:
        #             self.categorical_df[col_name] = pd.Series(dtype=int)
        #             self.input_params_num += 1

    def load_round_data(self, round_dict: dict):

        # print(f"Winner: {round_dict['winner'][self.round_num]}")
        # self.winner = round_dict['winner'][self.round_num]
        # self.input_params_num += 1

        # if str(round_dict['bomb_plant'][self.round_num]) != "<NA>":
        #     self.bomb_plant_time = int(round_dict['bomb_plant'][self.round_num])
        # else:
        #     self.bomb_plant_time = None
        # self.input_params_num += 1
        
        # print(f"Reason: {round_dict['reason'][self.round_num]}")
        # self.reason = round_dict['reason'][self.round_num]
        # self.input_params_num += 1

        self.df.loc[0, 'winner'] = round_dict['winner'][self.round_num]

        if round_dict['winner'][self.round_num] == "CT":
            self.df.loc[0, 'winner'] = 0
        else:
            self.df.loc[0, 'winner'] = 1

        # self.df.loc[0, 'reason'] = round_dict['reason'][self.round_num]
        # self.df.loc[0, 'bomb_plant'] = self.bomb_plant_time
        # self.input_params_num += 1

        self.df['game_tick'] = pd.Series([tick for tick in range(round_dict['freeze_end'][self.round_num], round_dict['end'][self.round_num]+1)])

        # with open("round_class_info.txt", "a") as f:
        #     f.write(str(round_dict))


    def load_player_tick_data(self, players):
       
        unscaled_dict = {}
        binary_dict = {}
        embed_dict = {}
        df = None

        for player in players:

            unscaled_dict[f"{player.player_name}_x"] = [pos[0] for pos in player.position]
            unscaled_dict[f"{player.player_name}_y"] = [pos[1] for pos in player.position]
            unscaled_dict[f"{player.player_name}_z"] = [pos[2] for pos in player.position]

            # Extract Yaw and Pitch

            unscaled_dict[f"{player.player_name}_pitch"] = [pitch for pitch in player.pitch]
            unscaled_dict[f"{player.player_name}_yaw"] = [yaw for yaw in player.yaw]

            unscaled_dict[f"{player.player_name}_hp"] = [health for health in player.health]
            unscaled_dict[f"{player.player_name}_flash_dur"] = [FlashDuration for FlashDuration in player.Flash]

            unscaled_dict[f"{player.player_name}_grenades"] = [grenade_count for grenade_count in player.grenade_count]

        # Values not scaled, so that can be pushed through embedding matrix
            # Inventory

            binary_dict[f"{player.player_name}_has_helm"] = [HasHelmet for HasHelmet in player.HasHelmet]
            binary_dict[f"{player.player_name}_has_armor"] = [HasArmor for HasArmor in player.HasArmor]
            binary_dict[f"{player.player_name}_has_defuse"] = [HasDefuser for HasDefuser in player.HasDefuser]
            binary_dict[f"{player.player_name}_has_bomb"] = [HasBomb for HasBomb in player.HasBomb]

            embed_dict[f"{player.player_name}_primary"] = [primary_weapon for primary_weapon in player.primary_weapon]
            embed_dict[f"{player.player_name}_secondary"] = [secondary_weapon for secondary_weapon in player.secondary_weapon]

        df = pd.DataFrame(unscaled_dict)

        binary_df = pd.DataFrame(binary_dict)

        embed_df = pd.DataFrame(embed_dict)

        embed_df.to_csv("embed_df.csv", index=True)

        scalar = MinMaxScaler(feature_range=(0, 1))
        norm_data = scalar.fit_transform(df)
        

        norm_df = pd.DataFrame(norm_data, columns=df.columns)

        norm_df.to_csv("scaled_df.csv", index=True)


        self.df = pd.concat([self.df, norm_df, binary_df, embed_df], axis=1)
        

    def write_round_to_csv(self):

        self.df.to_csv(self.csv_file, index=True)

        with open(self.round_txt_file, "a") as f:
            f.write(f"{self.csv_file}\n")


        