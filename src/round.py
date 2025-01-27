import os
import json
import pandas as pd
import sklearn

class Round:
    def __init__(self, round_title="",round_num=0, map_name=None, demo_data_root="../game_demos", enums_path:str="enums.json"):

        self.round_num=round_num
        self.round_title = round_title
        self.map_name = map_name

        self.start_tick = None
        self.end_tick = None
        self.players = []

        self.map = None
        self.bomb_planted = [False]
        self.bomb_postion = [(0,0,0)]

        self.tick_idxs = []
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

    def init_headers(self, players):
        for player in players:
            for attr in ['x', 'y', 'z', 'pitch', 'yaw', 'hp', 'flash_dur', 'has_helm', 'has_armor', 
                'has_defuse', 'primary', 'secondary', 'grenades']:
                col_name = f"{player.player_name}_{attr}"
                if col_name not in self.df.columns:
                    self.df[col_name] = pd.Series(dtype=object)
                    self.input_params_num += 1

    def load_round_data(self, round_dict: dict):

        print(f"Winner: {round_dict['winner'][self.round_num]}")
        self.winner = round_dict['winner'][self.round_num]
        self.input_params_num += 1

        if str(round_dict['bomb_plant'][self.round_num]) != "<NA>":
            self.bomb_plant_time = int(round_dict['bomb_plant'][self.round_num])
        else:
            self.bomb_plant_time = None
        self.input_params_num += 1
        
        print(f"Reason: {round_dict['reason'][self.round_num]}")
        self.reason = round_dict['reason'][self.round_num]
        self.input_params_num += 1

        self.df.loc[0, 'winner'] = round_dict['winner'][self.round_num]
        self.df.loc[0, 'reason'] = round_dict['reason'][self.round_num]
        self.df.loc[0, 'bomb_plant'] = self.bomb_plant_time
        self.input_params_num += 1

        # with open("round_class_info.txt", "a") as f:
        #     f.write(str(round_dict))



    def load_player_tick_data(self, players):

        for player in players:

            # Load postion of players

            self.df[f"{player.player_name}_x"] = pd.Series([pos[0] for pos in player.position])
            self.df[f"{player.player_name}_y"] = pd.Series([pos[1] for pos in player.position])
            self.df[f"{player.player_name}_z"] = pd.Series([pos[2] for pos in player.position])


            # Extract Yaw and Pitch
            self.df[f"{player.player_name}_pitch"] = pd.Series([pitch for pitch in player.pitch])
            self.df[f"{player.player_name}_yaw"] = pd.Series([yaw for yaw in player.yaw])

            self.df[f"{player.player_name}_hp"] = pd.Series([health for health in player.health])
            self.df[f"{player.player_name}_flash_dur"] = pd.Series([FlashDuration for FlashDuration in player.Flash])

            # Inventory
            self.df[f"{player.player_name}_has_helm"] = pd.Series([HasHelmet for HasHelmet in player.HasHelmet])
            self.df[f"{player.player_name}_has_armor"] = pd.Series([HasArmor for HasArmor in player.HasArmor])

            self.df[f"{player.player_name}_has_defuse"] = pd.Series([HasDefuser for HasDefuser in player.HasDefuser])
            self.df[f"{player.player_name}_primary"] = pd.Series([primary_weapon for primary_weapon in player.primary_weapon])
            self.df[f"{player.player_name}_secondary"] = pd.Series([secondary_weapon for secondary_weapon in player.secondary_weapon])

            self.df[f"{player.player_name}_grenades"] = pd.Series([grenade_count for grenade_count in player.grenade_count])

            self.df = self.df.copy()


        

    def write_round_to_csv(self):

        self.df.to_csv(self.csv_file, index=True)

        with open(self.round_txt_file, "a") as f:
            f.write(f"{self.csv_file}, InputParams: {self.input_params_num}\n")


        