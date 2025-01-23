import os
import json
import pandas as pd

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
        self.bomb_postion = []

        self.tick_idxs = []
        self.bomb_timer_left = 0

        self.winner = None
        self.bomb_plant_time = None
        self.reason = None

        self.df = pd.DataFrame()

        preprocessed_dir_pth = os.path.join(demo_data_root, "preprocessed", map_name)
        os.makedirs(preprocessed_dir_pth, exist_ok=True)

        self.csv_file = os.path.join(preprocessed_dir_pth, f"{self.round_title}.csv")

        emun_file = open("enums.json", 'r')
        self.enums = json.loads(emun_file.read())

    def init_headers(self, players, start_tick, end_tick):
        self.start_tick = start_tick
        self.end_tick = end_tick
        self.players = players
        for player in players:
            for attr in ['x', 'y', 'z', 'pitch', 'yaw', 'hp', 'flash_dur', 'has_helm', 'has_armor', 
                'has_defuse', 'primary', 'secondary', 'grenades']:
                col_name = f"{player.player_name}_{attr}"
                if col_name not in self.df.columns:
                    self.df[col_name] = pd.Series(dtype=object)

    def load_round_data(self, round_dict: dict):

        print(f"Winner: {round_dict['winner'][self.round_num]}")
        self.winner = round_dict['winner'][self.round_num]

        if str(round_dict['bomb_plant'][self.round_num]) != "<NA>":
            self.bomb_plant_time = int(round_dict['bomb_plant'][self.round_num])
        else:
            self.bomb_plant_time = None
        
        print(f"Reason: {round_dict['reason'][self.round_num]}")
        self.reason = round_dict['reason'][self.round_num]

        self.df.loc[0, 'winner'] = round_dict['winner'][self.round_num]
        self.df.loc[0, 'reason'] = round_dict['reason'][self.round_num]
        self.df.loc[0, 'bomb_plant'] = self.bomb_plant_time

        #print("TEST!")
        #print("START: ",self.start_tick)
        #print("END: ", self.end_tick)
        for x in range(self.end_tick - self.start_tick):
        #    print("TEST?")
        #    print("length: ", len(self.players))
            for player in self.players:
                #print(self.df[f"{player.player_name}_x"][x])
                if(self.df[f"{player.player_name}_has_bomb"][x] == 1):
                    self.bomb_postion.append(((self.df[f"{player.player_name}_x"][x]), (self.df[f"{player.player_name}_y"][x]), (self.df[f"{player.player_name}_z"][x])))

            if(len(self.bomb_postion) == 0):
                self.bomb_postion.append((-128.00403, -1632.0, -0.9609375)) #a position i found in a round where the bomb started, prob somewhere in T spawn
            elif(len(self.bomb_postion) == x):
                self.bomb_postion.append(self.bomb_postion[-1])

        #print(self.bomb_postion)
                    
        #        print("!!!!!!")
                #print("HASBOMB!: ",self.df[f"{player.player_name}_has_bomb"][x])#if(self.df[f"{player.player_name}_has_bomb"]) == 1:
                #    print(player.player_name)

        with open("round_class_info.txt", "a") as f:
            f.write(str(round_dict))



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

            self.df[f"{player.player_name}_has_bomb"] = pd.Series([HasBomb for HasBomb in player.HasBomb])
            #print(self.df[f"{player.player_name}_has_bomb"])

            self.df = self.df.copy()


        

    def write_round_to_csv(self):

        self.df.to_csv(self.csv_file, index=True)
        