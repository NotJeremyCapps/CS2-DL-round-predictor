import os
import json
import pandas as pd

class Round:
    def __init__(self, match_title="",demo_data_root="../game_demos", enums_path:str="enums.json"):

        self.round_title = match_title

        self.start_tick = None
        self.end_tick = None
        self.players = []

        self.map = None
        self.bomb_planted = [False]
        self.bomb_postion = [(0,0,0)]

        self.tick_idx = 0
        self.bomb_timer_left = 0

        self.outcome = None

        self.df = pd.DataFrame()

        preprocessed_dir_pth = os.path.join(demo_data_root, "preprocessed")
        os.makedirs(preprocessed_dir_pth, exist_ok=True)

        self.csv_file = os.path.join(preprocessed_dir_pth, f"{self.round_title}.csv")

        emun_file = open("enums.json", 'r')
        self.enums = json.loads(emun_file.read())


    def load_tick_data(self, tick_idx: int, tick_data):
        pass

    def load_player_tick_data(self, players):

        for player in players:

            # Load postion of players
            self.df[f"{player.player_name}_X"] = pd.concat([self.df[f"{player.player_name}_X"], pd.Series([player.position[0] for x in player.postion])], ignore_index=True)
            self.df[f"{player.player_name}_Y"] = pd.concat([self.df[f"{player.player_name}_X"], pd.Series([player.position[0] for x in player.postion])], ignore_index=True)
            self.df[f"{player.player_name}_Z"] = pd.concat([self.df[f"{player.player_name}_X"], pd.Series([player.position[0] for x in player.postion])], ignore_index=True)

            



        

    def write_round_to_csv(self):


        self.df['Outcome'] = self.outcome


        self.df.to_csv(index=True)
        pass