import json


class Player:
    
    def __init__(self, 
                 enums_path:str="enums.json"
                ):
        
        # Each parameter is a list where the index in the list corresponds to a tick
        self.postion = [(0,0,0)]
        self.pitch = []
        self.yaw = []
        self.health = []
        self.HasHelmet = []
        self.HasArmor = []

        self.primary_weapon = None
        self.secondary_weapon = None

        self.team_name = None

        # self.enums = json.loads(enums_path)
        self.enums = None


    def load_tick_data(self, tick_idx: int, tick_data):

        # Enumerate primary and secondary weapons
        for weap in tick_data.inventory.loc[tick_idx]:
            if weap in self.enums["primary_weapon"]:
                self.primary_weapon = self.enums["primary_weapon"][weap]

            elif weap in self.enums["secondary_weapon"]:
                self.primary_weapon = self.enums["secondary_weapon"][weap]

            else: 
                #This should be a melee weapon if isnt in first 2 conditions
                pass
        
        



    
