import json


class Player:
    
    def __init__(self, name, enums_path:str="enums.json"):
        
        self.player_name = name
        # Each parameter is a list where the index in the list corresponds to a tick
        self.position = []
        self.pitch = []
        self.yaw = []
        self.health = []
        self.HasHelmet = []
        self.HasArmor = []#need to fill

        self.primary_weapon = []
        self.secondary_weapon = []

        self.team_name = None

        emun_file = open("enums.json", 'r')
        self.enums = json.loads(emun_file.read())
        #self.enums = None

    def load_tick_data(self, tick_idx: int, tick_data, z):

        self.position.append([tick_data.X.loc[tick_idx+z], tick_data.Y.loc[tick_idx+z], tick_data.Z.loc[tick_idx+z]])
        self.pitch.append([tick_data.pitch.loc[tick_idx+z]])
        self.yaw.append(([tick_data.yaw.loc[tick_idx+z]]))
        self.health.append([tick_data.health.loc[tick_idx+z]])
        Helmet= [tick_data.has_helmet.loc[tick_idx+z]]
        
        if(Helmet[0] == False):
            self.HasHelmet.append(0) 
        else:
            self.HasHelmet.append(1)

  #        player_HasHelmet = [curr_tick_info.has_helmet.loc[start_idx_curr_tick+z]]
        
        # Enumerate primary and secondary weapons
        for weap in tick_data.inventory.loc[tick_idx+z]:
            if weap in self.enums["Player"]["primary_weapon"]:
                self.primary_weapon.append(self.enums["Player"]["primary_weapon"][weap])

            elif weap in self.enums["Player"]["secondary_weapon"]:
                self.secondary_weapon.append(self.enums["Player"]["secondary_weapon"][weap])

            else: 
                #This should be a melee weapon if isnt in first 2 conditions
                pass
        
    def print_stats(self):
        print("Player:", self.player_name)
        print("Position:",self.position)
        print("Pitch:", self.pitch)
        print("Yaw:", self.yaw)
        print("Health:", self.health)
        print("HasHelmet:", self.HasHelmet)
        print("HasArmor:", self.HasArmor)
        print("PrimaryWeapon", self.primary_weapon)
        print("SecondaryWeapon",self.secondary_weapon)
    
