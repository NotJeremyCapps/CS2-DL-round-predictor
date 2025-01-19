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
        self.HasArmor = []
        self.FlashDuration = []
        self.HasDefuser = []
        self.grenade_count = []

        self.primary_weapon = []
        self.secondary_weapon = []
        self.active_weapon = []#not sure how to fill

        self.team_name = None

        emun_file = open("enums.json", 'r')
        self.enums = json.loads(emun_file.read())
        #self.enums = None

#tick idx is data frame, tick data is eata of tick
    def load_tick_data(self, tick_idx: int, tick_data, z):
        if(self.team_name == None):
            self.team_name = tick_data.team_name.loc[tick_idx+z]

        self.position.append([tick_data.X.loc[tick_idx+z], tick_data.Y.loc[tick_idx+z], tick_data.Z.loc[tick_idx+z]])
        self.pitch.append([tick_data.pitch.loc[tick_idx+z]])
        self.yaw.append(([tick_data.yaw.loc[tick_idx+z]]))
        self.health.append([tick_data.health.loc[tick_idx+z]])
        self.FlashDuration.append([tick_data.flash_duration.loc[tick_idx + z]])

        Defuser = [tick_data.has_defuser[tick_idx+z]]
        Helmet= [tick_data.has_helmet.loc[tick_idx+z]]
        armorvalue = ([tick_data.armor_value.loc[tick_idx+z]])

        if(Helmet[0] == False):
            self.HasHelmet.append(0) 
        else:
            self.HasHelmet.append(1)

        if(armorvalue[0] == 0):
            self.HasArmor.append(0)
        else:
            self.HasArmor.append(1)

        if(Defuser[0] == False):
            self.HasDefuser.append(0) 
        else:
            self.HasDefuser.append(1)


  #        player_HasHelmet = [curr_tick_info.has_helmet.loc[start_idx_curr_tick+z]]
        
        grenade_counter = 0
        # Enumerate primary and secondary weapons
        for weap in tick_data.inventory.loc[tick_idx+z]:
            if weap in self.enums["Player"]["primary_weapon"]:
                self.primary_weapon.append(self.enums["Player"]["primary_weapon"][weap])

            elif weap in self.enums["Player"]["secondary_weapon"]:
                self.secondary_weapon.append(self.enums["Player"]["secondary_weapon"][weap])

            elif weap in self.enums["Player"]["Grenade"]:
                grenade_counter += 1
            else: 
                #This should be a melee weapon if isnt in first 2 conditions
                pass
            #print(z)
            #print(tick_data.inventory.loc[tick_idx+z])

        self.grenade_count.append(grenade_counter)
      
        
        
        '''
        #enumerate active weapon
        for weap in tick_data.inventory.loc[tick_idx+z]:
            if weap in self.enums["Player"]["primary_weapon"]:
                self.primary_weapon.append(self.enums["Player"]["primary_weapon"][weap])
        '''


    '''
        def timer(self, decrement_list):
            prev_val = decrement_list[len(decrement_list)-2]
            val = decrement_list[len(decrement_list)-1]
            if(val == 0):
                pass
            elif(prev_val ==):
    '''



    def print_stats(self):
        
        print("Player:", self.player_name)
        print("Team Name:", self.team_name)
        print("Position:",self.position)
        print("Pitch:", self.pitch)
        print("Yaw:", self.yaw)
        print("Health:", self.health)
        print("HasHelmet:", self.HasHelmet)
        print("HasArmor:", self.HasArmor)
        print("PrimaryWeapon", self.primary_weapon)
        print("SecondaryWeapon",self.secondary_weapon)
        print("HasArmor:", self.HasArmor)
        print("FlashDuration:",self.FlashDuration)
        print("HasDefuser:", self.HasDefuser)
        
        print("Grenade Counter:", self.grenade_count)
    
