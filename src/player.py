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
        self.Flash = []
        self.HasDefuser = []
        self.grenade_count = []

        self.primary_weapon = []
        self.secondary_weapon = []
        self.active_weapon = []#not sure how to fill
        self.HasBomb = []

        self.team_name = None

        emun_file = open("enums.json", 'r')
        self.enums = json.loads(emun_file.read())
        #self.enums = None

#tick idx is data frame, tick data is data of tick
    def load_tick_data(self, tick_idx: int, tick_data, z):
        if(self.team_name == None):
            self.team_name = tick_data.team_name.loc[tick_idx+z]

        self.position.append([tick_data.X.loc[tick_idx+z], tick_data.Y.loc[tick_idx+z], tick_data.Z.loc[tick_idx+z]])
        self.pitch.append([tick_data.pitch.loc[tick_idx+z]][0])
        self.yaw.append(([tick_data.yaw.loc[tick_idx+z]][0]))
        self.health.append([tick_data.health.loc[tick_idx+z]][0])


        
        Flash_Duration = [tick_data.flash_duration.loc[tick_idx + z]]
        Defuser = [tick_data.has_defuser[tick_idx+z]]
        Helmet= [tick_data.has_helmet.loc[tick_idx+z]]
        armorvalue = ([tick_data.armor_value.loc[tick_idx+z]])


        if(Flash_Duration[0] > 0):
            self.Flash.append(1)
        else:
            self.Flash.append(0)
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
        
        self.HasBomb.append(0)
        grenade_counter = 0
        primary_weapon = 0 
        secondary_weapon = 0 
        # Enumerate primary, secondary weapons, grenade, and has_bomb based on player inventory
        for weap in tick_data.inventory.loc[tick_idx+z]:
            if weap in self.enums["Player"]["primary_weapon"]:
                self.primary_weapon.append(self.enums["Player"]["primary_weapon"][weap])
                primary_weapon  = 1
            elif weap in self.enums["Player"]["secondary_weapon"]:
                self.secondary_weapon.append(self.enums["Player"]["secondary_weapon"][weap])
                secondary_weapon = 1
                '''
                sec_weap = self.enums["Player"]["secondary_weapon"][weap]
                if(sec_weap == ''):
                    self.secondary_weapon.append(0)
                else:
                    self.secondary_weapon.append(sec_weap)
                '''

            elif weap in self.enums["Player"]["grenade"]:
                grenade_counter += 1
            elif (weap == "C4"):#bomb
                self.HasBomb[-1] = 1#assumes only one bomb on all players at a time.
                hasBomb = True
            else: 
                #This should be a melee weapon if isnt in first 2 conditions
                pass
      
    
        if(secondary_weapon == 0):
            self.secondary_weapon.append(0)
        if(primary_weapon == 0):
            self.primary_weapon.append(0)


        self.grenade_count.append(grenade_counter)
      
        


    def print_stats(self):
        
        ''' 
        print("Player:", self.player_name)
        print("Team Name:", self.team_name)
        print("Position:",self.position)
        print("Pitch:", self.pitch)
        
        print("Health:", self.health)
        print("HasHelmet:", self.HasHelmet)
        print("HasArmor:", self.HasArmor)
        print("PrimaryWeapon", self.primary_weapon)
        print("SecondaryWeapon",self.secondary_weapon)
        print("HasArmor:", self.HasArmor)
        
        print("Flash:",self.Flash)
        print("HasDefuser:", self.HasDefuser)
        
        print("Grenade Counter:", self.grenade_count)
        print("HasBomb:", self.HasBomb)
        print("Yaw:", self.yaw)
        '''
        print("SecondaryWeapon",self.secondary_weapon)


    