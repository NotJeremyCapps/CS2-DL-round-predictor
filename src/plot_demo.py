from awpy import Demo
from awpy import plot
import cv2.text
from player import Player
import math
from typing import Sequence
import json
import pandas as pd
import torch
from model import CS2LSTM
#import sys

#sys.path.append("/c/Users/notje/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/LocalCache/local-packages/Python311/site-packages/opencv_cuda/install_script.py")
import cv2
import numpy as np
import os
# CPU only, would be nice to have a GPU version

PATH = "../game_demos/"
#g2-vs-heroic-m3-mirage.dem
#00nation-vs-eclot-m1-mirage.dem
DEMO_NAME = "replay.dem"

OUTPUT_PATH = "../parsed_videos/"

ASSETS_PATH = OUTPUT_PATH + "assets/"

MODEL_PATH = "trained_model.pt"

CT_COLOR = (247, 204, 29)
T_COLOR = (36, 219, 253)

ALPHA_THRESH = 0

#fps = 60

FRAME_SIZE = (1024, 1024)

FPS = 64

FOURCC = cv2.VideoWriter_fourcc(*'XVID')

weapon_translate_file = open("weapon_translate.json", 'r')
weapon_translate = json.loads(weapon_translate_file.read())
weapon_translate_file.close()

def main():

    # Example: Read frames from folder
    #image_folder = '../parsed_videos/assets'
    #images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    #for image in images:
        #frame = cv2.imread(os.path.join(image_folder, image))
        #cv2.circle(frame, (int(translate_position(-1656, "x")), int(translate_position(-1800, "y"))), 8, (0,0,255), -1)
        #video_writer.write(frame)


    #f = open("test.txt", "w")

    #weapon_translate_file = open("weapon_translate.json", 'r')
    #weapon_translate = json.loads(weapon_translate_file.read())
    #weapon_translate_file.close()


    parser = Demo(PATH + DEMO_NAME)

    #model = CS2LSTM(n_feature=None, out_feature=1,n_hidden=60 ,n_layers=2)
    #model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
    model = torch.load(MODEL_PATH, weights_only=False)
    model.eval()

    hidden = model.init_hidden(28) #initialize hidden variable
    model.eval()

    for param in model.parameters():
        print(param)
    #print(model.parameters)

    #print(model)

    #print(parser.infernos)
    #print(parser.smokes.columns)
    #print(parser.events["hegrenade_detonate"].columns)

    #with open("nades.txt", 'w') as file:
    #    file.write(parser.grenades.to_string())
    #file.close()

    #with open("nades2.txt", 'w') as file:
    #    file.write(parser.parser.parse_grenades().to_string())
    #file.close()
    ##print(parser.grenades.columns)

    #SOME DEMOS DO NOT HAVE WARMUP LEADING TO FIRST PISTOL ROUND NOT BEING PARSED!
    round_starts = (parser.rounds["freeze_end"]) #this is the ticks for end of freeze time at start of rounds
    round_ends = (parser.rounds["end"]) #does not include time between round determination and respawn


    all_weapons = parser.parser.parse_ticks(wanted_props=["active_weapon_name"])

    equipped_items_all = parser.parser.parse_event(event_name="item_equip")
    #print(equipped_items_all.columns)
    #equipped_ticks = equipped_items_all["tick"].tolist()
    #equipped_steamid = equipped_items_all["user_steamid"].tolist()
    #equipped_items = equipped_items_all["item"].tolist()
#"is_bomb_dropped","inventory","inventory_as_ids"

    #testing = parser.parser.parse_ticks(wanted_props=["inventory","is_bomb_dropped","is_bomb_planted","dropped_at_time"])
    #print(testing.query('tick == 120')["tick"].head(1))
    #print(testing.query('tick == 120')["is_bomb_dropped"].head(1))
    #print(testing.query('tick == 120')["inventory"].head(10))#.columns)
   #print(testing["dropped_at_time"].head(10))

    #print(parser.parser.list_game_events())

    #print(parser.parser.parse_event(event_name="item_pickup").query("item == 'defuser'"))

    defuser_pickups = parser.parser.parse_event(event_name="item_pickup").query("item == 'defuser'").sort_index(ignore_index=True)

    #print(defuser_pickups)
    #print("test")

    defuser_pickups_index = 0

    bomb_pickups = parser.parser.parse_event(event_name="bomb_pickup")
    bomb_drops = parser.parser.parse_event(event_name="bomb_dropped")
    bomb_plants = parser.parser.parse_event(event_name="bomb_planted")

    bomb_pickups_index = 0
    bomb_drops_index = 0
    bomb_plants_index = 0

    all_grenades = parser.grenades

    print(all_grenades.columns)

    #print(bomb_status[0])
    #print(bomb_status[1])
    #print(bomb_status[2])

    #print(round_starts)


    #print(parser.parser.parse_event("dropped_at_time"))


    #print(parser.parser.parse_item_drops())


    #for i in range(len(equipped_items_all)):
    #    print(equipped_items_all["item"][i], " ", equipped_items_all["weptype"][i])
    #equipped_skins = equipped_items_all.columns
    #print(equipped_items_all["defindex"])#.columns)
    #for i in equipped_items_all["hassilencer"]:
    #   print(i)
    #equip_list = [equipped_ticks, equipped_steamid, equipped_items]

    #equipped_items.values.tolist()

    #event_df = parser.parser.parse_event(event_name="item_equip")

    #for i in range(max(round_ends)):
    #    print(parser.parser.parse_ticks(wanted_props=["active_weapon_name"], ticks=[i]))

    #players:list[Player] = []
    #for i in range(0,10):
    #    players.append(Player(name=f"player{i}", enums_path = "enums.json"))

    #for z in range(0,10):
    #    players[z].load_tick_data(start_idx_curr_tick, curr_tick_info, z)

    #count = 0

    #sprint(cv2.__file__)

    for round_num in range(len(parser.rounds)): #loops for every round played

        current_round = round_num + 1
        he_detonate_this_round = parser.events["hegrenade_detonate"].query('round == @current_round')

        smokes_this_round = parser.smokes.query('round == @current_round')
        fires_this_round = parser.infernos.query('round == @current_round')

        print(smokes_this_round)
        print(fires_this_round)

        #print(min(he_detonate_this_round.index), max(he_detonate_this_round.index))

        print("ROUND " + str(round_num+1))

        first_tick_of_round = int(round_starts[round_num])
        start_tick_round_index = parser.ticks.query('tick == @first_tick_of_round').head(1).index[0] #query takes a long time and dont want to do it for every tick, index of data frame from index of tick 
        print("FIRST TICK: ", first_tick_of_round)
            
        #players:list[Player] = []
        #for i in range(0,10):
        #    players.append(Player(name=f"player{i}", enums_path = "enums.json"))

        video_writer = cv2.VideoWriter(OUTPUT_PATH+"Round_" + str(round_num+1) + ".avi", FOURCC, FPS, FRAME_SIZE, True)
            
        """steamid_list = []
        for i in range(100):
            repeat = False
            for x in steamid_list:
                if(parser.ticks.steamid.loc[i] == x):
                    repeat = True
                    break
            if(repeat == False):
                steamid_list.append(parser.ticks.steamid.loc[i])
                print("steamid: ",parser.ticks.steamid.loc[i], " team: ", parser.ticks.team_name.loc[i])

        print("num uni: ", len(steamid_list))
        print("unique ids: ", steamid_list)"""

        #for i in range(len(equip_list[0])):
            #if(i==0):
            #print("TICKLIST: ",equip_list[0][i], " first_tick_of_round: ", first_tick_of_round)
            #print(type(first_tick_of_round))
            #if(equip_list[0][i] == int(first_tick_of_round)):
            #    print("MATCH")
            #equip_first_index = 

        steamID_to_array = {}
        unique_ids = []
        unique_player_ids = []
        unique_player_names = []
        unique_nonplayer_ids = []
        num_players = 0
        num_nonplayers = 0
        for y in range(len(range(round_starts[round_num], round_ends[round_num] + 1))):
            repeat = False
            for player in unique_ids:
                if(parser.ticks.steamid.loc[y+start_tick_round_index] == player):
                    repeat = True
                    break
            if(repeat == False):
                
                unique_ids.append(parser.ticks.steamid.loc[y+start_tick_round_index])
                if(parser.ticks.team_name.loc[y+start_tick_round_index] == None):
                    unique_nonplayer_ids.append(parser.ticks.steamid.loc[y+start_tick_round_index])
                    num_nonplayers += 1
                else:
                    unique_player_ids.append(parser.ticks.steamid.loc[y+start_tick_round_index])
                    unique_player_names.append(parser.ticks.name.loc[y+start_tick_round_index])
                    steamID_to_array[parser.ticks.steamid.loc[y+start_tick_round_index]] = num_players
                    num_players += 1
                #print("steamid: ",parser.ticks.steamid.loc[y+start_tick_round_index], " team: ", parser.ticks.team_name.loc[y+start_tick_round_index])

        #print("num uni: ", len(unique_player_ids))
        #print("unique ids: ", unique_player_ids)

        #print(steamID_to_array)
        total_users = num_players + num_nonplayers

        players:list[Player] = []
        players_equipped = []
        player_has_defuser = []

        end_of_round = round_ends[round_num]
        #print(equipped_items_all)
        equips_this_round = equipped_items_all.query("tick <= @end_of_round")
        equipped_items_all= equipped_items_all[len(equips_this_round):]
        #print(equipped_items_all)
        players_equip_time = []
        player_equip_weapon = []
        player_equip_tick = []
        for i in range(0,num_players):
            players.append(Player(name=f"player{i}", enums_path = "enums.json", steam_id=unique_player_ids[i]))
            players_equipped.append([])
            player_has_defuser.append([])
            current_steamid = unique_player_ids[i]
            players_equip_time.append(equips_this_round.query("user_steamid == @current_steamid"))
            player_equip_weapon.append(players_equip_time[i]["item"].tolist())
            player_equip_tick.append(players_equip_time[i]["tick"].tolist())
            #equipped_items_all.query("tick <= end_of_round")

        round_dict = parser.rounds

        bomb_plant_time = None

        if str(round_dict['bomb_plant'][round_num]) != "<NA>":
            bomb_plant_time = int(round_dict['bomb_plant'][round_num]) - first_tick_of_round

        bomb_pos = []

        most_updated_tick = 0

        for y in range(len(range(round_starts[round_num], round_ends[round_num] + 1))): #loops for every tick in that round
            list_pos = []
            list_set = []

            start_idx_curr_tick = start_tick_round_index+(y*(total_users))

            # if y % 10 == 0:
            #     with open("tick_info.txt", "a") as f:
            #         f.write(str(parser.ticks.head(n=start_idx_curr_tick+10)))
            curr_tick_info = parser.ticks.loc[start_idx_curr_tick : start_idx_curr_tick+total_users-1] #get 1 dataframe per unique user

            """steamID_to_array = {}
            for i in range(len(curr_tick_info)):
                #print(curr_tick_info.steamid.loc[i+start_idx_curr_tick])
                steamID_to_array[curr_tick_info.steamid.loc[i+start_idx_curr_tick]] = i"""


            if(curr_tick_info.empty):
                for z in range(0, total_users):
                    
                    if(prev_curr_tick_info.team_name.loc[z+start_idx_curr_tick-(y*(total_users))] != None):
                        players[steamID_to_array[curr_tick_info.steamid.loc[z+start_idx_curr_tick-total_users]]].load_tick_data(prev_start_idx_curr_tick, prev_curr_tick_info, z)#steamID_to_array[curr_tick_info.steamid.loc[z+start_idx_curr_tick-(y*(total_users))]])
            else:
                # Ten players in game
                for z in range(0, total_users):
                    #try:
                    #    print("id: ", curr_tick_info.steamid.loc[z+start_idx_curr_tick], " num: ", steamID_to_array[curr_tick_info.steamid.loc[z+start_idx_curr_tick]])
                    #except:
                    #    print(curr_tick_info.steamid.loc[z+start_idx_curr_tick])
                    #    exit()
                    if(curr_tick_info.team_name.loc[z+start_idx_curr_tick] != None):
                        players[steamID_to_array[curr_tick_info.steamid.loc[z+start_idx_curr_tick]]].load_tick_data(start_idx_curr_tick, curr_tick_info, z)#steamID_to_array[curr_tick_info.steamid.loc[z+start_idx_curr_tick]])
                        #print(parser.parser.parse_player_info())
                        #player = [str(76561198799340122)]#[unique_player_ids[1]]#unique_player_ids[0]user_steamid
                        #parser.parser.parse_event(event_name="item_equip")#, player=player))#,player=[unique_player_names[0]]))#, players=unique_player_ids[0]))#,unique_player_ids[0],0)["item"])#.columns)#, ticks=[0,1000])) gets held weapon
                        #tick and item
                        prev_start_idx_curr_tick, prev_curr_tick_info= start_idx_curr_tick, curr_tick_info
                    #print(z, "TEST ", round_num)


            for z in range(0, num_players):
                if(math.isnan(players[z].position[y][0]) or math.isnan(players[z].position[y][1])):
                    if(y!=0):
                        players[z].position[y] = players[z].position[y-1]
                    else:
                        players[z].position[y] = (0,0,0)
                    #players[z].load_tick_data(prev_start_idx_curr_tick, prev_curr_tick_info, z)


            defuser_updated = False

            for i in range(len(defuser_pickups["tick"])):
                if(defuser_pickups["tick"][i+defuser_pickups_index] <= y+first_tick_of_round):
                    player_has_defuser[steamID_to_array[defuser_pickups["user_steamid"][i+defuser_pickups_index]]].append(True)
                    defuser_updated = True
                else:
                    defuser_pickups = defuser_pickups[i:]
                    if(defuser_updated):
                        defuser_pickups_index += i
                    break

            for i in range(num_players):
                while(len(player_has_defuser[i]) <= y):
                    if(len(player_has_defuser[i]) == 0):
                        player_has_defuser[i].append(False)
                    elif(players[z].health == 0):
                        player_has_defuser[i].append(False)
                    else:
                        player_has_defuser[i].append(player_has_defuser[i][-1])



            bomb_updated = False

            for i in range(len(bomb_pickups["tick"])):
                #print(type(bomb_pickups))
                if(bomb_pickups["tick"][i+bomb_pickups_index] <= y+first_tick_of_round):
                    if(bomb_pickups["tick"][i+bomb_pickups_index] > most_updated_tick):
                        current_bomb_holder = steamID_to_array[bomb_pickups["user_steamid"][i+bomb_pickups_index]]
                        most_updated_tick = bomb_pickups["tick"][i+bomb_pickups_index]

                    bomb_updated = True
                else:
                    bomb_pickups = bomb_pickups[i:]
                    if(bomb_updated):
                        bomb_pickups_index += i
                    break

            
            bomb_updated = False

            #print(bomb_drops["tick"])

            for i in range(len(bomb_drops["tick"])):
                if(bomb_drops["tick"][i+bomb_drops_index] <= y+first_tick_of_round):
                    if(bomb_drops["tick"][i+bomb_drops_index] > most_updated_tick):
                        current_bomb_holder = None
                        most_updated_tick = bomb_drops["tick"][i+bomb_drops_index]

                    bomb_updated = True
                else:
                    bomb_drops = bomb_drops[i:]
                    if(bomb_updated):
                        bomb_drops_index += i
                    break

            
            #print(bomb_plants)
            #print(bomb_plants_index,"!")

            bomb_updated = False

            for i in range(len(bomb_plants["tick"])):
                if(bomb_plants["tick"][i+bomb_plants_index] <= y+first_tick_of_round):
                    if(bomb_plants["tick"][i+bomb_plants_index] > most_updated_tick):
                        current_bomb_holder = None
                        most_updated_tick = bomb_plants["tick"][i+bomb_plants_index]
                        #bomb_updated = True
                    bomb_updated = True
                else:
                    bomb_plants = bomb_plants[i:]
                    if(bomb_updated):
                        bomb_plants_index += i
                    break


            if(current_bomb_holder != None):
                bomb_pos.append(players[current_bomb_holder].position[y])
            else:
                if(y==0):
                    bomb_pos.append((1136.0, 32.0, -164.78845))
                else:      
                    bomb_pos.append(bomb_pos[-1])


            #current_tick_weapons2 = parser.parser.parse_ticks(wanted_props=["active_weapon_name"], ticks=[y+first_tick_of_round])
            #for i in range(len(current_tick_weapons2)):
            #    if(str(current_tick_weapons2["steamid"][i]) == "76561198799340122"): #"76561198160709585"):
            #        print(current_tick_weapons2["active_weapon_name"][i])


            frame = cv2.imread(ASSETS_PATH+"de_mirage.png")

            for i in range(min(all_grenades.index), max(all_grenades.index)):
                if(all_grenades["tick"][i] >= y+first_tick_of_round):
                    all_grenades = all_grenades[i-min(all_grenades.index):]
                    break

            #print(all_grenades.columns)
            draw_nades(frame, all_grenades, y, first_tick_of_round, he_detonate_this_round, smokes_this_round, fires_this_round)

            current_tick_weapons = all_weapons.query('tick == @y+@first_tick_of_round')
            weapon_list = current_tick_weapons["active_weapon_name"].to_list()
            id_list = current_tick_weapons["steamid"].to_list()
            #print(current_tick_weapons)
            for z in range(0, num_players):
                
                #print(player_equip_tick[z])
                #print(len(player_equip_tick[z]), len(player_equip_weapon[z]))
                while (len(player_equip_tick[z]) > 1 and player_equip_tick[z][1] <= y+first_tick_of_round):
                    player_equip_tick[z] = player_equip_tick[z][1:]
                    player_equip_weapon[z] = player_equip_weapon[z][1:]
                    #for i in range(len(players_equip_time[i])):
                    #    if(players_equip_time[z]["tick"][i] < y+first_tick_of_round):

                
                for i in range(len(current_tick_weapons)):
                    if(str(id_list[i]) == players[z].steam_id):
                        if(weapon_list[i] == None):
                            if(len(player_equip_weapon[z]) != 0):
                                players_equipped[z].append(player_equip_weapon[z][0])
                            else:
                                players_equipped[z].append(players_equipped[z][-1])
                            #players_equipped[z].append("c4")
                            #for j in range(len(equip_list[0]))
                        else:
                            players_equipped[z].append(weapon_translate[weapon_list[i]])
                        break

                """update_found = 0
                for i in range(len(equip_list[0])):
                    #print(type(equip_list[0][i]), type(equip_list[1][i]), type(players[z].steam_id))
                    #print(int(equip_list[1][i]), " ", players[z].steam_id)

                    #print(equip_list[0][i], " ", y)
                    if((equip_list[0][i] == y+first_tick_of_round) and (equip_list[1][i] == players[z].steam_id)):
                        if(equip_list[2][i] == "deagle" or equip_list[2][i] == "m4a1" or equip_list[2][i] == "hkp2000" or equip_list[2][i] == "mp7"):
                            current_tick_weapons = parser.parser.parse_ticks(wanted_props=["active_weapon_name"], ticks=[y+first_tick_of_round])
                            for i in range(len(current_tick_weapons)):
                                if(str(current_tick_weapons["steamid"][i]) == players[z].steam_id):
                                    if(current_tick_weapons["active_weapon_name"][i] != None):
                                        players_equipped[z].append(weapon_translate[current_tick_weapons["active_weapon_name"][i]])
                                    else:
                                        players_equipped[z].append(equip_list[2][i])
                                    break
                        else:
                            players_equipped[z].append(equip_list[2][i])
                        equip_list[0].pop(i)
                        equip_list[1].pop(i)
                        equip_list[2].pop(i)
                        update_found = 1
                        break
                    
                    if(equip_list[0][i] > y+first_tick_of_round):
                        break

                if(update_found == 0 and not (y == 0) ):
                    players_equipped[z].append(players_equipped[z][-1])
                elif(update_found == 0 and (y == 0)):
                    current_tick_weapons = parser.parser.parse_ticks(wanted_props=["active_weapon_name"], ticks=[y+first_tick_of_round])
                    current_weapon = None
                    for i in range(len(current_tick_weapons)):
                        #print(type(current_tick_weapons["steamid"][i]))
                        #print(type(players[z].steam_id))

                        #print(current_tick_weapons["steamid"])
                        if(current_tick_weapons["active_weapon_name"][i] == None and str(current_tick_weapons["steamid"][i]) == players[z].steam_id):
                            break

                        if(str(current_tick_weapons["steamid"][i]) == players[z].steam_id):
                            current_weapon = weapon_translate[current_tick_weapons["active_weapon_name"][i]]
                            break

                    #print(current_tick_weapons["active_weapon_name"][i])

                    if(current_weapon == None):
                        current_weapon = "knife"


                    #print(current_weapon)

                    players_equipped[z].append(current_weapon)"""

                #if(players_equipped[z][-1] == "hkp2000"):


                #for i in range(len(equip_list[0])):
                #print(i)
                #if(equip_list[0][i] == y and equip_list[1][i] == players[i].steam_id):
                #    players_equipped[z].append(equip_list[2][i])
                #    break
                #players_equiped[steamID_to_array[curr_tick_info.steamid.loc[z+start_idx_curr_tick]]].append()

                draw_player(players[z], y, frame, players_equipped[z], first_tick_of_round, parser.parser, current_bomb_holder==z, player_has_defuser[z][y])
                #pos = (players[z].position[y][0],players[z].position[y][1],players[z].position[y][2])
                #frame = cv2.imread(ASSETS_PATH+"de_mirage.png")
                #cv2.circle(frame, (int(translate_position(pos[0], "x")), int(translate_position(pos[1], "y"))), 8, (0,0,255), -1)


            #for weap in tick_data.inventory.loc[tick_idx+z]:
            #bomb_updated = False

            #print("LEN: ", len(bomb_pickups[["tick"]]))
            #print(bomb_pickups["tick"])
            #print(bomb_pickups_index)



            #print(bomb_pos)



            #player_has_bomb = False
            #for i in range(len(players)):
            #    if(players[i].HasBomb[y]):
            #        bomb_pos.append(players[i].position[y])
            #        player_has_bomb = True
            #        break
        
            #if(not player_has_bomb):
            #    if(y==0):
            #        bomb_pos.append((1136.0, 32.0, -164.78845))
            #    else:
            #        bomb_pos.append(bomb_pos[-1])

            #print(bomb_pos.append(bomb_pos[-1]))

            draw_round_details(frame, y, bomb_pos, bomb_plant_time, current_bomb_holder != None, 0.6)

            video_writer.write(frame)
                #print("TEAM: ", players[z].team_name)
                #print("NUM: ", players[z].player_name)
                #list_pos.append(pos)
                #list_set.append({"size":6.0, "color": (29/255,204/255,247/255) if(players[z].team_name == "CT") else (253/255,219/255,36/255)})
            #if(count%480==0):
                #dict_list.append({"points":list_pos, "point_settings":list_set})
                #f.write(str(players[z].position[y][0]) + str(players[z].position[y][1]) + str(players[z].position[y][2]) + "\n")
            #count += 1
        #fig, ax = plot.plot(map_name, list_pos)
        video_writer.release()

    #f.close()
    

    #settings = {
    #    "size" : 6.0
    #}

    #size = 6.0
    #list_set = []
    #list_set.append(settings)

    #dict = {
    #    "points" : list_pos,#list_pos[0:0],
    #    "point_settings" : list_set#[{"size":6.0}]#list_set
    #}

    #list_dict = []
    #list_dict.append(dict)
    #print(parser.ticks)
    #Tuple = (0,0,0)

    #list = []


    #list.append(Tuple)


    #plot.gif(map_name=map_name, frames_data=dict_list, output_filename="game.gif", duration=4000)

    #fig, ax = plot.plot(map_name, list_pos)

    #ax.set_title("Simple Plot")

    #ani = animation.FuncAnimation(fig=fig, func=update, frames=1, interval=16)
    #plt.show()

    #def update(frame):


MIRAGE_DATA ={
        "pos_x": -3230,
        "pos_y": 1713,
        "scale": 5,
}

#position from awpy, axis being "x" or "y"
def translate_position(position, axis):
    axis = axis.lower()
    if axis not in ["x", "y"]:
        msg = f"'axis' has to be 'x' or 'y', not {axis}"
        raise ValueError(msg)
    start = MIRAGE_DATA["pos_" + axis]
    scale = MIRAGE_DATA["scale"]

    if axis == "x":
        return (position - start) / scale
    return (start - position) / scale


def overlay_image(frame, image_path, coordinates, opacity, resize):
    #background = cv2.imread('field.jpg')
    to_be_combined = frame.copy()
    overlay = cv2.imread(ASSETS_PATH + image_path, cv2.IMREAD_UNCHANGED)  # IMREAD_UNCHANGED => open image with the alpha channel
    overlay = cv2.resize(overlay, (int(overlay.shape[1]*resize), int(overlay.shape[0]*resize)))

    # separate the alpha channel from the color channels
    alpha_channel = overlay[:, :, 3] / 255 # convert from 0-255 to 0.0-1.0
    overlay_colors = overlay[:, :, :3]

    # To take advantage of the speed of numpy and apply transformations to the entire image with a single operation
    # the arrays need to be the same shape. However, the shapes currently looks like this:
    #    - overlay_colors shape:(width, height, 3)  3 color values for each pixel, (red, green, blue)
    #    - alpha_channel  shape:(width, height, 1)  1 single alpha value for each pixel
    # We will construct an alpha_mask that has the same shape as the overlay_colors by duplicate the alpha channel
    # for each color so there is a 1:1 alpha channel for each color channel
    alpha_mask = alpha_channel[:, :, np.newaxis]

    # The background image is larger than the overlay so we'll take a subsection of the background that matches the
    # dimensions of the overlay.
    # NOTE: For simplicity, the overlay is applied to the top-left corner of the background(0,0). An x and y offset
    # could be used to place the overlay at any position on the background.
    h, w = overlay.shape[:2]
    background_subsection = to_be_combined[coordinates[1]:h+coordinates[1], coordinates[0]:w+coordinates[0]]

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + overlay_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    to_be_combined[coordinates[1]:h+coordinates[1], coordinates[0]:w+coordinates[0]] = composite

    cv2.addWeighted(to_be_combined, opacity, frame, 1-opacity, 0, frame)

    #cv2.imwrite('combined.png', background)

    """#blank_frame = np.zeros((FRAME_SIZE[0], FRAME_SIZE[1], 3), dtype=np.uint8)
    #if(type(image_path) == )
    overlay = cv2.imread(ASSETS_PATH + image_path, cv2.IMREAD_UNCHANGED)
    overlay = cv2.resize(overlay, (int(overlay.shape[1]*resize), int(overlay.shape[0]*resize)))
    overlay_no_alpha = cv2.imread(ASSETS_PATH + image_path)
    overlay_no_alpha = cv2.resize(overlay_no_alpha, (int(overlay_no_alpha.shape[1]*resize), int(overlay_no_alpha.shape[0]*resize)))

    b, g, r, a = cv2.split(overlay)

    for i in range(overlay.shape[0]):
            for j in range(overlay.shape[1]):
                #print(a[i][j])
                if(a[i][j] > ALPHA_THRESH):
                    #this is a mess
                    frame[coordinates[1] + i, coordinates[0] + j] = (math.floor(overlay_no_alpha[i, j][0]*opacity*(a[i][j]/255) + frame[coordinates[1] + i, coordinates[0] + j][0]*(1-(opacity*(a[i][j]/255)))), math.floor(overlay_no_alpha[i, j][1]*opacity*(a[i][j]/255) + frame[coordinates[1] + i, coordinates[0] + j][1]*(1-(opacity*(a[i][j]/255)))), math.floor(overlay_no_alpha[i, j][2]*opacity*(a[i][j]/255) + frame[coordinates[1] + i, coordinates[0] + j][2]*(1-(opacity*(a[i][j]/255)))))

    #cv2.addWeighted(frame, 1, blank_frame, opacity, 0.0, frame)"""

#player is player object, tick is tick of round, frame is the frame to draw on
def draw_player(player, tick, frame, player_equipped, first_tick_of_round, demoparser2, hasBomb, hasDefuser):

    pos_x = int(translate_position(player.position[tick][0], "x"))
    pos_y = int(translate_position(player.position[tick][1], "y"))
    player_size = round(((player.position[tick][2]+370)/79) + 10)
    #player_size_offset = round((2*player_size)/3)

    player_color = CT_COLOR if(player.team_name == "CT") else T_COLOR

    if(player.health[tick] == 0):
        player_size_offset = round((2*player_size)/3)
        overlay = frame.copy()
        cv2.line(overlay, (pos_x-player_size_offset, pos_y-player_size_offset), (pos_x+player_size_offset, pos_y+player_size_offset), player_color, 5)
        cv2.line(overlay, (pos_x-player_size_offset, pos_y+player_size_offset), (pos_x+player_size_offset, pos_y-player_size_offset), player_color, 5)
        cv2.addWeighted(frame, 0.5, overlay, 0.5, 0, frame)

    else:
        overlay = frame.copy()
        cv2.ellipse(overlay, (pos_x, pos_y), (60,60), 0, -player.yaw[tick]-45, -player.yaw[tick]+45, (255,255,255), -1)
        cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)
        #cv2.putText(frame, "DEBUG:"+str(player.yaw[tick]), (pos_x, pos_y-50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2, cv2.LINE_8)
        if(player.Flash[tick] == 1):
            overlay_image(frame, "flashed.png", (pos_x-int(player_size*1.6), pos_y-int(player_size*1.9)), 1, player_size/38)
        cv2.circle(frame, (pos_x, pos_y), player_size, player_color, -1)
        if(player.HasHelmet[tick] == 1):
            overlay_image(frame, "head_armor.png", (pos_x-round(player_size*0.607), pos_y-round(player_size*0.607)), 0.8, (player_size/18)+0.2)
        elif(player.HasArmor[tick] == 1):
            overlay_image(frame, "armor.png", (pos_x-round(player_size*0.607), pos_y-round(player_size*0.607)), 0.8, (player_size/18)+0.2)

        cv2.rectangle(frame, (pos_x-20, pos_y-12), (pos_x-15, pos_y+13), (0,0,255), -1)
        cv2.rectangle(frame, (pos_x-20, pos_y+13-math.ceil(player.health[tick]/4)), (pos_x-15, pos_y+13), (0,255,0), -1)

        held_item = player_equipped[tick]
        #held_item = "knife"
        #for i in range(len(equip_list[0])):
           #print(i)
        #    if(equip_list[0][i] == tick and equip_list[1][i]):
        #        held_item = equip_list[2][i]
        #        break       
        
        if(held_item == "knife"):
            held_item = "knife_ct" if(player.team_name == "CT") else "knife_t"
        #elif(held_item == "deagle" or held_item == "m4a1" or held_item == "hkp2000" or held_item == "mp7"):
        #    current_tick_weapons = demoparser2.parse_ticks(wanted_props=["active_weapon_name"], ticks=[tick+first_tick_of_round])
        #    for i in range(len(current_tick_weapons)):
        #        if(str(current_tick_weapons["steamid"][i]) == player.steam_id):
        #            held_item = weapon_translate[current_tick_weapons["active_weapon_name"][i]]
        #            break

        #    print("deag")
            #check for deag vs r8
        #elif(held_item == "m4a1"):
            #check for a4 vs a1s
        #    print("m4")
        #elif(held_item == "hkp2000"):
        #    print("p2k")
            #check for p2k vs usps
        #elif(held_item == "mp7"):
        #    print("mp7")
            #check for mp7 vs mp5sd

        #TODO some weapons are considered the same
        #deag & r8
        #a1 & a4
        #p2k & usp
        #mp7 & mp5sd

        #print("HELD: ", held_item)
        #print(type(equip_list[0]))
        #print(player_equipped)
        try:
            overlay_image(frame, "weapons/"+held_item+".png", (pos_x+int(player_size*0.707), pos_y-int(player_size*0.707)-10), 1.0, 0.5)
        except:
            print("error at player ", player.steam_id, " with weapon ", held_item)
            exit()


        if(hasDefuser):
            overlay_image(frame, "defuse_kit.png", (pos_x+int(player_size*0.707)-10 , pos_y+int(player_size*0.707)-10), 1.0, 0.5)
        elif(hasBomb):
            overlay_image(frame, "bomb_icon.png", (pos_x+int(player_size*0.707)-5, pos_y+int(player_size*0.707)-5), 1.0, 0.6)

        #if(player.HasBomb[tick] == 1):
        #    overlay_image(frame)
            
            #blank_frame = np.zeros((FRAME_SIZE[0], FRAME_SIZE[1], 3), dtype=np.uint8)

            #armor = cv2.imread(ASSETS_PATH + "head_armor.png", cv2.IMREAD_UNCHANGED)
            #armor_no_alpha = cv2.imread(ASSETS_PATH + "head_armor.png")

            #b, g, r, a = cv2.split(armor)

            #print("ALPHA: ",type(a))

            #for i in range(armor.shape[0]):
            #    for j in range(armor.shape[1]):
            #        if(a[i][j] > 0):
            #            blank_frame[pos_y + i, pos_x + j] = armor_no_alpha[i, j]

            #blank_frame = 
            
            #armor = cv2.resize(armor, FRAME_SIZE)
            #cv2.addWeighted(frame, 1.0, blank_frame, 0.5, 0.0, frame)
            

            #frame = cv2.add(frame, armor)

def draw_round_details(frame, tick, bomb_position, plant_time, player_has_bomb, model_predict):
    #print("plant time: ", plant_time)
    if(not player_has_bomb):
        overlay_image(frame, "bomb_icon.png", (int(translate_position(bomb_position[tick][0], "x"))-15, int(translate_position(bomb_position[tick][1], "y"))-16), 1.0, 0.8)#(bomb_position[3]+370/400)+0.2))
    if(plant_time == None):
        plant_time = 100000

    if(plant_time>tick):
        round_timer = 115 - math.floor(tick/64)
        timer_display = str(math.floor(round_timer/60))+":"+str(round_timer%60)
        #print(len(timer_display))
        if(len(timer_display) == 3):
            timer_display = str(math.floor(round_timer/60))+":0"+str(round_timer%60)
        cv2.putText(frame, timer_display, (450,930), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255), 4, cv2.LINE_8)
    else:
        bomb_timer = 40 - math.floor((tick-plant_time)/64)
        if(bomb_timer < 10):
            if(bomb_timer < 0):
                bomb_timer = 0
            timer_display = "0:0"+str(bomb_timer)
        else:
            timer_display = "0:"+str(bomb_timer)
        cv2.putText(frame, timer_display, (450,930), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,255), 4, cv2.LINE_8)
        overlay_image(frame, "bomb_tick.png", (int(translate_position(bomb_position[tick][0], "x"))-15, int(translate_position(bomb_position[tick][1], "y"))-25), 1.0, 1.3)
    

    cv2.rectangle(frame, (212, 50), (212+round(600*model_predict), 100), CT_COLOR, -1)
    cv2.rectangle(frame, (212+round(600*model_predict), 50), (812, 100), T_COLOR, -1)
    overlay_image(frame, "CT_icon.png", (137,45), 1, 1)
    overlay_image(frame, "T_icon.png", (822,45), 1, 1)
    cv2.putText(frame, f'{100*model_predict: .1f}', (192,130), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2, cv2.LINE_8)
    cv2.putText(frame, f'{100*(1-model_predict): .1f}', (730,130), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2, cv2.LINE_8)

def draw_nades(frame, grenades, tick, first_tick_of_round, he_detonate_this_round, smokes_this_round, fires_this_round):
    
    for i in range(min(grenades.index),max(grenades.index)):

        if(grenades["tick"][i] == tick+first_tick_of_round and str(grenades["X"][i]) != "<NA>" and str(grenades["Y"][i]) != "<NA>"):
            if(grenades["grenade_type"][i] == "he_grenade"):
                for j in range(min(he_detonate_this_round.index), max(he_detonate_this_round.index)+1):
                    #print("j: ", j)
                    if(grenades["entity_id"][i] == he_detonate_this_round["entityid"][j]):
                        if(tick+first_tick_of_round < he_detonate_this_round["tick"][j]):
                            overlay_image(frame, "weapons/he_grenade.png", (round(translate_position(grenades["X"][i], "x")-4) ,round(translate_position(grenades["Y"][i], "y"))-9), 1, 0.5)
                        #print("test") #check if HE has exploded yet
                        else:
                            break
            else:
                overlay_image(frame, "weapons/"+grenades["grenade_type"][i]+".png", (round(translate_position(grenades["X"][i], "x")-4) ,round(translate_position(grenades["Y"][i], "y"))-9), 1, 0.5)
        elif(grenades["tick"][i] > tick+first_tick_of_round):
           break
    
    #print(len(smokes_this_round))
    if(len(smokes_this_round) > 0):
        for i in range(min(smokes_this_round.index), max(smokes_this_round.index)+1):
            #print(smokes_this_round["start_tick"][i], smokes_this_round["end_tick"][i], tick+first_tick_of_round)
            if(smokes_this_round["start_tick"][i] <= tick+first_tick_of_round and smokes_this_round["end_tick"][i] >= tick+first_tick_of_round):
                overlay_image(frame, "cloud.png", (round(translate_position(smokes_this_round["X"][i], "x"))-29 ,round(translate_position(smokes_this_round["Y"][i], "y"))-20), 1, 1.8)

            if(smokes_this_round["start_tick"][i] > tick+first_tick_of_round):
                break
    
    if(len(fires_this_round) > 0):
        for i in range(min(fires_this_round.index), max(fires_this_round.index)+1):
            if(fires_this_round["start_tick"][i] <= tick+first_tick_of_round and fires_this_round["end_tick"][i] >= tick+first_tick_of_round):
                overlay_image(frame, "fire.png", (round(translate_position(fires_this_round["X"][i], "x"))-24 ,round(translate_position(fires_this_round["Y"][i], "y"))-24), 1, 1.5)

            if(fires_this_round["start_tick"][i] > tick+first_tick_of_round):
                break

main()