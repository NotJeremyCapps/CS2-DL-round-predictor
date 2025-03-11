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
from round import Round
#import sys
import csv
import cv2
import numpy as np
import os
from dataset import CS2PredictionDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
# CPU only, would be nice to have a GPU version

ROUNDS_PATH = "../game_demos/visualizer"

PATH = "../game_demos/"
#g2-vs-heroic-m3-mirage.dem
#00nation-vs-eclot-m1-mirage.dem
DEMO_NAME = "heroic-vs-nemiga-m2-mirage.dem"#"g2-vs-heroic-m3-mirage.dem"#"replay.dem"

OUTPUT_PATH = "../parsed_videos/"

ASSETS_PATH = OUTPUT_PATH + "assets/"

MODEL_PATH = "trained_model.pt"

CT_COLOR = (247, 204, 29)
T_COLOR = (36, 219, 253)

ALPHA_THRESH = 0

#fps = 60

FRAME_SIZE = (1024, 1024)

FPS = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FOURCC = cv2.VideoWriter_fourcc(*'XVID')

weapon_translate_file = open("weapon_translate.json", 'r')
weapon_translate = json.loads(weapon_translate_file.read())
weapon_translate_file.close()

def main():

    parser = Demo(PATH + DEMO_NAME)



    #model = CS2LSTM(n_feature=None, out_feature=1,n_hidden=60 ,n_layers=2)
    #model.load_state_dict(torch.load(MODEL_PATH, weights_only=False))
    model = torch.load(MODEL_PATH, weights_only=False)
    model.eval()

    #hidden = model.init_hidden(1) #initialize hidden variable
    #model.eval()

    #for param in model.parameters():
    #    print(param)

    round_starts = (parser.rounds["freeze_end"]) #this is the ticks for end of freeze time at start of rounds
    round_ends = (parser.rounds["end"]) #does not include time between round determination and respawn



    all_weapons = parser.parser.parse_ticks(wanted_props=["active_weapon_name"])

    equipped_items_all = parser.parser.parse_event(event_name="item_equip")


    defuser_pickups = parser.parser.parse_event(event_name="item_pickup").query("item == 'defuser'").sort_index(ignore_index=True)

    defuser_pickups_index = 0

    bomb_pickups = parser.parser.parse_event(event_name="bomb_pickup")
    bomb_drops = parser.parser.parse_event(event_name="bomb_dropped")
    bomb_plants = parser.parser.parse_event(event_name="bomb_planted")

    bomb_pickups_index = 0
    bomb_drops_index = 0
    bomb_plants_index = 0

    all_grenades = parser.grenades

    #print(all_grenades.columns)

    has_defuser_at_end = []


    for round_num in range(len(parser.rounds)): #loops for every round played

        #if(round_num == 0):
        #    round_num = 12

        #print("ROUND: ", round_num)


        current_round = round_num + 1
        he_detonate_this_round = parser.events["hegrenade_detonate"].query('round == @current_round')

        smokes_this_round = parser.smokes.query('round == @current_round')
        fires_this_round = parser.infernos.query('round == @current_round')

        #print(smokes_this_round)
        #print(fires_this_round)

        #print(min(he_detonate_this_round.index), max(he_detonate_this_round.index))

        #print("ROUND " + str(round_num+1))

        first_tick_of_round = int(round_starts[round_num])
        start_tick_round_index = parser.ticks.query('tick == @first_tick_of_round').head(1).index[0] #query takes a long time and dont want to do it for every tick, index of data frame from index of tick 
        #print("FIRST TICK: ", first_tick_of_round)
            
        #players:list[Player] = []
        #for i in range(0,10):
        #    players.append(Player(name=f"player{i}", enums_path = "enums.json"))

        video_writer = cv2.VideoWriter(OUTPUT_PATH+"Round_" + str(round_num+1) + ".avi", FOURCC, FPS, FRAME_SIZE, True)
            

        steamID_to_array = {}
        unique_ids = []
        unique_player_ids = []
        unique_player_names = []
        unique_nonplayer_ids = []
        num_players = 0
        num_nonplayers = 0
        CT_counter = 0
        T_counter = 5
        for y in range(len(range(round_starts[round_num], round_ends[round_num] + 1))):
            repeat = False
            for player in unique_ids:
                if(parser.ticks.steamid.loc[y+start_tick_round_index] == player):
                    repeat = True
                    break
            if(repeat == False):
                
                unique_ids.append(parser.ticks.steamid.loc[y+start_tick_round_index])
                #print(parser.ticks.team_name.loc[y+start_tick_round_index+256])
                if(parser.ticks.team_name.loc[y+start_tick_round_index] == None):
                    unique_nonplayer_ids.append(parser.ticks.steamid.loc[y+start_tick_round_index])
                    num_nonplayers += 1
                else:
                    unique_player_ids.append(parser.ticks.steamid.loc[y+start_tick_round_index])
                    unique_player_names.append(parser.ticks.name.loc[y+start_tick_round_index])
                    if(parser.ticks.team_name.loc[y+start_tick_round_index] == "CT"):
                        steamID_to_array[parser.ticks.steamid.loc[y+start_tick_round_index]] = CT_counter
                        CT_counter += 1
                    elif(parser.ticks.team_name.loc[y+start_tick_round_index] == "TERRORIST"):
                        steamID_to_array[parser.ticks.steamid.loc[y+start_tick_round_index]] = T_counter
                        T_counter += 1
                    num_players += 1
                
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
        CT_track = 0
        for i in range(0,num_players):
            #print(parser.ticks.team_name.loc[i+start_tick_round_index])
            if(parser.ticks.team_name.loc[i+start_tick_round_index] == "CT"):
                players.append(Player(name=f"player{CT_track}", enums_path = "enums.json", steam_id=unique_player_ids[i]))
                players_equipped.append([])
                player_has_defuser.append([])
                current_steamid = unique_player_ids[i]
                players_equip_time.append(equips_this_round.query("user_steamid == @current_steamid"))
                player_equip_weapon.append(players_equip_time[CT_track]["item"].tolist())
                player_equip_tick.append(players_equip_time[CT_track]["tick"].tolist())
                CT_track += 1


        T_track = 5
        for i in range(0,num_players):
            #print(parser.ticks.team_name.loc[i+start_tick_round_index])
            if(parser.ticks.team_name.loc[i+start_tick_round_index] == "TERRORIST"):
                players.append(Player(name=f"player{T_track}", enums_path = "enums.json", steam_id=unique_player_ids[i]))
                players_equipped.append([])
                player_has_defuser.append([])
                current_steamid = unique_player_ids[i]
                players_equip_time.append(equips_this_round.query("user_steamid == @current_steamid"))
                player_equip_weapon.append(players_equip_time[T_track]["item"].tolist())
                player_equip_tick.append(players_equip_time[T_track]["tick"].tolist())
                T_track += 1

            #equipped_items_all.query("tick <= end_of_round")






        round_file = Round(round_title="data_for_visualizer", 
                                round_num=round_num,
                                map_name="visualizer",
                                demo_data_root=ROUNDS_PATH, 
                                enums_path="enums.json")
        round_file.init_headers()







        for y in range(len(range(round_starts[round_num], round_ends[round_num] + 1))): #loops for every tick in that round


            start_idx_curr_tick = start_tick_round_index+(y*total_users)

            # if y % 10 == 0:
            #     with open("tick_info.txt", "a") as f:
            #         f.write(str(parser.ticks.head(n=start_idx_curr_tick+10)))




            curr_tick_info = parser.ticks.loc[start_idx_curr_tick : start_idx_curr_tick+total_users-1] #get 1 dataframe per unique user


            if(curr_tick_info.empty):
                for z in range(0, total_users):
                    
                    if(prev_curr_tick_info.team_name.loc[z+start_idx_curr_tick-(total_users)] != None):
                        players[steamID_to_array[prev_curr_tick_info.steamid.loc[z+start_idx_curr_tick-total_users]]].load_tick_data(prev_start_idx_curr_tick, prev_curr_tick_info, z)#steamID_to_array[curr_tick_info.steamid.loc[z+start_idx_curr_tick-(y*(total_users))]])
            else:

                for z in range(0, total_users):
               
                    if(curr_tick_info.team_name.loc[z+start_idx_curr_tick] != None):
                        players[steamID_to_array[curr_tick_info.steamid.loc[z+start_idx_curr_tick]]].load_tick_data(start_idx_curr_tick, curr_tick_info, z)#steamID_to_array[curr_tick_info.steamid.loc[z+start_idx_curr_tick]])
            
                        prev_start_idx_curr_tick, prev_curr_tick_info= start_idx_curr_tick, curr_tick_info
                    #print(z, "TEST ", round_num)


            for z in range(0, num_players):
                if(math.isnan(players[z].position[y][0]) or math.isnan(players[z].position[y][1])):
                    if(y!=0):
                        players[z].position[y] = players[z].position[y-1]
                    else:
                        players[z].position[y] = (0,0,0)
                    #players[z].load_tick_data(prev_start_idx_curr_tick, prev_curr_tick_info, z)


        if(os.path.exists("../game_demos/visualizer/preprocessed/rounds.txt")):
            os.remove("../game_demos/visualizer/preprocessed/rounds.txt")

        if(os.path.exists("../game_demos/visualizer/preprocessed/visualizer/data_for_visualizer.csv")):
            os.remove("../game_demos/visualizer/preprocessed/visualizer/data_for_visualizer.csv")
            
        round_file.load_player_tick_data(players = players)
        round_file.load_round_data(round_dict=parser.rounds)
        round_file.write_round_to_csv()





        round_set = CS2PredictionDataset(list="../game_demos/visualizer/preprocessed/rounds.txt", sequence_length=120)


        test_data = DataLoader(dataset=round_set,
                                     batch_size=1,
                                     num_workers=0,
                                     drop_last=False
                                     )
        




        out_array = []


        hidden = model.init_hidden(1)
        with torch.no_grad(): #stops gradient computation - it is unnecessary
            with tqdm(test_data, unit="batch", leave=True) as tepoch:
                for batch_idx, (target, new_round, x_main_data, x_prim_weap, x_sec_weap) in enumerate(tepoch):

                    print(f"Len of main: {x_main_data.size(1)}") # get length of sequence dimension
                    if x_main_data.size(1) == 0: 
                        continue
                    
                    target, new_round, x_main_data, x_prim_weap, x_sec_weap = target.to(DEVICE), new_round.to(DEVICE), x_main_data.to(DEVICE), x_prim_weap.int().to(DEVICE), x_sec_weap.int().to(DEVICE)

                    # need categorical data as ints for embedding
                    x_prim_weap, x_sec_weap = x_prim_weap.int(), x_sec_weap.int()
                    # with open('batch_data.txt', 'a') as f:
                    #     print(f"Target: {target}, Shape: {target.size()};\n Main_data: {x_main_data}, Shape: {x_main_data.size()};\n Weapon_data: {x_prim_weap}, Shape: {x_prim_weap.size()};\n New_Round: {new_round}, Shape: {new_round.shape};\n\n\n", file = f)
            

                    out, hidden = model(x_main_data, x_prim_weap, x_sec_weap, hidden) #hidden is info from past

                    out_array.append(out.squeeze().item()) # Output comes out of self.model (batch_size, 1) for some reason



        #print("OUT LEN: ",len(out_array))


        #with open("../game_demos/visualizer/preprocessed/visualzer/data_for_visualizer.csv", 'r') as file: #open first csv file as file2 -- need to indicate based on order of csv file
        #        readingcsv = csv.reader(file)
        #        all_data = []
        #        for row in readingcsv: 
        #            all_data.append(row) #first index is row and then col


        #data_weap_ind = 0
        #for index, word in enumerate(self.all_data[0]):
        #    if "primary" in word:
        #        data_weap_ind = index
        #        break
        #    elif "secondary" in word:
        #        data_weap_ind = index
        #        break

    
        #create non prim/second data into floats
        #int_data = []
    
        #for i in range(1, len(self.all_data)):#increment over every row
        #    temp_array = []
        #    for m in range(data_weap_ind): #up to prim/secondary data -THIS 96 IS INCORRECT SHOULD PROB automate
                    #convert to float
        #            temp_array.append(float(self.all_data[i][m])) 
                   




        #int_data.append(temp_array)
            
        #change into tensor
        #data = torch.tensor(int_data) #data index 
        #create prim/secondary weapon data as floats
        #prim_data_weap = []
        #sec_data_weap = []
    
        #for i in range(1, len(all_data)):#increment over every row
        #    prim_array = []
        #    sec_array = []
        #    for m in range(data_weap_ind,len(all_data[0])-1): #from prim/secondary data to win 
                    #convert to float
        #            if(0 == (m-data_weap_ind)%2 ):
         #               prim_array.append(float(all_data[i][m])) 
        #            elif(1 == (m-data_weap_ind)%2):
        #                sec_array.append(float(all_data[i][m]))
                   
        #    prim_data_weap.append(prim_array)
        #    sec_data_weap.append(sec_array)
        
        # with open('batch_data.txt', 'a') as f:
        #     print(f"Loading tensors!! All Data: {self.all_data}, Shape: {len(self.all_data)};\n\n\n", file = f)
            
        #change into tensor
        #prim_data_weap_t = torch.tensor(prim_data_weap) #data index 
        #sec_data_weap_t = torch.tensor(sec_data_weap) #data index 
      








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


            defuser_updated = False

            for i in range(len(defuser_pickups["tick"])):
                if(defuser_pickups["tick"][i+defuser_pickups_index] <= y+first_tick_of_round):
                    #print(unique_ids)
                    #print(steamID_to_array)
                    player_has_defuser[steamID_to_array[defuser_pickups["user_steamid"][i+defuser_pickups_index]]].append(True)
                    defuser_updated = True
                else:
                    defuser_pickups = defuser_pickups[i:]
                    if(defuser_updated):
                        defuser_pickups_index += i
                    break

            for i in range(num_players):
                prev_defuser = False
                while(len(player_has_defuser[i]) <= y):
                    if(len(player_has_defuser[i]) == 0):
                        for j in range(len(has_defuser_at_end)):
                            prev_defuser = False
                            if(has_defuser_at_end[j] == players[i].steam_id and players[i].team_name == "CT"):
                                player_has_defuser[i].append(True)
                                prev_defuser = True
                                break
                        
                        if(prev_defuser == False):
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

               


                draw_player(players[z], y, frame, players_equipped[z], first_tick_of_round, parser.parser, current_bomb_holder==z, player_has_defuser[z][y])
  

            val_to_show = int((y-120)/120)
            if(val_to_show < 0):
                val_to_show = 0

            draw_round_details(frame, y, bomb_pos, bomb_plant_time, current_bomb_holder != None, 1-out_array[val_to_show])

            video_writer.write(frame)

        has_defuser_at_end = []
        for i in range(0, num_players):
            if(player_has_defuser[i][-1] == True):
                has_defuser_at_end.append(players[i].steam_id)
                
        video_writer.release()

    


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
        
        
        if(held_item == "knife"):
            held_item = "knife_ct" if(player.team_name == "CT") else "knife_t"
       

        try:
            overlay_image(frame, "weapons/"+held_item+".png", (pos_x+int(player_size*0.707), pos_y-int(player_size*0.707)-10), 1.0, 0.5)
        except:
            print("error at player ", player.steam_id, " with weapon ", held_item)
            exit()


        if(hasDefuser):
            overlay_image(frame, "defuse_kit.png", (pos_x+int(player_size*0.707)-10 , pos_y+int(player_size*0.707)-10), 1.0, 0.5)
        elif(hasBomb):
            overlay_image(frame, "bomb_icon.png", (pos_x+int(player_size*0.707)-5, pos_y+int(player_size*0.707)-5), 1.0, 0.6)

        

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