from awpy import Demo
from awpy import plot
from player import Player
from round import Round
import math
from typing import Sequence
import json
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

CT_COLOR = (247, 204, 29)
T_COLOR = (36, 219, 253)

ALPHA_THRESH = 0

#fps = 60

FRAME_SIZE = (1024, 1024)

FPS = 60

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

    #SOME DEMOS DO NOT HAVE WARMUP LEADING TO FIRST PISTOL ROUND NOT BEING PARSED!
    round_starts = (parser.rounds["freeze_end"]) #this is the ticks for end of freeze time at start of rounds
    round_ends = (parser.rounds["end"]) #does not include time between round determination and respawn

    equipped_items_all = parser.parser.parse_event(event_name="item_equip")
    equipped_ticks = equipped_items_all["tick"].tolist()
    equipped_steamid = equipped_items_all["user_steamid"].tolist()
    equipped_items = equipped_items_all["item"].tolist()


    #for i in range(len(equipped_items_all)):
    #    print(equipped_items_all["item"][i], " ", equipped_items_all["weptype"][i])
    #equipped_skins = equipped_items_all.columns
    #print(equipped_items_all["defindex"])#.columns)
    #for i in equipped_items_all["hassilencer"]:
    #   print(i)
    equip_list = [equipped_ticks, equipped_steamid, equipped_items]

    #equipped_items.values.tolist()

    #event_df = parser.parser.parse_event(event_name="item_equip")

    #for i in range(max(round_ends)):
    #    print(parser.parser.parse_ticks(wanted_props=["active_weapon_name"], ticks=[i]))

    #players:list[Player] = []
    #for i in range(0,10):
    #    players.append(Player(name=f"player{i}", enums_path = "enums.json"))

    #for z in range(0,10):
    #    players[z].load_tick_data(start_idx_curr_tick, curr_tick_info, z)
    list_pos = []
    list_set = []
    dict_list = []
    #count = 0

    print(cv2.__file__)

    for round_num in range(len(parser.rounds)): #loops for every round played

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

        for i in range(0,num_players):
            players.append(Player(name=f"player{i}", enums_path = "enums.json", steam_id=unique_player_ids[i]))
            players_equipped.append([])

        current_round = Round(round_num=round_num, need_write=False)

        current_round.init_headers()
        
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

            current_round.load_round_data(round_dict=parser.rounds)

            for z in range(0, num_players):
                if(math.isnan(players[z].position[y][0]) or math.isnan(players[z].position[y][1])):
                    if(y!=0):
                        players[z].position[y] = players[z].position[y-1]
                    else:
                        players[z].position[y] = (0,0,0)
                    #players[z].load_tick_data(prev_start_idx_curr_tick, prev_curr_tick_info, z)

            frame = cv2.imread(ASSETS_PATH+"de_mirage.png")
            for z in range(0, num_players):
                
                

                update_found = 0
                for i in range(len(equip_list[0])):
                    #print(type(equip_list[0][i]), type(equip_list[1][i]), type(players[z].steam_id))
                    #print(int(equip_list[1][i]), " ", players[z].steam_id)

                    #print(equip_list[0][i], " ", y)
                    if((equip_list[0][i] == y+first_tick_of_round) and (equip_list[1][i] == players[z].steam_id)):
                        if(equip_list[2][i] == "deagle" or equip_list[2][i] == "m4a1" or equip_list[2][i] == "hkp2000" or equip_list[2][i] == "mp7"):
                            current_tick_weapons = parser.parser.parse_ticks(wanted_props=["active_weapon_name"], ticks=[y+first_tick_of_round])
                            for i in range(len(current_tick_weapons)):
                                if(str(current_tick_weapons["steamid"][i]) == players[z].steam_id):
                                    players_equipped[z].append(weapon_translate[current_tick_weapons["active_weapon_name"][i]])
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

                    players_equipped[z].append(current_weapon)

                #if(players_equipped[z][-1] == "hkp2000"):


                #TODO split equip list for each round, delete entry after usage
                #for i in range(len(equip_list[0])):
                #print(i)
                #if(equip_list[0][i] == y and equip_list[1][i] == players[i].steam_id):
                #    players_equipped[z].append(equip_list[2][i])
                #    break
                #players_equiped[steamID_to_array[curr_tick_info.steamid.loc[z+start_idx_curr_tick]]].append()

                draw_player(players[z], y, frame, players_equipped[z], first_tick_of_round, parser.parser)
                #pos = (players[z].position[y][0],players[z].position[y][1],players[z].position[y][2])
                #frame = cv2.imread(ASSETS_PATH+"de_mirage.png")
                #cv2.circle(frame, (int(translate_position(pos[0], "x")), int(translate_position(pos[1], "y"))), 8, (0,0,255), -1)
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


    #blank_frame = np.zeros((FRAME_SIZE[0], FRAME_SIZE[1], 3), dtype=np.uint8)

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

    #cv2.addWeighted(frame, 1, blank_frame, opacity, 0.0, frame)

#player is player object, tick is tick of round, frame is the frame to draw on
def draw_player(player, tick, frame, player_equipped, first_tick_of_round, demoparser2):
    pos_x = int(translate_position(player.position[tick][0], "x"))
    pos_y = int(translate_position(player.position[tick][1], "y"))
    player_size = round(((player.position[tick][2]+370)/79) + 10)
    player_size_offset = round((2*player_size)/3)

    player_color = CT_COLOR if(player.team_name == "CT") else T_COLOR

    if(player.health[tick] == 0):
        cv2.line(frame, (pos_x-player_size_offset, pos_y-player_size_offset), (pos_x+player_size_offset, pos_y+player_size_offset), player_color, 5)
        cv2.line(frame, (pos_x-player_size_offset, pos_y+player_size_offset), (pos_x+player_size_offset, pos_y-player_size_offset), player_color, 5)

    else:
        cv2.circle(frame, (pos_x, pos_y), player_size, player_color, -1)
        if(player.HasHelmet[tick] == 1):
            overlay_image(frame, "head_armor.png", (pos_x-round(player_size*0.607), pos_y-round(player_size*0.607)), 1.0, (player_size/14)+0.2)
        elif(player.HasArmor[tick] == 1):
            overlay_image(frame, "armor.png", (pos_x-round(player_size*0.607), pos_y-round(player_size*0.607)), 1.0, (player_size/14)+0.2)

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

        overlay_image(frame, "weapons/"+held_item+".png", (pos_x, pos_y-50), 1.0, 0.25)

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

main()