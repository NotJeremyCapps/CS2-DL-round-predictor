from awpy import Demo
from awpy import plot
from player import Player
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from awpy.plot.utils import position_transform_axis
import math
#import sys

#sys.path.append("/c/Users/notje/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0/LocalCache/local-packages/Python311/site-packages/opencv_cuda/install_script.py")
import cv2
import numpy as np
import os
# CPU only, would be nice to have a GPU version

PATH = "../game_demos/"
DEMO_NAME = "00nation-vs-eclot-m1-mirage.dem"

OUTPUT_PATH = "../parsed_videos/"

ASSETS_PATH = OUTPUT_PATH + "assets/"

#fps = 60

FRAME_SIZE = (1024, 1024)

FPS = 60

FOURCC = cv2.VideoWriter_fourcc(*'XVID')

def main():

    # Example: Read frames from folder
    #image_folder = '../parsed_videos/assets'
    #images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    #for image in images:
        #frame = cv2.imread(os.path.join(image_folder, image))
        #cv2.circle(frame, (int(translate_position(-1656, "x")), int(translate_position(-1800, "y"))), 8, (0,0,255), -1)
        #video_writer.write(frame)


    #f = open("test.txt", "w")


    parser = Demo(PATH + DEMO_NAME)
    map_name = parser.header["map_name"]

    round_starts = (parser.rounds["freeze_end"]) #this is the ticks for end of freeze time at start of rounds
    round_ends = (parser.rounds["end"]) #does not include time between round determination and respawn

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

        first_tick_of_round = round_starts[round_num]
        start_tick_round_index = parser.ticks.query('tick == @first_tick_of_round').head(1).index[0] #query takes a long time and dont want to do it for every tick, index of data frame from index of tick 

            
        players:list[Player] = []
        for i in range(0,10):
            players.append(Player(name=f"player{i}", enums_path = "enums.json"))

        video_writer = cv2.VideoWriter(OUTPUT_PATH+"Round_" + str(round_num+1) + ".avi", FOURCC, FPS, FRAME_SIZE, True)
            
        
        for y in range(len(range(round_starts[round_num], round_ends[round_num] + 1))): #loops for every tick in that round
            list_pos = []
            list_set = []

            start_idx_curr_tick = start_tick_round_index+(y*10)

            # if y % 10 == 0:
            #     with open("tick_info.txt", "a") as f:
            #         f.write(str(parser.ticks.head(n=start_idx_curr_tick+10)))

            curr_tick_info = parser.ticks.loc[start_idx_curr_tick : start_idx_curr_tick+9] #gets 10 dataframes (1 for each player) for each tick



            if(curr_tick_info.empty):
                for z in range(0, 10):
                    players[z].load_tick_data(prev_start_idx_curr_tick, prev_curr_tick_info, z)
            else:
                # Ten players in game
                for z in range(0, 10):

                    players[z].load_tick_data(start_idx_curr_tick, curr_tick_info, z)
                    prev_start_idx_curr_tick, prev_curr_tick_info = start_idx_curr_tick, curr_tick_info
                    #print(z, "TEST ", round_num)

            for z in range(0, 10):
                if(math.isnan(players[z].position[y][0]) or math.isnan(players[z].position[y][1])):
                    if(y!=0):
                        players[z].position[y] = players[z].position[y-1]
                    else:
                        players[z].position[y] = (0,0,0)
                    #players[z].load_tick_data(prev_start_idx_curr_tick, prev_curr_tick_info, z)

            frame = cv2.imread(ASSETS_PATH+"de_mirage.png")
            for z in range(0, 10):
                pos = (players[z].position[y][0],players[z].position[y][1],players[z].position[y][2])
                #frame = cv2.imread(ASSETS_PATH+"de_mirage.png")
                cv2.circle(frame, (int(translate_position(pos[0], "x")), int(translate_position(pos[1], "y"))), 8, (0,0,255), -1)
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


main()