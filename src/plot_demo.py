from awpy import Demo
from awpy import plot
from player import Player
import matplotlib.pyplot as plt
import matplotlib.animation as animation

PATH = "../game_demos/"
DEMO_NAME = "00nation-vs-eclot-m1-mirage.dem"

def main():
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
    count = 0

    for round_num in range(len(parser.rounds)): #loops for every round played

        first_tick_of_round = round_starts[round_num]
        start_tick_round_index = parser.ticks.query('tick == @first_tick_of_round').head(1).index[0] #query takes a long time and dont want to do it for every tick, index of data frame from index of tick 

            
        players:list[Player] = []
        for i in range(0,10):
            players.append(Player(name=f"player{i}", enums_path = "enums.json"))
            
        
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
                pos = (players[z].position[y][0],players[z].position[y][1],players[z].position[y][2])
                print("TEAM: ", players[z].team_name)
                list_pos.append(pos)
                list_set.append({"size":6.0, "color": (29/255,204/255,247/255) if(players[z].team_name == "CT") else (253/255,219/255,36/255)})
            if(count%240==0):
                dict_list.append({"points":list_pos, "point_settings":list_set})
            count += 1
        #fig, ax = plot.plot(map_name, list_pos)

        
    

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


    plot.gif(map_name=map_name, frames_data=dict_list, output_filename="game.gif", duration=4000)

    #fig, ax = plot.plot(map_name, list_pos)

    #ax.set_title("Simple Plot")

    #ani = animation.FuncAnimation(fig=fig, func=update, frames=1, interval=16)
    #plt.show()

    #def update(frame):



main()