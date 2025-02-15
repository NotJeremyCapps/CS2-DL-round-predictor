from awpy import Demo
from awpy import plot
from player import Player
import matplotlib.pyplot as plt
import matplotlib.animation as animation

PATH = "../game_demos/"
DEMO_NAME = "spirit-vs-faze-m3-dust2.dem"

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

    for round_num in range(len(parser.rounds)): #loops for every round played

        first_tick_of_round = round_starts[round_num]
        start_tick_round_index = parser.ticks.query('tick == @first_tick_of_round').head(1).index[0] #query takes a long time and dont want to do it for every tick, index of data frame from index of tick 

            
        players:list[Player] = []
        for i in range(0,10):
            players.append(Player(name=f"player{i}", enums_path = "enums.json"))
            

        for y in range(len(range(round_starts[round_num], round_ends[round_num] + 1))): #loops for every tick in that round


            start_idx_curr_tick = start_tick_round_index+(y*10)

            # if y % 10 == 0:
            #     with open("tick_info.txt", "a") as f:
            #         f.write(str(parser.ticks.head(n=start_idx_curr_tick+10)))

            curr_tick_info = parser.ticks.loc[start_idx_curr_tick : start_idx_curr_tick+9] #gets 10 dataframes (1 for each player) for each tick


            if(curr_tick_info.empty):
                for z in range(0, 10):
                    players[z] = prev_players[z]
            else:
                # Ten players in game
                for z in range(0, 10):

                    players[z].load_tick_data(start_idx_curr_tick, curr_tick_info, z)
                    #print(z, "TEST ", round_num)

            prev_players = players

        pos = (int(players[0].position[0][0]),int(players[0].position[0][1]),int(players[0].position[0][2]))
        list_pos.append(pos)
        #fig, ax = plot.plot(map_name, list_pos)

        
    

    settings = {
        "size" : 6.0
    }

    size = 6.0
    list_set = []
    list_set.append(settings)

    dict = {
        "points" : list_pos[0:0],
        "points_settings" :  list_set
    }

    list_dict = []
    list_dict.append(dict)
    #print(parser.ticks)
    #Tuple = (0,0,0)

    #list = []


    #list.append(Tuple)


    plot.gif(map_name=map_name, frames_data=list_dict, output_filename="game.gif", duration=16)

    #fig, ax = plot.plot(map_name, list_pos)

    #ax.set_title("Simple Plot")

    #ani = animation.FuncAnimation(fig=fig, func=update, frames=1, interval=16)
    #plt.show()

    #def update(frame):



main()