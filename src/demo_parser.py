from awpy import Demo
from player import Player
from round import Round
import os

#two text files - one for training and one for testing, and run it for all games

PATH = "../game_demos/"
#spirit-vs-faze-m3-dust2.dem
#cloud9-vs-saw-m1-nuke.dem
#test2.dem

def main():

    loaded_maps = {}

    list_of_demo_names = [x for x in os.listdir(PATH) if x.endswith(".dem")]
    for i in range(len(list_of_demo_names)):
        parser = Demo(PATH + list_of_demo_names[i])

        # with open("player_state_info.txt", "a") as f:
        #     f.write(str(parser.ticks[["player_state"]].sample(n=10)))

        map_name = parser.header["map_name"]
        print("Map: ", map_name)

        round_starts = (parser.rounds["freeze_end"]) #this is the ticks for end of freeze time at start of rounds
        round_ends = (parser.rounds["end"]) #does not include time between round determination and respawn


        # print("round   start_tick   end_tick")
        for round in range(len(parser.rounds)):
            print(round+1, "      ", round_starts[round], "      ", round_ends[round])

        with open("round_info.txt", "a") as f:
            f.write(str(parser.rounds))

            



        '''
        elements for parser.rounds

        'round', 'start', 'freeze_end', 'end', 'official_end', 'winner',
        'reason', 'bomb_plant'
        '''
    
        ''' 
        elements for parser.tick

        'inventory', 'accuracy_penalty', 'zoom_lvl', 'is_bomb_planted', 'ping',
        'health', 'has_defuser', 'has_helmet', 'flash_duration',
        'last_place_name', 'which_bomb_zone', 'armor_value',
        'current_equip_value', 'team_name', 'team_clan_name', 'X', 'pitch',
        'yaw', 'Y', 'Z', 'game_time', 'tick', 'steamid', 'name', 'round'
        '''

        #data type is dataframe which is in panda library

        #x represents current round -1
        #y represents what tick of the round we are on (not total ticks but tick of that specific round)
    
        # Update loaded maps counting dictionary
        if map_name in loaded_maps.keys():
            loaded_maps[map_name] += 1
        else:
            loaded_maps[map_name] = 0 # Counter is indexed from 0

        match_title = f"{map_name}_{loaded_maps[map_name]}" #Format name of Match
        


        game:list[Round] = [] #game to contain all the rounds

        for n in range(len(parser.rounds)):
            round_title = f"{match_title}_round_{n+1}"

            current_round = Round(round_title=round_title, 
                                round_num=n,
                                map_name=map_name,
                                demo_data_root=PATH, 
                                enums_path="enums.json")
            game.append(current_round)

        skip_counter = 0

        for round_num in range(len(parser.rounds)): #loops for every round played

            first_tick_of_round = round_starts[round_num]
            start_tick_round_index = parser.ticks.query('tick == @first_tick_of_round').head(1).index[0] #query takes a long time and dont want to do it for every tick, index of data frame from index of tick 

            
            players:list[Player] = []
            for i in range(0,10):
                players.append(Player(name=f"player{i}", enums_path = "enums.json"))
            
            # init headers for each player in round dataframe
            #print(round_num)
            game[round_num-skip_counter].init_headers()


            for y in range(len(range(round_starts[round_num], round_ends[round_num] + 1))): #loops for every tick in that round


                start_idx_curr_tick = start_tick_round_index+(y*10)

                # if y % 10 == 0:
                #     with open("tick_info.txt", "a") as f:
                #         f.write(str(parser.ticks.head(n=start_idx_curr_tick+10)))

                curr_tick_info = parser.ticks.loc[start_idx_curr_tick : start_idx_curr_tick+9] #gets 10 dataframes (1 for each player) for each tick

                if(curr_tick_info.empty):
                    print("error parsing data for round ", round_num+1)
                    del game[round_num-skip_counter]
                    del players
                    skip_counter += 1
                    break

                # Ten players in game
                for z in range(0, 10):

                    players[z].load_tick_data(start_idx_curr_tick, curr_tick_info, z)

                
        
            try:
                game[round_num-skip_counter].load_player_tick_data(players=players)
                game[round_num-skip_counter].load_round_data(round_dict=parser.rounds)
                game[round_num-skip_counter].write_round_to_csv()
            except Exception as e:
                print(f"Couldnt load round, Error: {e}")
       





main()