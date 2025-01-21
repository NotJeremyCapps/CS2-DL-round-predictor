from awpy import Demo
from player import Player
from round import Round
#spirit-vs-faze-m3-dust2.dem
#cloud9-vs-saw-m1-nuke.dem
#test2.dem

def main():

    parser = Demo("../game_demos/spirit-vs-faze-m3-dust2.dem")

    map = parser.header["map_name"]
    print("Map: ", map)

    round_starts = (parser.rounds["freeze_end"]) #this is the ticks for end of freeze time at start of rounds
    round_ends = (parser.rounds["end"]) #does not include time between round determination and respawn

    print("round   start_tick   end_tick")
    for x in range(len(parser.rounds)):
        print(x+1, "      ", round_starts[x], "      ", round_ends[x])



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
   

    game = [] #game to contain all the rounds
    possible_inventory_items = []
    for x in range(len(parser.rounds)): #loops for every round played
        game.append(Round())
        first_tick_of_round = round_starts[x]
        start_tick_round_index = parser.ticks.query('tick == @first_tick_of_round').head(1).index[0] #query takes a long time and dont want to do it for every tick, index of data frame from index of tick 



        players = []
        for i in range(0,10):
            players.append(Player(name=f"player{i}", enums_path = "enums.json"))
        #for testing
        #round_player = []
        #round_team_name = []

        for y in range(len(range(round_starts[x], round_ends[x] + 1))): #loops for every tick in that round
            
            start_idx_curr_tick = start_tick_round_index+(y*10)

            #if y % 10 == 0:
            #    with open("tick_info.txt", "a") as f:
            #        f.write(str(parser.ticks.head(n=y+10)))

            curr_tick_info = parser.ticks.loc[start_idx_curr_tick : start_idx_curr_tick+9] #gets 10 dataframes (1 for each player) for each tick
            #curr_frame_info = parser.ticks.loc[start_idx_curr_tick : start_idx_curr_tick+9]
            #print(curr_tick_info.dtype())
            #parser.

            if(curr_tick_info.empty):
                print("error parsing data for round ", x+1)
                del game[len(game)-1]
                del players
                break



           
            #for testing
            #all_players_name = []
            #all_team_name = []

            # Ten players in game
            for z in range(0, 10):
                players[z].load_tick_data(start_idx_curr_tick, curr_tick_info, z)
                #plyr_tick_data_idx = start_idx_curr_tick+z

                # players[z].load_tick_data(plyr_tick_data_idx, curr_tick_info)

                #player_inventory = curr_tick_info.inventory.loc[start_idx_curr_tick+z]

                #printing weapons
                '''
                for weap in player_inventory:
                    if weap not in possible_inventory_items:
                        possible_inventory_items.append(weap)
                        with open("possible_weapons.txt", "a") as f:
                            f.write(str(weap) + "\n")
                '''

              


                #double checks to make sure players for data frames are in the same order
                '''
                player_name = [current_tick_info.name.loc[start_index_current_tick+z]]
                #print(player_name)
                all_players_name.append(player_name)

                if(y==0 and z==9):
                    print("Round number:", x,"\n")
                    print(all_players_name)
                elif(y > 0):
                    if(round_player[y-1][z] != player_name):
                        print("ERROR WITH SEQUENCE OF PLAYERS")
                        print(round_player[y-1][z])
                        print(player_name)
                        exit(0)
                '''
                
                '''
                #check teams
                player_team = [current_tick_info.team_name.loc[start_index_current_tick+z]]
                all_team_name.append(player_team)
                if(z==9 and y>0): #only check once per tick
                    print("Round number:", x,"\n")
                    print(y)
                    print(round_team_name[y-1])
                '''
           

            #for testing
            #round_player.append(all_players_name)
            #round_team_name.append(all_team_name)

        #if round has no bad frame data add to list
        try:
            players[5].print_stats()
            game[len(game)-1].players = players #add players stats for each round
          
        except:
            print("a round was invalid and can't be added to list")
            # does this to skip append if round was deleted

    #game_movements is a list of each valid round
    #each round in game_movements has a list of every tick that round
    #every tick has a list of all 10 players in the game
    #each of the 10 players in the tick has a list of their X,Y,Z positions



main()