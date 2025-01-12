from awpy import Demo

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


    #x represents current round -1
    #y represents what tick of the round we are on (not total ticks but tick of that specific round)
    game_movements = []

    for x in range(len(parser.rounds)): #loops for every round played
        first_tick_of_round = round_starts[x]
        start_tick_round_index = parser.ticks.query('tick == @first_tick_of_round').head(1).index[0] #query takes a long time and dont want to do it for every tick

        
        round_movements = []

        for y in range(len(range(round_starts[x], round_ends[x] + 1))): #loops for every tick in that round

            start_index_current_tick = start_tick_round_index+(y*10)

            current_tick_info = parser.ticks.loc[start_index_current_tick : start_index_current_tick+9] #gets 10 dataframes (1 for each player) for each tick

            if(current_tick_info.empty):
                print("error parsing data for round ", x+1)
                del round_movements
                break

            all_players_locations = []
            for z in range(10):

                player_location = [current_tick_info.X.loc[start_index_current_tick+z], current_tick_info.Y.loc[start_index_current_tick+z], current_tick_info.Z.loc[start_index_current_tick+z]]
            
                all_players_locations.append(player_location)
            
            round_movements.append(all_players_locations)

        #if round has no bad frame data add to list
        try:
            game_movements.append(round_movements)
        except:
            print("a round was invalid and can't be added to list")
            # does this to skip append if round was deleted

    #game_movements is a list of each valid round
    #each round in game_movements has a list of every tick that round
    #every tick has a list of all 10 players in the game
    #each of the 10 players in the tick has a list of their X,Y,Z positions
    print(game_movements)

main()