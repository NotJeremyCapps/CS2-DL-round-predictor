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

        
        players = [Player(), Player(),Player(),Player(),Player(),Player(),Player(),Player(),Player(),Player()]

        #for testing
        #round_player = []
        #round_team_name = []

        for y in range(len(range(round_starts[x], round_ends[x] + 1))): #loops for every tick in that round

            start_idx_curr_tick = start_tick_round_index+(y*10)

            # with open("tick_info.txt", "a") as f:
            #     f.write(str(parser.ticks.head(n=10)))

            curr_tick_info = parser.ticks.loc[start_idx_curr_tick : start_idx_curr_tick+9] #gets 10 dataframes (1 for each player) for each tick

            if(curr_tick_info.empty):
                print("error parsing data for round ", x+1)
                del game[len(game)-1]
                del players
                break



           
            #for testing
            #all_players_name = []
            #all_team_name = []

            # Ten players in game
            for z in range(10):

                player_location = [curr_tick_info.X.loc[start_idx_curr_tick+z], curr_tick_info.Y.loc[start_idx_curr_tick+z], curr_tick_info.Z.loc[start_idx_curr_tick+z]]
                player_pitch = [curr_tick_info.pitch.loc[start_idx_curr_tick+z]]
                player_yaw = [curr_tick_info.yaw.loc[start_idx_curr_tick+z]]
                player_health = [curr_tick_info.health.loc[start_idx_curr_tick+z]]
                player_HasHelmet = [curr_tick_info.has_helmet.loc[start_idx_curr_tick+z]]
                
                player_inventory = curr_tick_info.inventory.loc[start_idx_curr_tick+z]

                for weap in player_inventory:
                    if weap not in possible_inventory_items:
                        possible_inventory_items.append(weap)
                        with open("possible_weapons.txt", "a") as f:
                            f.write(str(weap) + "\n")


              
                #append individual player information for each tick
                players[z].postion.append(player_location)
                players[z].pitch.append(player_pitch)
                players[z].yaw.append(player_yaw)
                players[z].health.append(player_health)

                if(player_HasHelmet[0] == False):
                    players[z].HasHelmet.append(0)
                else:
                    players[z].HasHelmet.append(1)




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
            game[len(game)-1].players = players #add players stats for each round
          
        except:
            print("a round was invalid and can't be added to list")
            # does this to skip append if round was deleted

    #game_movements is a list of each valid round
    #each round in game_movements has a list of every tick that round
    #every tick has a list of all 10 players in the game
    #each of the 10 players in the tick has a list of their X,Y,Z positions



main()