from demoparser2 import DemoParser

class RoundTimes():
    def __init__(self, startTick, endTick):
        self.startTick = startTick
        self.endTick = endTick
    

#spirit-vs-faze-m3-dust2.dem
#test2.dem
def main():
    parser = DemoParser("../game_demos/spirit-vs-faze-m3-dust2.dem") #if a players name has a non-unicode character it will break print() and throw an error if you try to print the name


    round_end_ticks = parser.parse_event("round_end")["tick"] #array for ticks for the end of a round
    round_start_ticks = parser.parse_event("round_start")["tick"] #array for ticks for the start of a round

    #removes warm up from start ticks if it exists
    while(len(round_end_ticks) != len(round_start_ticks)):
        for x in range(len(round_end_ticks)-2):
            round_start_ticks[x] = round_start_ticks[x+1]
        del round_start_ticks[len(round_start_ticks)-1]

    #create structs to store start and end ticks of each round
    rounds = []
    for x in range(len(round_end_ticks)):
        rounds.append(RoundTimes(round_start_ticks[x] ,round_end_ticks[x]+447)) #447 is ticks between round score determination and start of new round

    #for now this program just prints the start and end ticks of each round so we can separate each round into its own thing

    print("round, start_tick, end_tick")
    for x in range(len(rounds)):
        print(x+1, "      ", rounds[x].startTick, "       " , rounds[x].endTick)

main()