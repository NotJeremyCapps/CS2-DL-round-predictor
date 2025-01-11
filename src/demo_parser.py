from awpy import Demo

#spirit-vs-faze-m3-dust2.dem
#test2.dem
def main():

    parser = Demo("../game_demos/spirit-vs-faze-m3-dust2.dem")

    map = parser.header["map_name"]
    print("Map: ", map)

    round_starts = (parser.rounds["start"])
    round_ends = (parser.rounds["end"])

    for x in range(len(round_ends)):
        round_ends[x] = round_ends[x] +447 #447 is the number of ticks between round score determination and last tick before next round

    print("round   start_tick   end_tick")
    for x in range(len(parser.rounds)):
        print(x+1, "      ", round_starts[x], "      ", round_ends[x])



main()