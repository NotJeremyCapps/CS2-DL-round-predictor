

class Round:
    def __init__(self, n_ticks:int = 0):

        self.start_tick = None
        self.end_tick = None
        self.players = []


        self.outcome = None
        

    def init_players(self, players):
        self.players = players