from model import Gomoku, Color
from agents import Bot, RandomAgent, Human

class Protocol():
    def __init__(self, s_player1, s_player2, model_path1=None, model_path2=None, insight=True, time=1):
        self.player1 = self.__set_player(s_player1, model_path1, insight, time)
        self.player2 = self.__set_player(s_player2, model_path2, insight, time)
    def run(self, no_of_games=1):
        blue, red, draw = [0,0,0]
        for game in range(no_of_games):
            gomoku = Gomoku()
            last_move = None
            while not gomoku.is_over():
                player = self.player1 if gomoku.turn == Color.BLUE else self.player2
                move = player.move(gomoku, last_move)
                gomoku = gomoku.move(move)
                last_move = move
            if gomoku.winner == Color.BLUE:
                blue += 1
            elif gomoku.winner == Color.RED:
                red += 1
            else:
                draw += 1
        print('Results ---->')
        print('Blue: ' + str(blue))
        print('Red: ' + str(red))
        print('Draw: ' + str(draw))
    def __set_player(self, player_type, model_path, insight, time):
        if player_type == 'B' and model_path is None:
            raise Exception('Bot (' + player_type + ') has no brain attached.')
        elif player_type == 'B':
            return Bot(model_path, insight=insight, time=time)
        elif player_type == 'R':
            return RandomAgent(insight)
        elif player_type == 'H':
            return Human()
        else:
            raise Exception('Unknown player type: ' + player_type)
            return None