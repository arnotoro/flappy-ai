import game
import numpy as np

def play():
    g = game.Game("user_agent", "cpu")
    while True:
        score = g.main(draw=True)
        print("Score: ", score)


if __name__ == "__main__":
    play()