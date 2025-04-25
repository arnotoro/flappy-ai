import argparse
from environment.game import Game

def main():
    parser = argparse.ArgumentParser(description="Run a trained Flappy Bird AI agent.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file (.pth)")
    args = parser.parse_args()

    g = Game("loaded_agent", "cpu", args.model)

    while True:
        score = g.main(draw=True)
        print("Score:", score)

if __name__ == "__main__":
    main()
