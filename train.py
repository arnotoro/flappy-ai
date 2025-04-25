import environment.game as game
import numpy as np

def train():
    hyperparameter = {
        "lr_start": 1e-4,
        "lr_end": 1e-4,
        "batch_size": 128,
        "gamma": 0.95,
        "eps_start": 0.9,
        "eps_end": 1e-2
    }

    episodes = []

    for i in range(1):
        print(f"Training iteration {i}")
        g = game.Game("dqn_agent", "cuda")
        num_episodes = g.train_agent(draw=True, episodes=250, batches=100, hyperparameter=hyperparameter, video_interval=10)
        episodes.append(num_episodes)
        score = g.main(draw=True)
        print("Training Complete. Final Score: ", score)

    print("Average number of episodes to converge to optimal solution: ", np.mean(episodes))


if __name__ == "__main__":
    train()