import game
import numpy as np

# hyperparameters for training the agent
hyperparameter = {
    "lr_start": 1e-4,
    "lr_end": 1e-4,
    "batch_size": 128,
    "gamma": 0.95,     
    "eps_start": 0.9,
    "eps_end": 1e-2
} 

# Decide whether the agent is trained or the human is playing
mode = "test"  # Set this to "train" to train the agent, or "play" for human play

# Training the agent
if mode == "train":
    episodes = []
    for i in range(1):
        print(f"Training iteration {i}")
        g = game.Game("dqn_agent", "cuda")
        num_episodes = g.train_agent(draw=True, episodes=250, batches=100, hyperparameter=hyperparameter, video_interval=10)
        episodes.append(num_episodes)
        score = g.main(draw=True) 
        print("Training Complete. Final Score: ", score)

    print("Average number of episodes to converge to optimal solution: ", np.mean(episodes))

# Playing the game as a human
elif mode == "play":
    g = game.Game("user_agent", "cpu")
    while True:   
        score = g.main(draw=True)     
        print("Score: ", score)

# Load a trained agent
if mode =="test":
    #model = input("Enter the model name: ")
    # Load the trained agent 
    model_file = f"model_files/fb_250.pth"  # Using f-str ing for formatting
 
    g = game.Game("loaded_agent", "cpu", model_file)
    # score = g.main(draw=True)
    # print("Score: ", score)

    while True:
        score = g.main(draw=True)
        print("Score: ", score)
