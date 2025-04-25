
import pygame
from pygame.locals import *
import random
import sys
import time
import torch
import numpy as np
import vidmaker
import matplotlib.pyplot as plt

# Local imports
from environment.config import *
from environment.objects import Pipe, Bird, Ground
from agents import user_agent, random_agent, dqn_agent, loaded_agent

#Agents
AGENTS = ["user_agent", "random_agent", "dqn_agent", "loaded_agent"]

#Pygame image loading
bird_image = pygame.image.load('assets/bluebird-upflap.png')
bird_image = pygame.transform.scale(bird_image, (BIRD_WIDTH, BIRD_HEIGHT))
pipe_image = pygame.image.load('assets/pipe-green.png')
pipe_image = pygame.transform.scale(pipe_image, (PIPE_WIDHT, PIPE_HEIGHT))
ground_image = pygame.image.load('assets/base.png')
ground_image = pygame.transform.scale(ground_image, (GROUND_WIDHT, GROUND_HEIGHT))
BACKGROUND = pygame.image.load('assets/background-day.png')
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDHT, SCREEN_HEIGHT))
GAME_FPS = 30


"""
Main game class which is running and controlling the game
"""
class Game:
    
    def __init__(self, agent_name, device, model_path=None):

        #Initialize agent
        if not agent_name in AGENTS: sys.exit("Agent not defined")
        if device != "cpu" and device != "cuda": sys.exit("Computing device not available")
        if agent_name == "user_agent": 
            self.agent = user_agent.User_agent()
            print("Initialize game with: User_agent")
        if agent_name == "random_agent": 
            self.agent = random_agent.Random_agent()
            print("Initialize game with: Random_agent")
        if agent_name == "dqn_agent": 
            self.agent = dqn_agent.DQN_agent(device)
            print("Initialize game with: DQN_agent")
            print("Trainable parameters: {}".format(sum(p.numel() for p in vars(self.agent)["model"].parameters())))
        if agent_name == "loaded_agent":
            self.agent = loaded_agent.Loaded_agent(device, trained_model_path=model_path)
            print("Initialize game with: Loaded_agent")
            print("Trained parameters: {}".format(sum(p.numel() for p in vars(self.agent)["model"].parameters())))
        self.device = device if device == "cuda" else "cpu"

        #Game objects (Get initialized new every game played)
        self.bird = None
        self.ground = None
        self.pipes = None
        self.score = None
        self.turn = None
        self.tries = 0

        # Episode counter
        self.episode_N = 0

        #Training mode for agent
        self.train = False

        # Results
        self.scores = []
        self.losses = []

    def init_game(self):

        #Initialize game objects
        self.bird = Bird(bird_image)
        self.ground = Ground(ground_image, 0)
        self.pipes = []
        self.score = 0
        self.turn = 0

        #Initialize pipes
        for i in range(3):
            #Pipe initial positions
            xpos = PIPE_DISTANCE * i + PIPE_DISTANCE
            ysize = random.randint(200, 300)

            #Append pipes to list
            self.pipes.append(Pipe(pipe_image, False, xpos, ysize))
            self.pipes.append(Pipe(pipe_image, True, xpos, SCREEN_HEIGHT - ysize - PIPE_GAP))

    def pipe_handling(self):
        #if pipes out of screen add new ones and remove old
        if vars(self.pipes[0])["pos"][0] <= -100:

            #Remove old pipes
            pipe_old_y = vars(self.pipes[0])["pos"][1]
            del self.pipes[0]
            del self.pipes[0]


            #New pipe initial positions
            xpos = PIPE_DISTANCE * 3 - 100
            ysize = random.randint(150, 350)

            #Append new pipes
            self.pipes.append(Pipe(pipe_image, False, xpos, ysize))
            self.pipes.append(Pipe(pipe_image, True, xpos, SCREEN_HEIGHT - ysize - PIPE_GAP))
                
    def collision(self):
        #Check ground and roof collision
        if vars(self.bird)["pos"][1] < 0 or vars(self.bird)["pos"][1] > SCREEN_HEIGHT - GROUND_HEIGHT - BIRD_HEIGHT:
            return True

        #Check for pipe collision
        if vars(self.pipes[0])["pos"][0] - vars(self.bird)["pos"][2] < vars(self.bird)["pos"][0] and vars(self.bird)["pos"][0] < vars(self.pipes[0])["pos"][0] +vars(self.pipes[0])["pos"][2]:
            if vars(self.pipes[0])["pos"][1] < vars(self.bird)["pos"][1] + vars(self.bird)["pos"][3] or vars(self.pipes[0])["pos"][1] - PIPE_GAP > vars(self.bird)["pos"][1]:
                return True

        return False

    def score_update(self):
        if vars(self.bird)["pos"][0] == vars(self.pipes[0])["pos"][0]:
            self.score += 1

    def game_state(self):
        state = []

        #Gamestate passing to the agent: 1-horizontal distance to next pipe, 2-vertical distance to lower next pipe, 3-bird speed
        for pipe in self.pipes:
            if vars(self.bird)["pos"][0] < vars(pipe)["pos"][0] + vars(pipe)["pos"][2]: #Check which pipe is the next one
                state.append((- vars(self.bird)["pos"][0] + vars(pipe)["pos"][2] + vars(pipe)["pos"][0]) / PIPE_DISTANCE)
                state.append((vars(pipe)["pos"][1] - PIPE_GAP/2 - vars(self.bird)["pos"][1] - vars(self.bird)["pos"][3] / 2) / SCREEN_HEIGHT * 2)
                break
        state.append(vars(self.bird)["speed"] / SPEED) # bird speed
        #state.append(vars(self.bird)["pos"][1] / SCREEN_HEIGHT) # bird height

        return state

    def reward(self):

        reward = 0.1 #reward of 0.1 for surviving
        if self.score > 0:
            reward = self.score*1.2 #reward for passing a pipe
        if self.collision():
            reward = -10 #reward -10 for colliding

        #print(round(reward,4))
        return round(reward,4)

    def main(self, draw, video_interval=None):
        # Initialize pygame screen if wanted
        if draw:
            pygame.init()
            screen = pygame.display.set_mode((SCREEN_WIDHT, SCREEN_HEIGHT))
            pygame.display.set_icon(bird_image)
            pygame.display.set_caption('Flappy Bird')
            clock = pygame.time.Clock()
            if self.train:
                video = vidmaker.Video(f'assets/videos/flappy_bird_{self.episode_N}_{time.strftime("%Y%m%d-%H%M")}.mp4', late_export=True)

            # Set up font for score display
            font = pygame.font.Font(pygame.font.match_font('Comic Sans MS'), 30)

        # Initialize game
        active_episode = True
        self.init_game()

        # Handle waiting for the user to press SPACE to start
        if isinstance(self.agent, user_agent.User_agent):
            waiting_for_input = True
            while waiting_for_input:
                # Display "Press SPACE to start" message
                if draw:
                    screen.blit(BACKGROUND, (0, 0))
                    start_text = font.render("Press SPACE to start", True, (255, 255, 255))  # White text
                    screen.blit(start_text, (SCREEN_WIDHT // 2 - 150, SCREEN_HEIGHT // 2 - 20))  # Centered
                    pygame.display.update()

                # Check for user input
                for event in pygame.event.get():
                    if event.type == QUIT:
                        active_episode = False
                        waiting_for_input = False
                    if event.type == KEYDOWN and event.key == K_SPACE:
                        # Stop waiting and make the bird jump
                        self.bird.bump()  # Perform the first jump
                        waiting_for_input = False  # Exit the loop

            # Add a brief countdown after SPACE is pressed
            if draw and active_episode:
                for countdown in range(3, 0, -1):
                    screen.blit(BACKGROUND, (0, 0))
                    countdown_text = font.render(f"Starting in {countdown}...", True, (255, 255, 255))  # White text
                    screen.blit(countdown_text, (SCREEN_WIDHT // 2 - SCREEN_WIDHT//4, SCREEN_HEIGHT // 2 - 20))  # Centered
                    pygame.display.update()
                    pygame.time.delay(250)  # 0.5-second delay per step

        # Game loop (once the player pressed SPACE)
        while active_episode:
            if draw:
                clock.tick(GAME_FPS)
                screen.blit(BACKGROUND, (0, 0))

                # Check for closing game window
                if not isinstance(self.agent, user_agent.User_agent):
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            active_episode = False
                # Render current score
                score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))  # White color
                screen.blit(score_text, (10, 10))  # Position it at the top-left corner

                if self.train:
                    # Render current epsilon
                    eps_text = font.render(f"Generation: {self.episode_N}", True, (255, 255, 255))
                    screen.blit(eps_text, (10, 40))
                else:
                    eps_text = font.render(f"Tries: {round(self.tries, 2)}", True, (255, 255, 255))
                    screen.blit(eps_text, (10, 40))

                # Handle bird action when space is pressed again
                for event in pygame.event.get():
                    if event.type == KEYDOWN and event.key == K_SPACE:
                        self.bird.bump()  # Make the bird flap
                    if event.type == QUIT:
                        active_episode = False

            # Get and execute agent action
            state = self.game_state()
            action = self.agent.act(state, self.train)
            if action == 1:
                self.bird.bump()  # Bird flaps
            if action == -1:
                active_episode = False  # End game

            # Update environment
            self.bird.update()
            for pipe in self.pipes:
                pipe.update()
            self.score_update()
            
            if self.score >= 200:
                active_episode = False
            # Remove pipes if out of screen and instantiate new ones
            self.pipe_handling()

            # Check for collisions
            if self.collision():
                active_episode = False
            
            # Give state to experience buffer if in training mode of dqn_agent
            if self.train:
                vars(self.agent)["buffer"][0].append(state)
                vars(self.agent)["buffer"][1].append(self.game_state())
                vars(self.agent)["buffer"][2].append(self.reward())
                if action == 0: 
                    vars(self.agent)["buffer"][3].append(torch.Tensor([0]))
                if action == 1: 
                    vars(self.agent)["buffer"][3].append(torch.Tensor([1]))
                
                #print(self.reward())
            self.turn += 1

            # Update screen
            if draw:
                self.bird.draw(screen)
                self.ground.draw(screen)
                for pipe in self.pipes:
                    pipe.draw(screen)

                #print(self.episode_N, video_interval)
                if self.train and self.episode_N % video_interval == 0:
                    video.update(pygame.surfarray.pixels3d(pygame.display.get_surface()).swapaxes(0, 1), inverted=False)
                pygame.display.update()

        # Quit pygame window
        if draw:
            self.tries += 1
            pygame.display.quit()
            pygame.quit()

            if self.train and self.episode_N % video_interval == 0:
                video.export(verbose=True)

        #print(self.score)
        return self.score
    

        
    def train_agent(self, draw, episodes, batches, hyperparameter, video_interval=None):
        # Training control parameters
        convergence = 0
        loss = 0
        mean_score = []
        time_start = time.time()
        rewards = []  # To store rewards for each episode

        # Training initializations
        print("Start training process of agent")
        if self.device == "cuda":
            print("Using {} device".format(self.device), ": ", torch.cuda.get_device_name(0))
        else:
            print("Using {} device".format(self.device))
        print("Used training hyperparameters: ", hyperparameter)

        if not isinstance(self.agent, dqn_agent.DQN_agent):
            sys.exit("Agent is not trainable")

        self.train = True

        for episode in range(1, episodes + 1):
            self.episode_N = episode

            # Specify episode lr and epsilon
            eps = hyperparameter["eps_end"] + (hyperparameter["eps_start"] - hyperparameter["eps_end"]) * np.exp(-1. * episode / episodes * 10)
            lr = hyperparameter["lr_end"] + (hyperparameter["lr_start"] - hyperparameter["lr_end"]) * np.exp(-1. * episode / episodes * 10)
            vars(self.agent)["lr"] = lr
            vars(self.agent)["batch_size"] = hyperparameter["batch_size"]
            vars(self.agent)["gamma"] = hyperparameter["gamma"]
            vars(self.agent)["epsilon"] = eps

            # Run an episode
            if video_interval is not None:
                draw = self.episode_N % video_interval == 0
                train_score = self.main(draw, video_interval)
            else:
                train_score = self.main(draw)

            if train_score >= 100:
                convergence += 1

            # Train agent
            for i in range(batches):
                loss += self.agent.train()

            # reward for current episode = current - last episode reward
            if episode > 1:
                print(np.sum(self.agent.buffer[2]) - rewards[-1])
                rewards.append(np.sum(self.agent.buffer[2]) - rewards[-1])
            else:
                rewards.append(np.sum(self.agent.buffer[2]))  # Store total reward for the episode

           # print(np.sum(rewards))


            # rewards.append(np.sum(self.agent.buffer[2]))  # Store total reward for the episode
            # print(np.sum(rewards))
            # Test agent
            self.train = False
            test_score = self.main(False)
            self.scores.append(test_score)
            mean_score.append(test_score)

            if test_score >= 100:
                convergence += 1
            else:
                convergence = 0

            # Print training performance log
            time_step = time.time()
            if episode % 10 == 0 or convergence == 2:
                print("Episode: [{}/{}]".format(episode, episodes) +
                    "    -Time: [{}<{}]".format(time.strftime("%M:%S", time.gmtime(time_step - time_start)),
                                                time.strftime("%M:%S", time.gmtime((time_step - time_start) * episodes / episode))) +
                    " {}s/it".format(round((time_step - time_start) / episode, 1)) +
                    "    -Loss: {}".format(round(loss / batches, 6)) +
                    "    -MeanTestScore: {}".format(round(np.mean(mean_score))) +
                    "    -Cumulative reward for : {}".format(round(np.sum(self.agent.buffer[2]))))
                mean_score = []
            self.losses.append(loss / batches)
            loss = 0

            # Terminate training if agent never collides after two training procedures in a row
            if convergence == 2:
                print("Agent performed faultless")
                break
            self.train = True

        # Save model after training
        torch.save(self.agent.model.state_dict(),
                "model_files/flappy_bird_{}_{}.pth".format(episodes, time.strftime("%Y%m%d-%H%M%S")))

        self.train = False
        self.plot_performance(rewards)  # Plot performance with rewards for each episode
        print("Training finished after {} episodes".format(episode))
        print("Model saved as flappy_bird_{}.pth".format(episodes))

        return episode


    def plot_performance(self, rewards):
        # Plot loss and reward per episode
        plt.figure(figsize=(12, 6))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.scores, label="Score")
        plt.title("Score per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Score")

        # Plot reward (cumulative rewards)
        plt.subplot(1, 2, 2)
        plt.plot(rewards, label="Total Rewards", color='blue')
        plt.title("Total Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Rewards")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'performance_{time.strftime("%Y%m%d-%H%M%S")}.png')
        plt.show()