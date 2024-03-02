#!/bin/python3
import pygame
import sys
import random


# Step 1: Import the necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
import torch.nn.functional as F
from itertools import count


env = YourGameEnvironment()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# AI code

# Step 2: Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Step 3: Define the Replay Memory
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

    BATCH_SIZE = 128
    GAMMA = 0.999
    TARGET_UPDATE = 10

    policy_net = QNetwork(state_size, action_size, seed).to(device)
    target_net = QNetwork(state_size, action_size, seed).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()  # set the target network in evaluation mode
    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

    def select_action(state):
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    episode_durations = []
    # Step 4: Define the DQN training procedure
    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                      dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

    # Step 5: Define the main loop that will use the DQN to play the game and learn
    num_episodes = 50
    for i_episode in range(num_episodes):
        state = env.reset()
        for t in count():
            action = select_action(state)
            next_state, reward, done, _ = env.step(action.item())
            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model()
            if done:
                episode_durations.append(t + 1)
                break
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

# Game code
# Define screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

CELL_SIZE = 10 # Define the size of the snake and food

# Define directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class Snake:
    def __init__(self):
        self.length = 1
        self.positions = [((SCREEN_WIDTH // 2), (SCREEN_HEIGHT // 2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.color = (0, 255, 0)
        self.eaten = False
        self.game_over = False
        self.move_counter = 0
        self.move_frequency = 10  # Increase this value to make the snake move slower

    def draw(self, screen):
        for position in self.positions:
            pygame.draw.rect(screen, self.color, pygame.Rect(position[0], position[1], CELL_SIZE, CELL_SIZE))

    def move(self):
        self.move_counter += 1
        if self.move_counter >= self.move_frequency:
            head = self.get_head_position()
            new_position = (head[0] + self.direction[0] * CELL_SIZE, head[1] + self.direction[1] * CELL_SIZE)
            if new_position[0] < 0 or new_position[0] >= SCREEN_WIDTH or new_position[1] < 0 or new_position[
                1] >= SCREEN_HEIGHT or new_position in self.positions:
                self.game_over = True
            else:
                self.positions.insert(0, new_position)
                if len(self.positions) > self.length and not self.eaten:
                    self.positions.pop()
                self.eaten = False
            self.move_counter = 0

    def get_head_position(self):
        return self.positions[0]
    
    def get_positions(self):
        return self.positions

    def get_score(self):
        return (self.length - 1) // 5

    def eat(self, food):
        if self.get_head_position() == food.position:
            self.length += 5
            self.eaten = True
            food.randomize_position()

class Food:
    def __init__(self):
        self.position = (0, 0)
        self.color = (255, 0, 0)
        self.randomize_position()

    def randomize_position(self):
        position = (random.randint(0, SCREEN_WIDTH // CELL_SIZE - 1) * CELL_SIZE, random.randint(0, SCREEN_HEIGHT // CELL_SIZE - 1) * CELL_SIZE)
        while position in Snake().get_positions():
            position = (random.randint(0, SCREEN_WIDTH // CELL_SIZE - 1) * CELL_SIZE, random.randint(0, SCREEN_HEIGHT // CELL_SIZE - 1) * CELL_SIZE)
        self.position = position

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, pygame.Rect(self.position[0], self.position[1], CELL_SIZE, CELL_SIZE))



def display_game_over_screen(screen):
    font = pygame.font.Font(None, 72)  # Increase the font size to make the text bigger
    text = font.render("Game Over", True, (255, 255, 255))
    text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))  # Adjust the position to center the text

    button_font = pygame.font.Font(None, 36)  # Use a smaller font for the button text
    button_text = button_font.render("Replay", True, (255, 255, 255))
    button_rect = pygame.Rect(SCREEN_WIDTH // 2 - 70, SCREEN_HEIGHT // 2 + 20, 140, 60)  # Create a rectangle for the button
    button_text_rect = button_text.get_rect(center=button_rect.center)  # Center the button text in the button rectangle

    screen.fill((0, 0, 0))
    screen.blit(text, text_rect)
    pygame.draw.rect(screen, (255, 0, 0), button_rect)  # Draw the button rectangle
    pygame.draw.rect(screen, (255, 255, 255), button_rect, 2)  # Draw a border for the button rectangle
    screen.blit(button_text, button_text_rect)  # Draw the button text
    pygame.display.flip()

    return button_rect

def main():
    global replay_button
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)

    snake = Snake()
    food = Food()

    font = pygame.font.Font(None, 36)  # Use a smaller font for the score
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    snake.direction = UP
                elif event.key == pygame.K_DOWN:
                    snake.direction = DOWN
                elif event.key == pygame.K_LEFT:
                    snake.direction = LEFT
                elif event.key == pygame.K_RIGHT:
                    snake.direction = RIGHT
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if replay_button.collidepoint(mouse_pos):
                    snake = Snake()
                    food = Food()

        screen.fill((0, 0, 0))

        snake.eat(food)
        snake.move()

        if snake.game_over:
            replay_button = display_game_over_screen(screen)
        else:
            snake.draw(screen)
            food.draw(screen)
            # Draw the score
            score_text = font.render(f"Score: {snake.get_score()}", True, (255, 255, 255))
            screen.blit(score_text, (10, 10))  # Draw the score at the top left of the screen

        pygame.display.update()
        clock.tick(120)

if __name__ == "__main__":
    main()