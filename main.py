#!/bin/python3
import pygame
import sys
import random
import pickle

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
        self.score = 0
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
            new_direction = (head[0] + self.direction[0] * CELL_SIZE, head[1] + self.direction[1] * CELL_SIZE)
            opposite_directions = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
            if new_direction == opposite_directions[self.direction]:
                return
            new_position = new_direction
            print(f"New position: {new_position}")
            print(f"Snake positions: {self.positions}")
            if new_position[0] < 0 or new_position[0] >= SCREEN_WIDTH or new_position[1] < 0 or new_position[
                1] >= SCREEN_HEIGHT or new_position in self.positions:
                self.game_over = True
                print("Game over condition triggered in move method")
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

    def eat(self, food):
        if self.get_head_position() == food.position:
            print("Snake ate the fruit")
            self.length += 1
            self.score += 1
            self.eaten = True
            food.randomize_position()

class QLearningSnake(Snake):
    def __init__(self):
        super().__init__()
        self.q_table = {}  # Initialize Q-table

    def get_state(self, food):
        # Get the head position
        head = self.get_head_position()

        # Get the food position
        food_position = food.position

        # Get the ghost food position
        ghost_food_position = food.next_position

        # Get the direction
        direction = self.direction

        # Return a tuple representing the state
        return (head, food_position, ghost_food_position, direction)

    def get_possible_actions(self):
        # Return a list of possible actions
        return [UP, DOWN, LEFT, RIGHT]

    def get_reward(self, food):
        # Get the head position
        head = self.get_head_position()

        # Calculate the Euclidean distance to the food before the move
        distance_before_move = ((head[0] - food.position[0]) ** 2 + (head[1] - food.position[1]) ** 2) ** 0.5

        # Move the snake
        self.move()

        # Get the new head position
        new_head = self.get_head_position()

        # Calculate the Euclidean distance to the food after the move
        distance_after_move = ((new_head[0] - food.position[0]) ** 2 + (new_head[1] - food.position[1]) ** 2) ** 0.5

        # If the snake eats the food, return a positive reward
        if self.get_head_position() == food.position:
            return 10

        # If the snake is touching the rest of its body, return a negative reward
        if self.get_head_position() in self.positions[1:]:
            return -10

        # If the game is over, return a negative reward
        if self.game_over:
            return -10

        # If the snake moves closer to the food, return a positive reward
        if distance_after_move < distance_before_move:
            return 1

        # If the snake moves away from the food, return a negative reward
        if distance_after_move > distance_before_move:
            return -1

        # If the snake doesn't eat the food, return a small negative reward
        return -0.1

    def update_q_values(self, old_state, action, reward, new_state):
        # First, Define the learning rate and discount factor
        learning_rate = 0.1
        discount_factor = 0.9

        # Then, we need to get the old Q-value
        old_q_value = self.q_table.get((old_state, action), 0)

        # Then, we need to get the maximum Q-value for the new state
        max_new_q_value = max([self.q_table.get((new_state, a), 0) for a in self.get_possible_actions()])

        # Now, we can calculate the new Q-value
        new_q_value = old_q_value + learning_rate * (reward + discount_factor * max_new_q_value - old_q_value)

        # Print out the old state, action, reward, new state, and new Q-value
        # print(f"Old state: {old_state}, Action: {action}, Reward: {reward}, New state: {new_state}, New Q-value: {new_q_value}")

        # Finally, we update the Q-table with the new Q-value
        self.q_table[(old_state, action)] = new_q_value

    def choose_action(self, state):
        # Define the opposite directions
        opposite_directions = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

        # Define the left directions
        left_directions = {UP: LEFT, DOWN: RIGHT, LEFT: DOWN, RIGHT: UP}

        # Get the opposite direction to the current direction
        opposite_direction = opposite_directions[self.direction]

        # Get the possible actions, excluding the opposite direction
        possible_actions = [action for action in self.get_possible_actions() ]#if action != opposite_direction]

        # Choose an action based on the current state and Q-values
        q_values = [self.q_table.get((state, action), 0) for action in possible_actions]
        max_q_value = max(q_values)

        # If multiple actions have the same max Q-value, we choose randomly among them
        actions_with_max_q_value = [action for action, q_value in zip(possible_actions, q_values) if
                                    q_value == max_q_value]

        chosen_action = random.choice(actions_with_max_q_value)

        # # If the chosen action is the opposite direction, go left instead
        # if chosen_action == opposite_direction:
        #     chosen_action = left_directions[self.direction]

        return chosen_action

    def move(self):
        self.move_counter += 1
        if self.move_counter >= self.move_frequency:
            head = self.get_head_position()
            new_direction = self.choose_action(self.get_state(Food()))
            opposite_directions = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
            if new_direction == opposite_directions[self.direction]:
                return
            self.direction = new_direction
            new_position = (head[0] + self.direction[0] * CELL_SIZE, head[1] + self.direction[1] * CELL_SIZE)
            print(f"New position: {new_position}")
            print(f"Snake positions: {self.positions}")
            # if new_position[0] < 0 or new_position[0] >= SCREEN_WIDTH or new_position[1] < 0 or new_position[
            #     1] >= SCREEN_HEIGHT or new_position in self.positions:
            #     self.game_over = True
            #     print("Game over condition triggered in move method")
            if new_position[0] < 0 or new_position[0] >= SCREEN_WIDTH or new_position[1] < 0 or new_position[1] >= SCREEN_HEIGHT:
                self.game_over = True
                print("Game over condition triggered in move method")
            else:
                self.positions.insert(0, new_position)
                if len(self.positions) > self.length and not self.eaten:
                    self.positions.pop()
                self.eaten = False
            self.move_counter = 0

    def save_q_table(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, file_name):
        try:
            with open(file_name, 'rb') as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            pass

class Food:
    def __init__(self):
        self.position = self.random_position()
        self.next_position = self.random_position()
        self.color = (255, 0, 0)
        self.ghost_color = (128, 128, 128)

    def random_position(self):
        while True:
            position = (random.randint(0, SCREEN_WIDTH // CELL_SIZE - 1) * CELL_SIZE,
                        random.randint(0, SCREEN_HEIGHT // CELL_SIZE - 1) * CELL_SIZE)
            if position not in Snake().get_positions():
                return position

    def randomize_position(self):
        self.position = self.next_position
        self.next_position = self.random_position()

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, pygame.Rect(self.position[0], self.position[1], CELL_SIZE, CELL_SIZE))

    def draw_ghost(self, screen):
        pygame.draw.rect(screen, self.ghost_color, pygame.Rect(self.next_position[0], self.next_position[1], CELL_SIZE, CELL_SIZE))


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

    # snake = Snake() # Use this line to play the game manually
    snake = QLearningSnake()
    snake.load_q_table('q_table.pkl')
    food = Food()

    font = pygame.font.Font(None, 36)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
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
            # elif event.type == pygame.MOUSEBUTTONDOWN:
            #     mouse_pos = pygame.mouse.get_pos()
            #     if replay_button.collidepoint(mouse_pos):
            #         # snake = Snake() # Use this line to play the game manually
            #         snake = QLearningSnake()
            #         food = Food()

        screen.fill((0, 0, 0))

        score_text = font.render(f"Score: {snake.score}", True, (255, 255, 255))  # Render the score as a text surface
        screen.blit(score_text, (10, 10))

        # Get the current state
        old_state = snake.get_state(food)

        # Choose an action
        action = snake.choose_action(old_state)

        # Perform the action
        snake.direction = action
        snake.move()

        # Check if the snake eats the food
        snake.eat(food)

        # Get the reward
        reward = snake.get_reward(food)

        # Get the new state
        new_state = snake.get_state(food)

        # Update the Q-values
        snake.update_q_values(old_state, action, reward, new_state)

        if snake.game_over:
            # pygame.time.delay(2000)
            #replay_button = display_game_over_screen(screen)
            snake.save_q_table('q_table.pkl')
            snake = QLearningSnake()
            food = Food()
            snake.load_q_table('q_table.pkl')
        else:
            snake.draw(screen)
            food.draw(screen)
            food.draw_ghost(screen)

        pygame.display.update()
        clock.tick(100000)

if __name__ == "__main__":
    main()