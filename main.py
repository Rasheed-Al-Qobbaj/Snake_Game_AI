#!/bin/python3
import pygame
import sys
import random

# Define screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

CELL_SIZE = 10

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
                1] >= SCREEN_HEIGHT:
                self.game_over = True
            else:
                self.positions.insert(0, new_position)
                if len(self.positions) > self.length and not self.eaten:
                    self.positions.pop()
                self.eaten = False
            self.move_counter = 0

    def get_head_position(self):
        return self.positions[0]

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
        self.position = (random.randint(0, SCREEN_WIDTH // CELL_SIZE - 1) * CELL_SIZE, random.randint(0, SCREEN_HEIGHT // CELL_SIZE - 1) * CELL_SIZE)

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
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)

    snake = Snake()
    food = Food()

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

        pygame.display.update()
        clock.tick(120)

if __name__ == "__main__":
    main()