import pygame
import random
from math import sqrt
from snake import Snake
from apple import Apple
from config import *


class Game:
    def __init__(self):        
        self.display = pygame.display.set_mode((game_w, game_h))
        pygame.display.set_caption('snake')
        self.color = Color.white        
        pygame.font.init()
        self.font = pygame.font.SysFont("calibri", 10)

    def play(self):
        global rows
        
        snake = Snake()
        apple = Apple()

        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()

            # Detect collision with apple
            if snake.x == apple.x and snake.y == apple.y:
                snake.eat()
                apple = Apple()

            # collision with body
            for part in snake.body:
                if snake.x == part[0] and snake.y == part[1]:
                    snake = Snake()
                    
            direction = snake.vision(apple)
            snake.pre_direction = snake.direction
            snake.decision(direction)
       
            if snake.collision_with_wall(snake.direction):
                direction = (snake.direction + 1) % 4
                if snake.collision_with_wall(direction):
                    direction = (snake.direction - 1) % 4
                    if snake.collision_with_wall(direction):
                        snake = Snake()

                snake.direction = direction
                
            snake.move()
     
            self.display.fill(self.color)
            pygame.draw.rect(self.display, Color.black, ((0, 0), (game_w, game_h)), wall_offset)

            apple.draw(self.display)
            snake.draw(self.display)

            score = self.font.render(f'Score: {snake.score}', True, Color.black)
            score_rect = score.get_rect(center=(game_w / 2, game_h - 10))
            self.display.blit(score, score_rect)

            pygame.display.update()
            clock.tick(fps)


if __name__ == "__main__":
    game = Game()
    game.play()
