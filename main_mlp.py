import pygame
import random
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from snake import Snake
from apple import Apple
from config import *


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(20, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

model = Model()
model.load_state_dict(torch.load('snake.pt', map_location=torch.device('cpu')))

def add_data():
    w0 = snake.y - wall_offset  # up
    w1 = game_w - wall_offset - snake.x  # right
    w2 = game_h - wall_offset - snake.y  # down
    w3 = snake.x - wall_offset  # left

    if snake.x == apple.x and snake.y > apple.y:
        a0 = abs(snake.x - apple.x) + abs(snake.y - apple.y)
    else:
        a0 = 0

    if abs(snake.x - apple.x) == abs(snake.y - apple.y) and snake.x < apple.x and snake.y > apple.y:
        a01 = abs(snake.x - apple.x) + abs(snake.y - apple.y)
    else:
        a01 = 0

    if snake.x < apple.x and snake.y == apple.y:
        a1 = abs(snake.x - apple.x) + abs(snake.y - apple.y)
    else:
        a1 = 0

    if abs(snake.x - apple.x) == abs(snake.y - apple.y) and snake.x < apple.x and snake.y < apple.y:
        a12 = abs(snake.x - apple.x) + abs(snake.y - apple.y)
    else:
        a12 = 0

    if snake.x == apple.x and snake.y < apple.y:
        a2 = abs(snake.x - apple.x) + abs(snake.y - apple.y)
    else:
        a2 = 0

    if abs(snake.x - apple.x) == abs(snake.y - apple.y) and snake.x > apple.x and snake.y < apple.y:
        a23 = abs(snake.x - apple.x) + abs(snake.y - apple.y)
    else:
        a23 = 0

    if snake.x > apple.x and snake.y == apple.y:
        a3 = abs(snake.x - apple.x) + abs(snake.y - apple.y)
    else:
        a3 = 0

    if abs(snake.x - apple.x) == abs(snake.y - apple.y) and snake.x > apple.x and snake.y > apple.y:
        a30 = abs(snake.x - apple.x) + abs(snake.y - apple.y)
    else:
        a30 = 0

    for part in snake.body:
        if snake.x == part[0] and snake.y > part[1]:
            b0 = abs(snake.x - part[0]) + abs(snake.y - part[1])
            break
    else:
        b0 = 0

    for part in snake.body:
        if snake.x < part[0] and snake.y > part[1]:
            b01 = abs(snake.x - part[0]) + abs(snake.y - part[1])
            break
    else:
        b01 = 0

    for part in snake.body:
        if snake.x < part[0] and snake.y == part[1]:
            b1 = abs(snake.x - part[0]) + abs(snake.y - part[1])
            break
    else:
        b1 = 0

    for part in snake.body:
        if snake.x < part[0] and snake.y < part[1]:
            b12 = abs(snake.x - part[0]) + abs(snake.y - part[1])
            break
    else:
        b12 = 0

    for part in snake.body:
        if snake.x == part[0] and snake.y < part[1]:
            b2 = abs(snake.x - part[0]) + abs(snake.y - part[1])
            break
    else:
        b2 = 0

    for part in snake.body:
        if snake.x > part[0] and snake.y < part[1]:
            b23 = abs(snake.x - part[0]) + abs(snake.y - part[1])
            break
    else:
        b23 = 0

    for part in snake.body:
        if snake.x > part[0] and snake.y == part[1]:
            b3 = abs(snake.x - part[0]) + abs(snake.y - part[1])
            break
    else:
        b3 = 0

    for part in snake.body:
        if snake.x > part[0] and snake.y > part[1]:
            b30 = abs(snake.x - part[0]) + abs(snake.y - part[1])
            break
    else:
        b30 = 0

    return np.array([w0, w1, w2, w3,
                    a0, a01, a1, a12, a2, a23, a3, a30,
                    b0, b01, b1, b12, b2, b23, b3, b30,
                    ], dtype=np.float32)


class Game:
    def __init__(self):        
        self.display = pygame.display.set_mode((game_w, game_h))
        pygame.display.set_caption('snake')
        self.color = Color.white        
        pygame.font.init()
        self.font = pygame.font.SysFont("calibri", 10)

    def play(self):
        global snake, apple

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

            snake.x_change_old, snake.y_change_old = snake.x, snake.y

            with torch.no_grad():
                data = add_data()
                data = data.reshape(1, 20)
                data = torch.tensor(data)
                result = model(data)
                snake.direction = np.argmax(result)

            snake.move()
     
            self.display.fill(self.color)
            pygame.draw.rect(self.display, Color.black, ((0, 0), (game_w, game_h)), 10)

            apple.draw(self.display)
            snake.draw(self.display)

            if snake.x < 0 or snake.y < 0 or snake.x > game_w or snake.y > game_h:
                self.play()

            score = self.font.render(f'Score: {snake.score}', True, Color.black)
            score_rect = score.get_rect(center=(game_w / 2, game_h - 10))
            self.display.blit(score, score_rect)

            pygame.display.update()
            clock.tick(30)  # fps


if __name__ == "__main__":
    game = Game()
    game.play()
