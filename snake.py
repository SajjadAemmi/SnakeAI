import random
import pygame
from config import *


class Snake:
    def __init__(self):
        self.x = game_w // 2
        self.y = game_h // 2
        self.body = []
        self.body_color_1 = Color.green
        self.body_color_2 = Color.green_dark
        self.w = 10
        self.h = 10
        self.pre_direction = None
        self.direction = random.randint(0, 3)
        self.x_change = 0
        self.y_change = 0
        self.speed = 10
        self.score = 0

    def eat(self):
        self.score += 1

    def move(self):
        if self.direction == 0:
            self.x_change = 0
            self.y_change = -1

        elif self.direction == 1:
            self.x_change = 1
            self.y_change = 0

        elif self.direction == 2:
            self.x_change = 0
            self.y_change = 1

        elif self.direction == 3:
            self.x_change = -1
            self.y_change = 0

        self.body.append([self.x, self.y])
        if len(self.body) > self.score:
            del (self.body[0])

        self.x += self.x_change * self.speed
        self.y += self.y_change * self.speed

    def draw(self, display):
        pygame.draw.rect(display, self.body_color_1, [self.x, self.y, self.w, self.h])
        
        for index, item in enumerate(self.body):
            if index % 2 == 0:
                pygame.draw.rect(display, self.body_color_1, [item[0], item[1], self.w, self.h])
            else:
                pygame.draw.rect(display, self.body_color_2, [item[0], item[1], self.w, self.h])

    def collision_with_body(self, direction):

        for part in snake.body:

            if direction == 0:
                if abs(snake.x - part[0]) < game.wall_offset and abs(snake.y - 8 - part[1]) == 0:
                    return True
                if abs(snake.x - part[0]) == 0 and abs(snake.y - 10 - part[1]) < game.wall_offset:
                    return True

            if direction == 1:
                if abs(snake.x + 10 - part[0]) < game.wall_offset and abs(snake.y - part[1]) == 0:
                    return True
                if abs(snake.x + 10 - part[0]) == 0 and abs(snake.y - part[1]) < game.wall_offset:
                    return True

            if direction == 2:
                if abs(snake.x - part[0]) < game.wall_offset and abs(snake.y + 10 - part[1]) == 0:
                    return True
                if abs(snake.x - part[0]) == 0 and abs(snake.y + 10 - part[1]) < game.wall_offset:
                    return True

            if direction == 3:
                if abs(snake.x - 10 - part[0]) < game.wall_offset and abs(snake.y - part[1]) == 0:
                    return True
                if abs(snake.x - 10 - part[0]) == 0 and abs(snake.y - part[1]) < game.wall_offset:
                    return True

        return False

    def collision_with_wall(self, direction):

        if direction == 0:
            if self.y - 10 > wall_offset:
                return False

        elif direction == 1:
            if self.x + 10 < game_w - wall_offset:
                return False

        elif direction == 2:
            if self.y + 10 < game_h - wall_offset:
                return False

        elif direction == 3:
            if self.x - 10 > wall_offset:
                return False

        return True

    def distance(self, direction, method, apple):

        if direction == 0:
            x = self.x
            y = self.y - 8

        elif direction == 1:
            x = self.x + 8
            y = self.y

        elif direction == 2:
            x = self.x
            y = self.y + 8

        elif direction == 3:
            x = self.x - 8
            y = self.y

        if method == 'manhattan':
            return abs(x - apple.x) + abs(y - apple.y)
        elif method == 'euclidean':
            return sqrt(abs(x - apple.x) ** 2 + abs(y - apple.y) ** 2)
        elif method == 'chess':
            return max(abs(x - apple.x), abs(y - apple.y))

    def vision(self, apple):
        # up
        if self.x == apple.x and self.y > apple.y:
            for part in self.body:
                if self.x == part[0] and self.y > part[1] > apple.y:
                    break
            else:
                return '0'

        # up right
        if abs(self.x - apple.x) == abs(self.y - apple.y) and self.x < apple.x and self.y > apple.y:
            for part in self.body:
                if abs(self.x - part[0]) == abs(self.y - part[1]) and self.x < part[0] < apple.x and self.y > part[1] > apple.y:
                    break
            else:
                return '01'

        # right
        if self.x < apple.x and self.y == apple.y:
            for part in self.body:
                if self.x < part[0] < apple.x and self.y == part[1]:
                    break
            else:
                return '1'

        # down right
        if abs(self.x - apple.x) == abs(self.y - apple.y) and self.x < apple.x and self.y < apple.y:
            for part in self.body:
                if abs(self.x - part[0]) == abs(self.y - part[1]) and self.x < part[0] < apple.x and self.y < part[1] < apple.y:
                    break
            else:
                return '12'

        # down
        if self.x == apple.x and self.y < apple.y:
            for part in self.body:
                if self.x == part[0] and self.y < part[1] < apple.y:
                    break
            else:
                return '2'

        # down left
        if abs(self.x - apple.x) == abs(self.y - apple.y) and self.x > apple.x and self.y < apple.y:
            for part in self.body:
                if abs(self.x - part[0]) == abs(self.y - part[1]) and self.x > part[0] > apple.x and self.y < part[1] < apple.y:
                    break
            else:
                return '23'

        if self.x > apple.x and self.y == apple.y:
            for part in self.body:
                if self.x > part[0] > apple.x and self.y == part[1]:
                    break
            else:
                return '3'
        
        if abs(self.x - apple.x) == abs(self.y - apple.y) and self.x > apple.x and self.y > apple.y:
            for part in self.body:
                if abs(self.x - part[0]) == abs(self.y - part[1]) and self.x > part[0] > apple.x and self.y > part[1] > apple.y:
                    break
            else:
                return '30'

        return None

    def decision(self, direction):
        # up
        if direction == '0':
            if self.direction != 2:
                self.direction = 0

        # up right
        elif direction == '01':
            if self.direction != 2:
                self.direction = 0
            elif self.direction != 3:
                self.direction = 1

        # right
        elif direction == '1':
            if self.direction != 3:
                self.direction = 1

        # down right
        elif direction == '12':
            if self.direction != 3:
                self.direction = 1
            elif self.direction != 0:
                self.direction = 2

        # down
        elif direction == '2':
            if self.direction != 0:
                self.direction = 2
        
        # down left
        elif direction == '23':
            if self.direction != 0:
                self.direction = 2
            elif self.direction != 1:
                self.direction = 3

        # left
        elif direction == '3':
            if self.direction != 1:
                self.direction = 3

        # up left
        elif direction == '30':
            if self.direction != 1:
                self.direction = 3
            elif self.direction != 2:
                self.direction = 0
