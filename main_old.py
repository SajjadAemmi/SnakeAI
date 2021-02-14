import pygame
import random
from math import sqrt

class Color:
    red = (255, 0, 0)
    green = (0, 255, 0)
    green_dark = (0, 127, 0)
    white = (255, 255, 255)
    black = (0, 0, 0)


class Apple:
    def __init__(self):
        self.x = random.randint(50, game.w-50) // 10 * 10
        self.y = random.randint(50, game.h-50) // 10 * 10
        self.radius = 5
        self.color = Color.red

    def draw(self):
        return pygame.draw.circle(game.display, self.color, [self.x + 5, self.y + 5], self.radius)


class Snake:
    def __init__(self):
        self.x = game.w // 2
        self.y = game.h // 2
        self.body = []
        self.body_color_1 = Color.green
        self.body_color_2 = Color.green_dark
        self.w = 10
        self.h = 10
        self.direction = 1
        self.x_change = 1
        self.y_change = 0
        self.speed = 10
        self.score = 0

    def eat(self):
        self.score += 1

    def move(self):
        self.body.append([self.x, self.y])
        if len(self.body) > self.score:
            del (self.body[0])

        self.x += self.x_change * self.speed
        self.y += self.y_change * self.speed

    def draw(self):
        pygame.draw.rect(game.display, self.body_color_1, [self.x, self.y, self.w, self.h])
        
        for index, item in enumerate(self.body):
            if index % 2 == 0:
                pygame.draw.rect(game.display, self.body_color_1, [item[0], item[1], self.w, self.h])
            else:
                pygame.draw.rect(game.display, self.body_color_2, [item[0], item[1], self.w, self.h])


def collision_with_body(direction):

    for part in snake.body:

        if direction == 0:
            if abs(snake.x - part[0]) < game.wall_offset and abs(snake.y - 8 - part[1]) == 0:
                return True
            if abs(snake.x - part[0]) == 0 and abs(snake.y - 8 - part[1]) < game.wall_offset:
                return True

        if direction == 1:
            if abs(snake.x + 8 - part[0]) < game.wall_offset and abs(snake.y - part[1]) == 0:
                return True
            if abs(snake.x + 8 - part[0]) == 0 and abs(snake.y - part[1]) < game.wall_offset:
                return True

        if direction == 2:
            if abs(snake.x - part[0]) < game.wall_offset and abs(snake.y + 8 - part[1]) == 0:
                return True
            if abs(snake.x - part[0]) == 0 and abs(snake.y + 8 - part[1]) < game.wall_offset:
                return True

        if direction == 3:
            if abs(snake.x - 8 - part[0]) < game.wall_offset and abs(snake.y - part[1]) == 0:
                return True
            if abs(snake.x - 8 - part[0]) == 0 and abs(snake.y - part[1]) < game.wall_offset:
                return True

    return False


def collision_with_wall(direction):

    if direction == 0:
        if snake.y - 8 >= game.wall_offset:
            return False

    elif direction == 1:
        if snake.x + 8 <= game.w - game.wall_offset:
            return False

    elif direction == 2:
        if snake.y + 8 <= game.h - game.wall_offset:
            return False

    elif direction == 3:
        if snake.x - 8 >= game.wall_offset:
            return False

    return True


def distance(direction, method):

    if direction == 0:
        x = snake.x
        y = snake.y - 8

    elif direction == 1:
        x = snake.x + 8
        y = snake.y

    elif direction == 2:
        x = snake.x
        y = snake.y + 8

    elif direction == 3:
        x = snake.x - 8
        y = snake.y

    if method == 'manhattan':
        return abs(x - apple.x) + abs(y - apple.y)
    elif method == 'euclidean':
        return sqrt(abs(x - apple.x) ** 2 + abs(y - apple.y) ** 2)
    elif method == 'chess':
        return max(abs(x - apple.x), abs(y - apple.y))


class Game:
    def __init__(self):        
        self.w = 400
        self.h = 300
        self.wall_offset = 10
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('snake')
        self.color = Color.white        
        pygame.font.init()
        self.font = pygame.font.SysFont("calibri", 10)

    def play(self):
        global snake
        global apple
        
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()

            # Detect collision with apple
            if snake.x == apple.x and snake.y == apple.y:
                snake.eat()
                apple = Apple()
                    
            childes = []
            for i in range(4):
                if not collision_with_body(i) and not collision_with_wall(i):
                    childes.append({'direction': i, 'distance': distance(i, 'manhattan')})

            if len(childes) == 0:
                snake = Snake()
                continue

            childes = sorted(childes, key=lambda k: k['distance'])
            child = childes[0]

            if len(childes) > 1:
                if abs(childes[0]['direction'] - snake.direction) == 2:
                    child = childes[1]
            
            # up
            if child['direction'] == 0 and snake.direction != 2:
                snake.x_change = 0
                snake.y_change = -1
                snake.direction = 0

            # right
            elif child['direction'] == 1 and snake.direction != 3:
                snake.x_change = 1
                snake.y_change = 0
                snake.direction = 1

            # down
            elif child['direction'] == 2 and snake.direction != 0:
                snake.x_change = 0
                snake.y_change = 1
                snake.direction = 2

            # left
            elif child['direction'] == 3 and snake.direction != 1:
                snake.x_change = -1
                snake.y_change = 0
                snake.direction = 3

            snake.move()
     
            self.display.fill(self.color)
            pygame.draw.rect(self.display, Color.black, ((0, 0), (self.w, self.h)), 10)

            apple.draw()
            snake.draw()

            if snake.x < 0 or snake.y < 0 or snake.x > self.w or snake.y > self.h:
                self.play()

            score = self.font.render(f'Score: {snake.score}', True, Color.black)
            score_rect = score.get_rect(center=(self.w / 2, self.h - 10))
            self.display.blit(score, score_rect)

            pygame.display.update()
            clock.tick(30)  # fps


if __name__ == "__main__":
    game = Game()
    snake = Snake()
    apple = Apple()
    game.play()
