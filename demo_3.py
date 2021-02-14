import pygame
import random
from math import sqrt

f = open('snake.csv', 'w')
# f = open('snake_test.csv', 'w')

f.write('w0,w1,w2,w3,a0,a1,a2,a3,b0,b1,b2,b3,direction' + '\n')

def add_data():

    w0 = snake.y - game.wall_offset  # up
    w1 = game.w - game.wall_offset - snake.x  # right
    w2 = game.h - game.wall_offset - snake.y  # down
    w3 = snake.x - game.wall_offset  # left

    if snake.x == apple.x and snake.y > apple.y:
        a0 = abs(snake.x - apple.x) + abs(snake.y - apple.y)
    else:
        a0 = 0

    if snake.x < apple.x and snake.y == apple.y:
        a1 = abs(snake.x - apple.x) + abs(snake.y - apple.y)
    else:
        a1 = 0

    if snake.x == apple.x and snake.y < apple.y:
        a2 = abs(snake.x - apple.x) + abs(snake.y - apple.y)
    else:
        a2 = 0

    if snake.x > apple.x and snake.y == apple.y:
        a3 = abs(snake.x - apple.x) + abs(snake.y - apple.y)
    else:
        a3 = 0

    for part in snake.body:
        if snake.x == part[0] and snake.y > part[1]:
            b0 = abs(snake.x - part[0]) + abs(snake.y - part[1])
            break
    else:
        b0 = 0

    for part in snake.body:
        if snake.x < part[0] and snake.y == part[1]:
            b1 = abs(snake.x - part[0]) + abs(snake.y - part[1])
            break
    else:
        b1 = 0

    for part in snake.body:
        if snake.x == part[0] and snake.y < part[1]:
            b2 = abs(snake.x - part[0]) + abs(snake.y - part[1])
            break
    else:
        b2 = 0

    for part in snake.body:
        if snake.x > part[0] and snake.y == part[1]:
            b3 = abs(snake.x - part[0]) + abs(snake.y - part[1])
            break
    else:
        b3 = 0
        
    direction = snake.direction

    if snake.direction != snake.pre_direction or random.random() < 0.05:

        f.write(",".join([str(w0), str(w1), str(w2), str(w3),
                        str(a0), str(a1), str(a2), str(a3), 
                        str(b0), str(b1), str(b2), str(b3),
                        # str(hu), str(hr), str(hd), str(hl), 
                        # str(tu), str(tr), str(td), str(tl),
                        # str(u), str(r), str(d), str(l),
                        str(direction)]) + '\n')
        return True

    else:
        return False


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
        if snake.y - 8 > game.wall_offset:
            return False

    elif direction == 1:
        if snake.x + 8 < game.w - game.wall_offset:
            return False

    elif direction == 2:
        if snake.y + 8 < game.h - game.wall_offset:
            return False

    elif direction == 3:
        if snake.x - 8 > game.wall_offset:
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


def vision():
    if snake.x == apple.x and snake.y > apple.y:
        for part in snake.body:
            if snake.x == part[0] and snake.y > part[1] > apple.y:
                break
        else:
            return 0

    if snake.x < apple.x and snake.y == apple.y:
        for part in snake.body:
            if snake.x < part[0] < apple.x and snake.y == part[1]:
                break
        else:
            return 1

    if snake.x == apple.x and snake.y < apple.y:
        for part in snake.body:
            if snake.x == part[0] and snake.y < part[1] < apple.y:
                break
        else:
            return 2

    if snake.x > apple.x and snake.y == apple.y:
        for part in snake.body:
            if snake.x > part[0] > apple.x and snake.y == part[1]:
                break
        else:
            return 3
    
    return -1

rows = 0

class Game:
    def __init__(self):        
        self.w = 400
        self.h = 300
        self.wall_offset = 16

    def play(self):
        global snake
        global apple
        global rows
        
        snake = Snake()

        while rows < 1000000:

            # Detect collision with apple
            if snake.x == apple.x and snake.y == apple.y:
                snake.eat()
                apple = Apple()

            # collision with body
            for part in snake.body:
                if snake.x == part[0] and snake.y == part[1]:
                    snake = Snake()
                    
            direction = vision()

            snake.pre_direction = snake.direction

            # up
            if direction == 0 and snake.direction != 2:
                snake.direction = 0

            # right
            elif direction == 1 and snake.direction != 3:
                snake.direction = 1

            # down
            elif direction == 2 and snake.direction != 0:
                snake.direction = 2

            # left
            elif direction == 3 and snake.direction != 1:
                snake.direction = 3
            
            elif collision_with_wall(snake.direction):
                direction = (snake.direction + 1) % 4
                if collision_with_wall(direction):
                    direction = (snake.direction - 1) % 4
                    if collision_with_wall(direction):
                        snake = Snake()
                
                snake.direction = direction

            if add_data():
                rows += 1
                if rows % 100000 == 0:
                    print(rows)
                
            snake.move()


if __name__ == "__main__":
    game = Game()
    snake = Snake()
    apple = Apple()
    game.play()
    f.close()