import random
import pygame
from config import *


class Apple:
    def __init__(self):
        self.x = random.randint(40, game_w - 40) // 10 * 10
        self.y = random.randint(40, game_h - 40) // 10 * 10
        self.radius = 5
        self.color = Color.red

    def draw(self, display):
        return pygame.draw.circle(display, self.color, [self.x + 5, self.y + 5], self.radius)
