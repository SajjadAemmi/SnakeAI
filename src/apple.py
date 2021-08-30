import random
import pygame
from .color import Color


class Apple:
    def __init__(self, config):
        self.x = random.randint(40, config.game_w - 40) // 10 * 10
        self.y = random.randint(40, config.game_h - 40) // 10 * 10
        self.radius = 5
        self.color = Color.red

    def draw(self, display):
        return pygame.draw.circle(display, self.color, [self.x + 5, self.y + 5], self.radius)
