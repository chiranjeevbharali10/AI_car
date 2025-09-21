# car.py
import pygame
import math

class Car:
    def __init__(self, x, y, angle=0, width=30, height=60, color_front=(255, 0, 0), color_back=(0, 0, 255)):
        self.x = x
        self.y = y
        self.angle = angle  # degrees, 0 means facing right
        self.width = width
        self.height = height
        self.color_front = color_front
        self.color_back = color_back
        self.speed = 0

    def draw(self, surface):
        """Draw the car with rotation applied"""
        box = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(box, self.color_front, (0, 0, self.width, self.height // 2))
        pygame.draw.rect(box, self.color_back, (0, self.height // 2, self.width, self.height // 2))

        rotated_car = pygame.transform.rotate(box, -self.angle)
        rect = rotated_car.get_rect(center=(self.x, self.y))
        surface.blit(rotated_car, rect.topleft)

    def move_forward(self, step=1):
        """Move the car forward in the direction of its current angle"""
        rad = math.radians(self.angle)
        self.x += step * math.cos(rad)
        self.y -= step * math.sin(rad)  # minus because pygame's y is inverted

    def rotate_towards(self, dx, dy):
        """Rotate the car to face direction (dx, dy)"""
        if dx != 0 or dy != 0:
            self.angle = math.degrees(math.atan2(-dy, dx))

    def set_position(self, x, y):
        self.x = x
        self.y = y
