import time
import pygame

pygame.init()
pygame.joystick.init()
joystick = pygame.joystick.Joystick(0)
joystick.init()

while True:
    pygame.event.pump()
    print([joystick.get_axis(i) for i in range(6)])
    time.sleep(0.1)