import enum
import random

import numpy as np
import pygame
from numba import njit


class TileTypes(enum.Enum):
    EMPTY = 0
    SNAKE_HEAD = 11
    SNAKE_BODY = 12
    WALL = 13
    FOOD = 100


@njit
def get_rand_empty_pos(game: [[int]]) -> (int, int):
    """
    Returns a random empty position in the game_array.
    """
    empty_positions = np.where(game == TileTypes.EMPTY.value)
    if empty_positions[0].shape[0] - 1 == 0:
        return -1, -1
    index = random.randint(0, empty_positions[0].shape[0] - 1)
    return empty_positions[0][index], empty_positions[1][index]


@njit
def generate_game(food_pos=None, snake_pos=None, game_size=(300, 300), direction=None):
    """
    Generates a game_array board with a snake_data and food.
    """
    game = np.zeros(game_size, dtype=np.int8)
    # make walls
    for y in range(game.shape[0]):
        for x in range(game.shape[1]):
            if y == 0 or y == game.shape[0] - 1 or x == 0 or x == game.shape[1] - 1:
                game[y, x] = TileTypes.WALL.value
    snake = []
    if food_pos is None:
        food_pos = get_rand_empty_pos(game)
        game[food_pos[0], food_pos[1]] = TileTypes.FOOD.value
    if snake_pos is None:
        snake_pos = get_rand_empty_pos(game)
        game[snake_pos[0], snake_pos[1]] = TileTypes.SNAKE_HEAD.value
        snake.append(snake_pos)

    return game, food_pos, np.array(snake)


def render(surface: pygame.Surface, game_array: [[int]], color_map_input: dict = None):
    """
    Renders the game_array board to the surface.
    :param surface: pygame surface
    :param game_array: game_array board
    :param color_map_input: color map
    :return:
    """
    color_map = {
        TileTypes.EMPTY.value: (0, 0, 0),
        TileTypes.SNAKE_HEAD.value: (0, 255, 0),
        TileTypes.SNAKE_BODY.value: (0, 255, 0),
        TileTypes.WALL.value: (255, 255, 255),
        TileTypes.FOOD.value: (255, 255, 0)

    }
    if color_map_input is not None:
        color_map.update(color_map_input)
    grid_size = surface.get_width() / game_array.shape[1]
    for y in range(game_array.shape[0]):
        for x in range(game_array.shape[1]):
            tile_type = game_array[y, x]
            if tile_type == TileTypes.EMPTY.value:
                continue
            pygame.draw.rect(surface, color_map[game_array[y, x]], (x * grid_size, y * grid_size, grid_size, grid_size), 2)


def move_snake(game_array: [[int]], snake_data: [[int]], direction, food_pos):
    """
    Moves the snake_data in the given direction.
    """
    # get head position
    head_pos = snake_data[-1]
    tail_pos = snake_data[0]
    eaten = False
    dead = False
    # get new head position
    new_head_pos = head_pos[0] + direction[0], head_pos[1] + direction[1]
    # check if new head position is valid
    if new_head_pos[0] < 0 or new_head_pos[0] >= game_array.shape[0] \
            or new_head_pos[1] < 0 or new_head_pos[1] >= game_array.shape[1]:
        dead = True
    # check if new head position is empty
    if game_array[new_head_pos] == TileTypes.EMPTY.value:
        # move head
        game_array[new_head_pos] = TileTypes.SNAKE_HEAD.value
        game_array[head_pos[0], head_pos[1]] = TileTypes.SNAKE_BODY.value
        # remove tail
        game_array[tail_pos[0], tail_pos[1]] = TileTypes.EMPTY.value
        snake_data = np.concatenate([snake_data, [new_head_pos]], axis=0)
        snake_data = snake_data[1:]
    # check if new head position is food
    elif game_array[new_head_pos] == TileTypes.FOOD.value:
        # move head
        game_array[new_head_pos] = TileTypes.SNAKE_HEAD.value
        game_array[head_pos[0], head_pos[1]] = TileTypes.SNAKE_BODY.value
        # spawn new food
        food_pos = get_rand_empty_pos(game_array)
        game_array[food_pos] = TileTypes.FOOD.value
        snake_data = np.concatenate([snake_data, [new_head_pos]], axis=0)
        eaten = True
    # check if new head position is snake_data
    elif game_array[new_head_pos] == TileTypes.SNAKE_BODY.value:
        dead = True
    # check if
    elif game_array[new_head_pos] == TileTypes.WALL.value:
        dead = True
    return snake_data, dead, food_pos, eaten
