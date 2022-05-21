"""
need a 10x10 grid
for each episode, random init grid with cans and robot

each step during an episode:

observe the state
choose an action (greedy action selection)
perform action
recieve reward if able
observe new state
update q matrix

q-matrix entry = current s/a value + (reward + discounted(random(successor s/a value) - current s/a value)) 

state string: 5 cells (current, North, south, east, west) 1 = can, 2 = wall, 0 = empty eg 10012
actions = n(move north), s(move south), e(move east), w(move west), p(pick up can)
state-action key = 10002 + action, eg 

objects:
q-matrix: dict for q(sa):value
episode: steps, total reward
simulation: grid, robot, 
grid: x by x cells
cell: has a state (can/no can with 1/2 chance of either)
robot: has a sensor state (eg 10102), current coordinate, q-matrix: chooses own actions

state/action entry: just a key/value pair in dict. key = state-action (eg 10021P), value = reqard

flow:

start in random cell
observe state
choose actions with epsilon greedy
perform action
get reward if able
observe new state
update old q-state with new value

for each episode:
    run x number of steps
"""
import random
from typing import List


SIZE = 10
EPISODE_COUNT = 5000
STEP_COUNT = 200
PROBABILITY = 0.2
DISCOUNT = 0.9


class QMatrix:
    def __init__(self) -> None:
        self.sa_dict = self.initialize()

    def __str__(self) -> str:
        return self.sa_dict

    def initialize(self):
        return {}


class Grid:
    def __init__(self, size) -> None:
        self.size = size
        self.grid = self.initialize()

    def initialize(self):
        grid = [[]] * self.size
        for column in grid:
            for _ in range(0, self.size):
                column.append(Cell())

        return grid

    def calculate_state(self, xy):
        x = xy[0]
        y = xy[1]

        surrounding_cells = [
            # North, south, east, west
            [-1, 0],
            [1, 0],
            [0, 1],
            [0, -1],
        ]

        state = f"{self.grid[x][y]}"
        for coordinate in surrounding_cells:
            new_x = x + coordinate[0]
            new_y = y + coordinate[1]

            if 0 <= new_x < SIZE and 0 <= new_y < SIZE:
                state += f"{self.grid[x+coordinate[0]][y+coordinate[1]]}"
            else:
                state += f"2"

        return state


class Robot:
    def __init__(self, xy: List, grid: Grid) -> None:
        self.current_xy = xy
        self.current_state = grid.calculate_state(self.xy)
        self.q_matrix = {}
        self.actions = ["p", "n", "s", "e", "w"]

    def progress_state(self, grid: Grid):
        state_actions = []
        state = None
        """
        calculaue all possible sucessor states
        check if any exist in q matrix
        if not, add to matrix with value = 0
        if so, add state to list of possibler actions
        """
        for action in self.actions:
            state = self.current_state + action
            state_value = self.q_matrix.setdefault(state, 0)
            state_actions[state] = state_value

        selected_state_action = self.select_state_action(state_actions)
        reward = self.calculate_reward(selected_state_action)
        self.q_matrix[selected_state_action] += reward
        self.xy = self.determine_next_position(selected_state_action)
        self.current_state = grid.calculate_state(self.xy)

    def select_state_action(self):
        """find best reward or random select"""
        pass
    
    def calculate_reward(self, state_action):
        """calculate reward for selected state-action"""
        pass
    
    def determine_next_position(self, state_action):
        """determine return xy resulting from selected state action"""
        pass


class Cell:
    def __init__(self) -> None:
        self.has_can = random.randint(0, 1)

    def __str__(self) -> str:
        return str(self.has_can)


class Simulation:
    def __init__(self, robot) -> None:
        self.robot = robot
        self.episode_count = EPISODE_COUNT
        self.episodes = []

    def run_simulation(self):
        count = self.episode_count
        while count:
            episode = Episode()
            episode.run_episode(self.robot)
            self.episodes.append(episode)
            count -= 1


class Episode:
    def __init__(self) -> None:
        self.grid = Grid(SIZE)
        self.step_count = STEP_COUNT
        self.total_reward = 0

    def run_episode(self, robot: Robot):
        start_xy = [random.randint(0, SIZE - 1), random.randint(0, SIZE - 1)]
        robot.xy = start_xy
        robot.current_state = self.grid.calculate_state(robot.xy)

        while self.step_count:
            self.total_reward += robot.progress_state(self.grid)
            self.step_count -= 1


rob = Robot()
rob.current_xy = [0, 0]
grid = Grid(SIZE)
print(grid.calculate_state(rob.current_xy))
