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
from re import I
from time import sleep
from numpy import random
from typing import List


SIZE = 10
EPISODE_COUNT = 5000
STEP_COUNT = 200
PROBABILITY = 0.2
DISCOUNT = 0.9
EPSILON = 0.1
N_VALUE = 0.2
DISCOUNT_RATE = 0.9


class Grid:
    def __init__(self, size) -> None:
        self.size = size
        self.grid = self.initialize()

    def __str__(self) -> str:
        return self.grid

    def initialize(self):
        grid = [[]] * self.size
        for column in grid:
            for _ in range(0, self.size):
                cell = Cell()
                # print(cell)
                column.append(cell)

        return grid

    def calculate_state(self, xy: List):
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
                state += f"{self.grid[x+coordinate[0]][y+coordinate[1]].has_can}"
            else:
                state += f"2"

        return state


class Robot:
    def __init__(self, xy: List, grid: Grid) -> None:
        self.current_xy = xy
        self.current_state = grid.calculate_state(self.current_xy)
        self.q_matrix = {}
        # pick up can, move north, move south, move eats, move west
        self.actions = ["p", "n", "s", "e", "w"]
        self.discount_rate = DISCOUNT_RATE

    def progress_state(self, grid: Grid, epsilon):
        state_actions = {}
        state_action = None
        """
        calculaue all possible sucessor states
        check if any exist in q matrix
        if not, add to matrix with value = 0
        if so, add state to list of possibler actions
        """
        for action in self.actions:
            state_action = self.current_state + action
            state_value = self.q_matrix.setdefault(state_action, 0)
            state_actions[state_action] = state_value
            # print(state, state_value)

        selected_state_action = self.select_state_action(state_actions, epsilon)
        # successor_state_actions = self.calculate_successor_state_actions(selected_state_action, grid)
        state_value = self.calculate_state_value(selected_state_action, grid)
        self.q_matrix[selected_state_action] += state_value
        self.current_xy = self.determine_next_position(selected_state_action)
        self.current_state = grid.calculate_state(self.current_xy)

        # print(selected_state_action, state_value)
        return state_value

    def select_state_action(self, state_actions, epsilon):
        """find best reward or random select
        1-e chance of selecting best option (ties broken at random)
        else random select
        """
        selected_state_action = None
        probability_of_best_choice = 1 - epsilon
        random_choice = round(random.uniform(0, 1), 2)
        # print(f"random choice: {random_choice} {probability_of_best_choice}")

        if random_choice > probability_of_best_choice:
            selected_state_action = random.choice(list(state_actions), 1)
        else:
            max_value = state_actions[max(state_actions, key=state_actions.get)]
            # print(f"max val: {max_value}")
            best_actions = [
                state_action
                for state_action in state_actions
                if state_actions.get(state_action) == max_value
            ]
            # print(best_actions)
            selected_state_action = random.choice(best_actions, 1)

        return selected_state_action[0]

    def calculate_successor_state_actions(self, state_action, grid: Grid):
        move_modifiers = [
            # North, south, east, west
            [-1, 0],
            [1, 0],
            [0, 1],
            [0, -1],
        ]
        state_actions = {}
        action = state_action[-1]
        x = self.current_xy[0]
        y = self.current_xy[1]
        if action != "p":
            if state_action[self.actions.index(action)] != "2":
                x += move_modifiers[self.actions.index(action)][0]
                y += move_modifiers[self.actions.index(action)][1]

        next_state = grid.calculate_state([x, y])
        for action in self.actions:
            state = next_state + action
            state_value = self.q_matrix.setdefault(state, 0)
            state_actions[state] = state_value
            # print(state, state_value)

        return state_actions

    def calculate_state_value(self, state_action, grid: Grid):
        """calculate reward for selected state-action"""
        current_value = self.q_matrix[state_action]
        state_reward = self.calculate_state_reward(state_action, grid)
        max_successor_value = self.calculate_max_successor_value(state_action, grid)
        return state_reward + N_VALUE * (
            self.discount_rate * max_successor_value - current_value
        )

    def calculate_state_reward(self, state_action, grid: Grid):
        state_reward = 0
        action = state_action[-1]
        if action == "p":
            x = self.current_xy[0]
            y = self.current_xy[1]

            if grid.grid[x][y].has_can == 1:
                state_reward = 10
                grid.grid[x][y].has_can = 0
            else:
                state_reward = -1
        else:
            if state_action[self.actions.index(action)] == "2":
                state_reward = -5

        return state_reward

    def calculate_max_successor_value(self, state_action, grid: Grid):
        state_actions = self.calculate_successor_state_actions(state_action, grid)
        max_value = state_actions[max(state_actions, key=state_actions.get)]
        return max_value

    def calculate_successor_state_actions(self, state_action, grid: Grid):
        state_actions = {}
        xy = self.determine_next_position(state_action)
        next_state = grid.calculate_state(xy)
        for action in self.actions:
            state = next_state + action
            state_value = self.q_matrix.setdefault(state, 0)
            state_actions[state] = state_value
            # print(state, state_value)

        return state_actions

    def determine_next_position(self, state_action):
        """determine return xy resulting from selected state action"""
        move_modifiers = [
            # North, south, east, west
            [-1, 0],
            [1, 0],
            [0, 1],
            [0, -1],
        ]
        action = state_action[-1]
        x = self.current_xy[0]
        y = self.current_xy[1]
        if action != "p":
            if state_action[self.actions.index(action)] != "2":
                x += move_modifiers[self.actions.index(action) - 1][0]
                y += move_modifiers[self.actions.index(action) - 1][1]

        return [x, y]


class Cell:
    def __init__(self) -> None:
        self.has_can = random.randint(0, 2)

    def __str__(self) -> str:
        return str(self.has_can)


class Simulation:
    def __init__(self) -> None:
        self.robot = None
        self.episode_count = EPISODE_COUNT
        self.training_episodes = []
        self.test_episodes = []
        self.epsilon = EPSILON

    def run_full_simulation(self):
        self.run_training_simulation()
        self.run_test_simulation()

    def run_training_simulation(self):
        count = self.episode_count
        counter = 0
        while count:
            if counter % 50 == 0:
                self.epsilon = max((self.epsilon - 0.01), 0)
            counter += 1

            episode = Episode(self.epsilon)
            if not self.robot:
                self.robot = Robot([0, 0], episode.grid)

            episode.run_episode(self.robot)
            self.training_episodes.append(episode)
            count -= 1
            if count % 100 == 0:
                print(episode.total_reward)
        print(f"training avg: {sum(ep.total_reward for ep in self.training_episodes)/EPISODE_COUNT}")

    def run_test_simulation(self):
        print("test results")
        self.epsilon = EPSILON
        count = self.episode_count
        while count:
            episode = Episode(self.epsilon)
            if not self.robot:
                xy = [random.randint(0, SIZE), random.randint(0, SIZE)]
                self.robot = Robot(xy, episode.grid)

            episode.run_episode(self.robot)
            self.test_episodes.append(episode)
            count -= 1
        sum = 0
        for ep in self.test_episodes:
            sum += ep.total_reward
        print(sum / EPISODE_COUNT)


class Episode:
    def __init__(self, epsilon) -> None:
        self.epsilon = epsilon
        self.grid = Grid(SIZE)
        self.step_count = STEP_COUNT
        self.total_reward = 0
        self.average_reward = 0

    def run_episode(self, robot: Robot):
        start_xy = [random.randint(0, SIZE - 1), random.randint(0, SIZE - 1)]
        robot.current_xy = start_xy
        robot.current_state = self.grid.calculate_state(robot.current_xy)
        count = self.step_count
        while count:
            # sleep(0.3)
            self.total_reward += robot.progress_state(self.grid, self.epsilon)
            count -= 1
        self.average_reward = self.total_reward / self.step_count


# grid = Grid(SIZE)
# rob = Robot([0,0], grid)
# print(grid.calculate_state(rob.current_xy))
# print(rob.progress_state(grid))

# episode = Episode()
# episode.run_episode(Robot([0, 0], episode.grid))

simulation = Simulation()
simulation.run_full_simulation()
