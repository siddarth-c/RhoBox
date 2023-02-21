"""
Name: boxies.py
Usage: Environment for the problem of finding the smallest square that can accomodate n-smaller 1x1 boxes

Authors: Anshuman Senapati, Siddarth Chandrasekar
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import gymnasium
from gymnasium import spaces
import sys
sys.modules["gym"] = gymnasium



INITIALIZATION = -1


# calculates distance between 2 points
def distance_btw_points(p1, p2):
    return np.linalg.norm(p1 - p2)

# given 2 points, returns the corresponding line equation
def get_line_eqn(two_points):
    x_coords, y_coords = zip(*two_points)
    A = np.vstack([x_coords,np.ones(len(x_coords))]).T
    m, c = np.linalg.lstsq(A, y_coords, rcond = None)[0]

    return m, c

# given the center and theta of a square, returns all 4 corner points
def get_corners_from_rectangle(center, angle):
   
    angle = np.radians(angle)
    v1 = np.array([np.cos(angle), np.sin(angle)]) / 2
    v2 = np.array([-v1[1], v1[0]])  # rotate by 90

    return [
        center + v1 + v2,
        center - v1 + v2,
        center - v1 - v2,
        center + v1 - v2,
    ]


# given all the squares, returns the smallest square that accomodates all the squares
def form_outer_square(centers, thetas):

    max_points = []
    for center, theta in zip(centers, thetas):
        points = get_corners_from_rectangle(center, theta)
        max_point = max(np.array(points).flatten())
        max_points.append(max_point)

    return max(max_points)


# Given 2 squares (Existing E, Current C), 2 closest points of C wrt E and closest + farthest points of E wrt C
def near2_and_near_far(center1, center2, theta1, theta2):

    four_corners = get_corners_from_rectangle(center = center1, angle = theta1)
    distance_btw_corner_center = [distance_btw_points(c, center2) for c in four_corners]

    # 2 closest points
    idx = np.argsort(distance_btw_corner_center)
    p1, p2 = four_corners[idx[0]], four_corners[idx[1]]

    # 1 closest point, 1 farthest point
    four_corners = get_corners_from_rectangle(center = center2, angle = theta2)
    distance_btw_corner_center = [distance_btw_points(c, center1) for c in four_corners]

    idx = np.argsort(distance_btw_corner_center)
    p3, p4 = four_corners[idx[0]], four_corners[idx[-1]]

    return [[p1, p2], [p3, p4]]


# Given past centers and thetas, find whether the current square overlaps with any
def does_it_overlap(past_centers, past_thetas, center, theta, permissible_s):

    overlapping = False

    points = get_corners_from_rectangle(center, theta)

    # Check whether the square lies within the permissible square
    for point in points:        
        if point[0] < 0 or point[1] > permissible_s:
            return True


    for p_cent, p_thet in zip(past_centers, past_thetas):

        if distance_btw_points(p_cent, center) < 1:
            print('Distance less than 1')
            return True
        
        elif distance_btw_points(p_cent, center) >= np.sqrt(2):
            pass

        else:

            points = near2_and_near_far(np.array(p_cent), np.array(center), p_thet, theta)

            if points[0][0][0] == points[0][1][0]:
                line_x = points[0][0][0]

                overlapping = (center[0] - line_x) * (points[1][0][0] - line_x) < 0

                if overlapping:
                    return True
                
            else:

                m, c = get_line_eqn(points[0])
                overlapping = (points[1][0][1] - m * points[1][0][0] - c) * (center[1] - m * center[0] - c) < 0

                if overlapping:
                    return True
    
    return overlapping


class boxesinbox(gymnasium.Env):
    
    def __init__(self, n = 3) -> None:

        self.n = n

        self.largest_square_permissible = np.ceil(np.sqrt(self.n))
        self.colors = ['red', 'blue', 'green', 'yellow', 'pink', 'cyan', 'olive', 'brown']

        # * Copy to reset()
        self.observation = np.ones((self.n, 3)) * INITIALIZATION
        self.observation[0] = np.array([0.5, 0.5, 0])
        self.time_step = 1
        self.terminated = False
        self.truncated = False
        self.info = {}

        self.max_time_step = 50

        self.observation_space = spaces.Box(low = -1, high = n, shape = (self.n, 3), dtype = np.float64)
        self.action_space = spaces.Box(low = -1, high = n, shape = (2,), dtype = np.float64)

    
    def update_observation(self, action):
        self.observation[self.time_step] = action


    def render(self):

        ax = plt.gca()

        for center, theta in zip(self.observation[:self.time_step, :2], self.observation[:self.time_step, 2]):

            four_points = get_corners_from_rectangle(center, theta)
            bottom_left = four_points[np.argmin(np.array(four_points)[:, 1])]

            ax.add_patch(Rectangle(bottom_left, 1, 1, theta, alpha = 0.5, color = np.random.choice(self.colors), lw = 4))
            plt.scatter(center[0], center[1], marker = '*', c = 'black')
        
        plt.xlim(left = 0, right = self.largest_square_permissible)
        plt.ylim(bottom = 0, top = self.largest_square_permissible)

        plt.grid()
        plt.gca().set_aspect("equal")

        plt.title(f'n = {self.n}')

        plt.show()



    def reset(self):

        self.observation = np.ones((self.n, 3)) * INITIALIZATION
        self.observation[0] = np.array([0.5, 0.5, 0])
        self.time_step = 1
        self.terminated = False
        self.truncated = False
        self.info = {}

        return self.observation, self.info
    

    def step(self, action):
        
        # print(self.observation[:, :2], '*', self.observation[:, 2], '*', action[:2], '*', action[-1])

        overlapping = does_it_overlap(self.observation[:, :2], self.observation[:, 2], action[:2], action[-1], self.largest_square_permissible)

        if not overlapping:
            self.observation[self.time_step, :2] = action[:2]
            self.observation[self.time_step, 2] = action[-1]
        
        else:
            self.terminated = False
            reward = -1
            self.time_step -= 1
            # self.done = True
            # reward = -1 * (self.largest_square_permissible**2)

        if not overlapping and self.time_step < self.n - 1:
            reward = 0

        elif not overlapping and self.time_step == self.n - 1:
            outer_square_s = form_outer_square(self.observation[:, :2], self.observation[:, 2])
            # reward = (outer_square_s**2) - (self.largest_square_permissible**2)
            reward = -outer_square_s
            self.terminated = True

        self.time_step += 1

        if self.time_step == self.max_time_step:
            self.truncated = True

        return self.observation, reward, self.terminated, self.truncated, self.info