import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import math
import sys

def next_state(state, h):

    temp_state = State([0,0],[0,0],[0,0],[0,0])

    k1 = diff_value(state)
    k2 = diff_value(state + (k1*(h/2)))
    k3 = diff_value(state + (k2*(h/2)))
    k4 = diff_value(state + (k3*h))

    delta = (k1 + k2*2 + k3*2 + k4)*(h/6)

    temp_state = state + delta

    x1 = state.length[0]*math.sin(temp_state.theta[0])
    y1 = -state.length[0]*math.cos(temp_state.theta[0])

    temp_state.pos_x = [x1, x1 + state.length[1]*math.sin(temp_state.theta[1])]
    temp_state.pos_y = [y1, y1 - state.length[1]*math.cos(temp_state.theta[1])]

    return temp_state


def diff_value(state):

    temp_state = State([0,0],[0,0],[0,0],[0,0])
    
    t1, t2 = state.theta[0], state.theta[1]
    o1, o2 = state.omega[0], state.omega[1]

    m1, m2 = state.mass[0], state.mass[1]
    l1, l2 = state.length[0], state.length[1]

    g = 9.807

    a = -g*(2*m1 + m2)*math.sin(t1) - m2*g*math.sin(t1 - 2*t2) - 2*math.sin(t1 - t2)*m2*((o2**2)*l2 + (o1**2)*l1*math.cos(t1 - t2))
    b = l1*(2*m1 + m2 - m2*math.cos(2*t1 - 2*t2))

    temp_state.omega[0] = a/b
    temp_state.theta[0] = state.omega[0]

    a = 2*math.sin(t1 - t2)*((o1**2)*l1*(m1 + m2) + g*(m1 + m2)*math.cos(t1) + (o2**2)*l2*m2*math.cos(t1 - t2))
    b = l2*(2*m1 + m2 - m2*math.cos(2*t1 - 2*t2))

    temp_state.omega[1] = a/b
    temp_state.theta[1] = state.omega[1]

    return temp_state


def update(num, dot1, dot2, line3, line4, trail, x, y):
    val = max(0, num - 1000)

    ax.set_title(num)

    dot1.set_data(x[num, 0], y[num, 0])
    dot2.set_data(x[num, 1], y[num, 1])

    line3.set_data([0, x[num, 0]], [0, y[num, 0]])
    line4.set_data([x[num, 0], x[num, 1]], [y[num, 0], y[num, 1]])

    trail.set_data(x[val:num, 1], y[val:num, 1])


class State():

    def __init__(
        self,
        pos_x = None,
        pos_y = None,
        theta = [(180/math.pi)*90, (180/math.pi)*-90],# 0 = South
        omega = [0, 0],
        mass = [1, 1.5],
        length = [5, 7]
    ):
        try:
            self.pos_x = [math.sin((180/math.pi)*90)*length[0], math.sin((180/math.pi)*90)*length[0] + length[1]*math.sin((180/math.pi)*-90)] if pos_x == None else pos_x
            self.pos_y = [-math.cos((180/math.pi)*90)*length[0], -math.cos((180/math.pi)*90)*length[0] - length[1]*math.cos((180/math.pi)*-90)] if pos_y == None else pos_y
            self.theta = theta
            self.omega = omega
            self.mass = mass
            self.length = length
        except:
            print("ERROR - state __init__")

    def __str__(self):

        return "({0},{1})".format(self.pos_x, self.pos_y)

    def __add__(self, other):

        t_x = [0,0]
        t_y = [0,0]
        t_t = [0,0]
        t_o = [0,0]

        for i in range(2):
            t_x[i] = self.pos_x[i] + other.pos_x[i]
            t_y[i] = self.pos_y[i] + other.pos_y[i]
            t_t[i] = self.theta[i] + other.theta[i]
            t_o[i] = self.omega[i] + other.omega[i]

        return State(t_x, t_y, t_t, t_o)

    def __mul__(self, other):

        t_x = [0,0]
        t_y = [0,0]
        t_t = [0,0]
        t_o = [0,0]

        for i in range(2):
            t_x[i] = self.pos_x[i] * other
            t_y[i] = self.pos_y[i] * other
            t_t[i] = self.theta[i] * other
            t_o[i] = self.omega[i] * other

        return State(t_x, t_y, t_t, t_o)


size = 1000
skip = 100
d = 2


x_matrix = np.zeros(shape = (size, 2, d))
y_matrix = np.zeros(shape = (size, 2, d))

states = [State(mass=[random.random(), random.random()]) for i in range(10)]

for i in range(skip*size):
    for j in range(len(states)):
        if i%skip == 0:
            for k in range(d):
                print(states[j].pos_x)
                print(x_matrix[i//skip, :, j])
                x_matrix[i//skip, :, j] = states[j].pos_x
                y_matrix[i//skip, :, j] = states[j].pos_y

        states[j] = next_state(states[j], 0.0001)


fig = plt.figure()
ax = fig.add_subplot()

x = x_matrix[:, :, 0]
y = y_matrix[:, :, 0]


dot = ax.plot(0, 0, 'o', color = "#1b5e20")

dot1, = ax.plot(x[:, 0], y[:, 0], 'o', color = "#b71c1c")
dot2, = ax.plot(x[:, 1], y[:, 1], 'o', color = "#311b92")

line3, = ax.plot([0, x[0, 1]], [0, y[0, 1]], '-')
line4, = ax.plot([x[0, 0], x[0, 1]], [y[0, 0], y[0, 1]], '-')

trail, = ax.plot([x[0, 1], y[0, 1]], '-')


ani = animation.FuncAnimation(fig, update, size, fargs=(dot1, dot2, line3, line4, trail, x, y), interval = 1)

plt.show()


# goodluck future self