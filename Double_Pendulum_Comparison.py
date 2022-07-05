import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import math

# Graphical comparison of the differences in motion caused by subtle differences in starting parameters
#
# --------------------------------------------------------------------------
#
# next_state: returns a updated state using Runge Kutta
# diff_state: returns the difference in state objects, using differential equations
# update: produces a plot of several pendulums at a time step
#
# --------------------------------------------------------------------------
#
# Change Variation:
# line 157
# [State(pos_x,pos_y,[t1,t2],[o1,o2],[m1,m2],[l1 + random.random(),l2 + random.random()]) for i in range(dimension)]
# Move the addition of the random values (between 0 -> 1), to other parameters

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


def update(num, arguments, ax, x, y):
    val = max(0, num - 1000)

    ax.set_title(num)

    for (dot1, dot2, line3, line4, trail), i in zip(arguments, range(len(arguments))):
        dot1.set_data(x[num, 0, i], y[num, 0, i])
        dot2.set_data(x[num, 1, i], y[num, 1, i])

        line3.set_data([0, x[num, 0, i]], [0, y[num, 0, i]])
        line4.set_data([x[num, 0, i], x[num, 1, i]], [y[num, 0, i], y[num, 1, i]])

        trail.set_data(x[val:num, 1, i], y[val:num, 1, i])


class State():

    def __init__(
        self,
        pos_x = None,
        pos_y = None,
        theta = None,# 0 = South
        omega = None,
        mass = None,
        length = None
    ):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.theta = theta
        self.omega = omega
        self.mass = mass
        self.length = length

    def __str__(self):

        return "({0},{1})".format(self.pos_x, self.pos_y)

    def __add__(self, other):

        t_x = [0,0]
        t_y = [0,0]
        t_t = [0,0]
        t_o = [0,0]
        t_m = self.mass
        t_l = self.length

        for i in range(2):
            t_x[i] = self.pos_x[i] + other.pos_x[i]
            t_y[i] = self.pos_y[i] + other.pos_y[i]
            t_t[i] = self.theta[i] + other.theta[i]
            t_o[i] = self.omega[i] + other.omega[i]

        return State(t_x, t_y, t_t, t_o, t_m, t_l)

    def __mul__(self, other):

        t_x = [0,0]
        t_y = [0,0]
        t_t = [0,0]
        t_o = [0,0]
        t_m = self.mass
        t_l = self.length

        for i in range(2):
            t_x[i] = self.pos_x[i] * other
            t_y[i] = self.pos_y[i] * other
            t_t[i] = self.theta[i] * other
            t_o[i] = self.omega[i] * other

        return State(t_x, t_y, t_t, t_o, t_m, t_l)


size = 1000
skip = 100
dimension = 50

x_matrix = np.zeros(shape = (size*skip, 2, dimension))
y_matrix = np.zeros(shape = (size*skip, 2, dimension))

t1, t2 = 0.5*math.pi, 0.5*math.pi
o1, o2 = 0, 0
m1, m2 = 1, 1.5
l1, l2 = 5, 7

x1 = math.sin(t1)*l1
y1 = -math.cos(t1)*l2

pos_x = [x1, x1 + l2*math.sin(t2)]
pos_y = [y1, y1 - l2*math.cos(t2)]

states = [State(pos_x,pos_y,[t1,t2],[o1,o2],[m1,m2],[l1 + random.random(),l2 + random.random()]) for i in range(dimension)]

for i in range(size*skip):
    for j in range(dimension):
        states[j] = next_state(states[j], 0.0001)
        if i%skip == 0:
            x_matrix[i//skip, :, j] = states[j].pos_x
            y_matrix[i//skip, :, j] = states[j].pos_y


fig = plt.figure()
ax = fig.add_subplot()

x = x_matrix
y = y_matrix

arguments = []

dot = ax.plot(0, 0, 'o', color = "#1b5e20")

for i in range(dimension):
    dot1, = ax.plot(x[:, 0, i], y[:, 0, i], 'o', color = "#b71c1c")
    dot2, = ax.plot(x[:, 1, i], y[:, 1, i], 'o', color = "#311b92")

    line3, = ax.plot([0, x[0, 1, i]], [0, y[0, 1, i]], '-')
    line4, = ax.plot([x[0, 0, i], x[0, 1, i]], [y[0, 0, i], y[0, 1, i]], '-')

    trail, = ax.plot([x[0, 1, i], y[0, 1, i]], '-')

    arguments.append((dot1, dot2, line3, line4, trail))


ani = animation.FuncAnimation(fig, update, size, fargs=(arguments, ax, x, y), interval = 1)

plt.show()