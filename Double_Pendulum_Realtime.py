import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import math


def next_state(state, h):

    temp_state = State([0,0],[0,0],[0,0],[0,0])

    k1 = diff_value(state)
    k2 = diff_value(state + (k1*(h/2)))
    k3 = diff_value(state + (k2*(h/2)))
    k4 = diff_value(state + (k3*h))

    delta = (k1 + k2*2 + k3*2 + k4)*(h/6)

    temp_state = state + delta

    x1 = 5*math.sin(temp_state.theta[0])
    y1 = -5*math.cos(temp_state.theta[0])

    temp_state.pos_x = [x1, x1 + 7*math.sin(temp_state.theta[1])]
    temp_state.pos_y = [y1, y1 - 7*math.cos(temp_state.theta[1])]

    return temp_state


def diff_value(state):

    temp_state = State([0,0],[0,0],[0,0],[0,0])
    
    t1, t2 = state.theta[0], state.theta[1]
    o1, o2 = state.omega[0], state.omega[1]

    m1, m2 = 1, 1.5
    l1, l2 = 5, 7

    g = 9.807

    a = -g*(2*m1 + m2)*math.sin(t1) - m2*g*math.sin(t1 - 2*t2) - 2*math.sin(t1 - t2)*m2*((o2**2)*l2 + (o1**2)*l1*math.cos(t1 - t2))
    b = l1*(2*m1 + m2 - m2*math.cos(2*t1 - 2*t2))

    temp_state.omega[0] = (a/b)
    temp_state.theta[0] = state.omega[0]

    a = 2*math.sin(t1 - t2)*((o1**2)*l1*(m1 + m2) + g*(m1 + m2)*math.cos(t1) + (o2**2)*l2*m2*math.cos(t1 - t2))
    b = l2*(2*m1 + m2 - m2*math.cos(2*t1 - 2*t2))

    temp_state.omega[1] = (a/b)
    temp_state.theta[1] = state.omega[1]

    return temp_state


def update(num, state, dot1, dot2, line3, line4):
    temp_state = state

    x, y = state.pos_x, state.pos_y

    for i in range(100):
        temp_state = next_state(temp_state, 0.0001)

    state.pos_x = temp_state.pos_x
    state.pos_y = temp_state.pos_y
    state.theta = temp_state.theta
    state.omega = temp_state.omega

    ax.set_title(num)

    dot1.set_data(x[0], y[0])
    dot2.set_data(x[1], y[1])

    line3.set_data([0, x[0]], [0, y[0]])
    line4.set_data([x[0], x[1]], [y[0], y[1]])


class State():

    def __init__(
        self,
        pos_x = [math.sin((180/math.pi)*90)*5, math.sin((180/math.pi)*90)*5 + 7*math.sin((180/math.pi)*-90)],
        pos_y = [-math.cos((180/math.pi)*90)*5, -math.cos((180/math.pi)*90)*5 - 7*math.cos((180/math.pi)*-90)],
        theta = [(180/math.pi)*90, (180/math.pi)*-90],# 0 = South
        omega = [0, 0]
    ):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.theta = theta
        self.omega = omega

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
    


size = 10000
skip = 100



x_matrix = np.zeros(shape = (size, 2))
y_matrix = np.zeros(shape = (size, 2))

state = State()

fig = plt.figure()
ax = fig.add_subplot()

# for i in range(skip*size):
#     if i%skip == 0:
#         x_matrix[i//skip, :], y_matrix[] = state.pos_x, state.pos_y

plt.xlim(-12, 12)
plt.ylim(-12, 12)

dot = ax.plot(0, 0, 'o', color = "#1b5e20")

dot1, = ax.plot(0, 0, 'o', color = "#b71c1c")
dot2, = ax.plot(0, 0, 'o', color = "#311b92")

line3, = ax.plot([0, 0], [0, 0], '-')
line4, = ax.plot([0, 0], [0, 0], '-')



ani = animation.FuncAnimation(fig, update, size, fargs=(state, dot1, dot2, line3, line4), interval = 1)

plt.show()











# goodluck future self