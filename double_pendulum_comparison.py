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
    k1 = diff_value(state)
    k2 = diff_value(state + (k1*(h/2)))
    k3 = diff_value(state + (k2*(h/2)))
    k4 = diff_value(state + (k3*h))

    delta = (k1 + k2*2 + k3*2 + k4)*(h/6)

    state = state + delta

    return state

def diff_value(state):
    temp = np.zeros(len(state))

    t1, t2 = state[0], state[1]
    o1, o2 = state[2], state[3]

    m1, m2 = state[4], state[5]
    l1, l2 = state[6], state[7]

    g = 9.807

    a = -g*(2*m1 + m2)*math.sin(t1) - m2*g*math.sin(t1 - 2*t2) - 2*math.sin(t1 - t2)*m2*((o2**2)*l2 + (o1**2)*l1*math.cos(t1 - t2))
    b = l1*(2*m1 + m2 - m2*math.cos(2*t1 - 2*t2))

    temp[2] = a/b
    temp[0] = state[2]

    a = 2*math.sin(t1 - t2)*((o1**2)*l1*(m1 + m2) + g*(m1 + m2)*math.cos(t1) + (o2**2)*l2*m2*math.cos(t1 - t2))
    b = l2*(2*m1 + m2 - m2*math.cos(2*t1 - 2*t2))

    temp[3] = a/b
    temp[1] = state[3]

    return temp

def update(num, arguments, ax, x, y):
    val1 = max(0, num - 300)
    val2 = max(0, num - 1000)

    ax.set_title(num)

    for (dot1, dot2, line3, line4, trail6), i in zip(arguments, range(len(arguments))):
        dot1.set_data(x[num, 0, i], y[num, 0, i])
        dot2.set_data(x[num, 1, i], y[num, 1, i])

        line3.set_data([0, x[num, 0, i]], [0, y[num, 0, i]])
        line4.set_data([x[num, 0, i], x[num, 1, i]], [y[num, 0, i], y[num, 1, i]])

        # trail5.set_data(x[val1:num, 0, i], y[val1:num, 0, i])
        trail6.set_data(x[val2:num, 1, i], y[val2:num, 1, i])

size = 1000
skip = 100
dimension = 1

x_matrix = np.zeros(shape = (size*skip, 2, dimension))
y_matrix = np.zeros(shape = (size*skip, 2, dimension))

t1, t2 = 1, 1
o1, o2 = 0.7, 0
m1, m2 = 1, 1.5
l1, l2 = 5, 7

vector = np.zeros((dimension, 8))

for i in range(dimension):
    vector[i, :] = t1*math.pi, t2*math.pi, o1 + 0.5*random.random(), o2 + 0.5*random.random(), m1, m2, l1, l2

for i in range(size*skip):
    for j in range(dimension):
        if i%skip == 0:
            x1 = math.sin(vector[j, 0])*vector[j, 6]
            y1 = -math.cos(vector[j, 0])*vector[j, 6]

            x_matrix[i//skip, :, j] = [x1, x1 + vector[j, 7]*math.sin(vector[j, 1])]
            y_matrix[i//skip, :, j] = [y1, y1 - vector[j, 7]*math.cos(vector[j, 1])]

        vector[j, :] = next_state(vector[j, :], 0.0001)

fig = plt.figure()
ax = fig.add_subplot()

x = x_matrix
y = y_matrix

arguments = []

for i in range(dimension):
    line3, = ax.plot([0, x[0, 1, i]], [0, y[0, 1, i]], '-', color = "#000000")
    line4, = ax.plot([x[0, 0, i], x[0, 1, i]], [y[0, 0, i], y[0, 1, i]], '-', color = "#000000")

    # trail5, = ax.plot([x[0, 0, i], y[0, 0, i]], '-', color = "#1f77b4")
    trail6, = ax.plot([x[0, 1, i], y[0, 1, i]], '-.', color = "#ff7f0e")

    dot1, = ax.plot(x[:, 0, i], y[:, 0, i], 'o', color = "#1f77b4")
    dot2, = ax.plot(x[:, 1, i], y[:, 1, i], 'o', color = "#ff7f0e")

    arguments.append((dot1, dot2, line3, line4, trail6))

dot, = ax.plot(0, 0, 'o', color = "#2ca02c")

ani = animation.FuncAnimation(fig, update, size, fargs=(arguments, ax, x, y), interval = 1)

ani.save("animation4.mp4", fps=120, dpi=300)
# plt.show()