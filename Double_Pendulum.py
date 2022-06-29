import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import math


def next_state(state, h):

    print("STATE_____________________")
    print(state)
    temp_state = State([0,0],[0,0],[0,0],[0,0])

    k1 = diff_value(state)
    k2 = diff_value(state + (k1*(h/2)))
    k3 = diff_value(state + (k2*(h/2)))
    k4 = diff_value(state + (k3*h))

    delta = (k1 + k2*2 + k3*2 + k4)*(h/6)

    print("delta_____________________")
    print(delta)
    print(f"{delta.pos_x = }")
    print(f"{delta.pos_y = }")

    temp_state = state + delta

    x1 = 5*math.sin(temp_state.theta[0])### 5 = l1
    y1 = -5*math.cos(temp_state.theta[0])

    temp_state.pos_x = [x1, x1 + 7*math.sin(temp_state.theta[1])]#### 7 = l2
    temp_state.pos_y = [y1, y1 - 7*math.cos(temp_state.theta[1])]

    print("STATE2_____________________")
    print(state)

    return temp_state


def diff_value(state):
    
    print("STATEB_____________________")
    print(state)

    temp_state = State([0,0],[0,0],[0,0],[0,0])
    
    theta1, theta2 = state.theta[0], state.theta[1]
    w1, w2 = state.w[0], state.w[1]

    m1, m2 = 1, 1.5
    l1, l2 = 5, 7

    g = 9.807

    a = -g*(2*m1 + m2)*math.sin(theta1) - m2*g*math.sin(theta1 - 2*theta2) - 2*math.sin(theta1 - theta2)*m2*((w2**2)*l2 + (w1**2)*l1*math.cos(theta1 - theta2))
    b = l1*(2*m1 + m2 - m2*math.cos(2*theta1 - 2*theta2))

    print(f"1{a = }")
    print(f"1{b = }")

    temp_state.w[0] = (a/b)
    temp_state.theta[0] = state.w[0]

    a = 2*math.sin(theta1 - theta2)*((w1**2)*l1*(m1 + m2) + g*(m1 + m2)*math.cos(theta1) + (w2**2)*l2*m2*math.cos(theta1 - theta2))
    b = l2*(2*m1 + m2 - m2*math.cos(2*theta1 - 2*theta2))

    print(f"{a = }")
    print(f"{b = }")

    temp_state.w[1] = (a/b)
    temp_state.theta[1] = state.w[1]

    # x1 = state.w[0]*l1*math.cos(theta1)
    # y1 = state.w[0]*l1*math.sin(theta1)

    # temp_state.pos_x = [x1, x1 + state.w[1]*l2*math.cos(theta2)]
    # temp_state.pos_y = [y1, y1 + state.w[1]*l2*math.sin(theta2)]
  

    print("STATE2B_____________________")
    print(state)


    return temp_state


def update(num, line1, line2, line3, line4, x, y):

    ax.set_title(num)

    line1.set_data(x[num, 0], y[num, 0])
    line2.set_data(x[num, 1], y[num, 1])

    line3.set_data([0, x[num, 0]], [0, y[num, 0]])
    line4.set_data([x[num, 0], x[num, 1]], [y[num, 0], y[num, 1]])

    return line1, line2


class State():
    def __init__(
        self,
        pos_x = [math.sin((180/math.pi)*25)*5, math.sin((180/math.pi)*25)*5 + 7*math.sin((180/math.pi)*10)],
        pos_y = [-math.cos((180/math.pi)*25)*5, -math.cos((180/math.pi)*25)*5 - 7*math.cos((180/math.pi)*10)],
        theta = [(180/math.pi)*25, (180/math.pi)*-10],# 0 = South
        w = [0, 0]
    ):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.theta = theta
        self.w = w

    def __str__(self):
        return "({0},{1})".format(self.pos_x, self.pos_y)

    def __add__(self, other):
        t_x = [0,0]
        t_y = [0,0]
        t_t = [0,0]
        t_w = [0,0]
        for i in range(2):
            t_x[i] = self.pos_x[i] + other.pos_x[i]
            t_y[i] = self.pos_y[i] + other.pos_y[i]
            t_t[i] = self.theta[i] + other.theta[i]
            t_w[i] = self.w[i] + other.w[i]

        return State(t_x, t_y, t_t, t_w)

    def __mul__(self, other):
        t_x = [0,0]
        t_y = [0,0]
        t_t = [0,0]
        t_w = [0,0]
        for i in range(2):
            t_x[i] = self.pos_x[i] * other
            t_y[i] = self.pos_y[i] * other
            t_t[i] = self.theta[i] * other
            t_w[i] = self.w[i] * other

        return State(t_x, t_y, t_t, t_w)
    
size = 10000
skip = 10

x_array = np.zeros(shape = (size, 2))
y_array = np.zeros(shape = (size, 2))

state = State()

for i in range(skip*size):
    if i%skip == 0:
        x_array[i//skip, :], y_array[i//skip, :] = state.pos_x, state.pos_y


    # state = state + (diff_value(state)*0.01)
    # print("ADDED_STATE_____________________")
    # print(state)

    state = next_state(state, 0.0001)


fig = plt.figure()
ax = fig.add_subplot()

x = x_array
y = y_array

print(f'{y = }')
print(f'{x = }')

dot = ax.plot(0, 0, 'o', color = "#1b5e20")# Lines?

line1, = ax.plot(x[:, 0], y[:, 0], 'o', color = "#b71c1c")
line2, = ax.plot(x[:, 1], y[:, 1], 'o', color = "#311b92")

line3, = ax.plot([0, x[0, 1]], [0, y[0, 1]], '-')
line4, = ax.plot([x[0, 0], x[0, 1]], [y[0, 0], y[0, 1]], '-')


ani = animation.FuncAnimation(fig, update, size, fargs=(line1, line2, line3, line4, x, y), interval = 1)

plt.show()