import numpy as np

class RMSprop():
    def __init__(self, grad, guess):
        self.n = len(guess)
        self.G = grad
        self.decay = 0.01
        self.learn = 0.5
        self.epsilon = [0.000000001] * self.n

        self.result_x = [guess]
        self.result_E = [[0]*self.n]
        self.result_G = [self.G(guess)]

    def clear(self, guess):
        self.result_x = [guess]
        self.result_E = [[0]]
        self.result_G = []

    def E(self, prev_E, curr_x=None):
        if curr_x:
            g = self.G(curr_x)
            self.result_G.append(g)
        else:
            g = self.result_G[-1]
        e = self.decay * np.array(prev_E) + (1-self.decay) * np.array(g)**2
        return e.tolist()

    @property
    def update(self):
        return np.array(self.result_G[-1]) / np.sqrt(np.array(self.result_E[-1])+self.epsilon)

    def next_x(self, prev_x):
        x = np.array(prev_x) - self.learn * self.update
        return x.tolist()

    def start(self, max_n):
        self.result_E.append(self.E(self.result_E[-1]))  # will not append result_G
        N = 0
        while N < max_n:
            N += 1
            prev_x = self.result_x[-1]
            self.result_x.append(self.next_x(prev_x))
            self.result_E.append(self.E(self.result_E[-1], self.result_x[-1]))  # will append result_G
        # if N == max_n:
        #     print('REMARK -- maximum iteration step is achieved, convergence criterion may not be satisfied.')


# def F(x):
#     return math.log(x-10) ** 2
# def G(x):
#     return 2 * math.log(x-10) / (x-10)