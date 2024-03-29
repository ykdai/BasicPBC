import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class NBodySimulation:
    def __init__(self, pos_start, fixed, alpha=0.01, gamma=1, weight=None):
        self.pos_start = pos_start
        self.pos = pos_start.copy()
        self.fixed = fixed.copy()
        self.alpha = alpha
        self.N = pos_start.shape[0]
        self.D = pos_start.shape[1]
        self.pos_history = None
        self.gamma = gamma
        if weight is None:
            self.weight = np.ones(self.fixed.shape)
        elif weight.shape == fixed.shape:
            self.weight = weight ** (-1)
        else:
            assert False, "Weight length not match!"

    def calculate_force(self):
        force_direction = np.zeros((self.N, self.D))
        dist_sum = 0
        for i in range(len(self.pos)):
            if self.fixed[i]:
                continue
            min_dist = float("inf")
            min_idx = -1
            for j in range(len(self.pos)):
                if i == j:
                    continue
                dist = np.linalg.norm(self.pos[i] - self.pos[j]) * self.weight[j]
                if dist < min_dist:
                    min_dist = dist * self.weight[j]
                    min_idx = j
            dist_sum += min_dist
            if min_idx != -1:
                direction = self.pos[i] - self.pos[min_idx]
                norm = np.linalg.norm(direction)
                if norm < 0.0001:
                    # avoid zero
                    vec = np.random.randn(self.D)
                    direction = vec / np.linalg.norm(vec)
                else:
                    direction /= np.linalg.norm(direction)
                force_direction[i] = direction * self.weight[j]
        # print(dist_sum)
        return force_direction

    def run_iter(self):
        thres_min = 0.1
        thres_max = 0.9
        direction = self.calculate_force()
        self.pos += self.alpha * direction
        self.alpha *= self.gamma
        self.pos[self.pos < thres_min] = thres_min
        self.pos[self.pos > thres_max] = thres_max
        fixed_3d = np.broadcast_to(self.fixed[:, None], (self.fixed.shape[0], 3))
        self.pos = fixed_3d * self.pos_start + (1 - fixed_3d) * self.pos

    def run(self, T):
        self.pos_history = np.zeros((T, self.N, self.D))
        self.pos_history[0] = self.pos.copy()
        gamma_start = self.gamma
        for i in range(1, T):
            self.run_iter()
            self.pos_history[i] = self.pos.copy()
        self.gamma = gamma_start
        self.pos_history = self.pos_history
        self.pos = self.pos_history[-1]

    def visualize_history(self):
        if self.D == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            for i in range(len(self.pos)):
                x, y, z = (
                    self.pos_history[:, i, 0],
                    self.pos_history[:, i, 1],
                    self.pos_history[:, i, 2],
                )
                ax.plot(x, y, z)
            ax.set_xlabel("X Label")
            ax.set_ylabel("Y Label")
            ax.set_zlabel("Z Label")
            plt.show()
        elif self.D == 2:
            # TODO
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i in range(len(self.pos)):
                x, y = self.pos_history[:, i, 0], self.pos_history[:, i, 1]
                ax.plot(x, y)
            ax.set_xlabel("X Label")
            ax.set_ylabel("Y Label")
            plt.show()
        else:
            assert False, "We only support visualizing dimension 2 or 3."

    def visualize_result(self):
        if self.D == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            for i in range(len(self.pos)):
                x, y, z = self.pos[i, 0], self.pos[i, 1], self.pos[i, 2]
                ax.scatter(x, y, z)
            ax.set_xlabel("X Label")
            ax.set_ylabel("Y Label")
            ax.set_zlabel("Z Label")
            plt.show()
        elif self.D == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for i in range(len(self.pos)):
                x, y = self.pos[i, 0], self.pos[i, 1]
                ax.scatter(x, y)
            ax.set_xlabel("X Label")
            ax.set_ylabel("Y Label")
            plt.show()
        else:
            assert False, "We only support visualizing dimension 2 or 3."


def redistribute_colors(all_color_np, fixed_np, random_color=False):
    # all_color_np are numpy colors in range [0,255] with shape [N,3].
    # fixed_np will be 1 if the color is locked with shape [N]
    # Finally, a numpy with new colors in [0,255] will be outputed.
    all_color_np = all_color_np / 255
    all_color_np = np.vstack((all_color_np, np.array([0, 0, 0])))
    fixed_np = np.append(fixed_np, 1)
    if random_color:
        fixed_3d = np.broadcast_to(fixed_np[:, None], (fixed_np.shape[0], 3))
        all_color_np = fixed_3d * all_color_np + (1 - fixed_3d) * np.random.rand(*(all_color_np.shape))
    alpha = 0.003  # 0.02
    sim = NBodySimulation(all_color_np, fixed_np, alpha, gamma=0.99)
    sim.run(40)  # 200
    output = np.round(255 * sim.pos)
    return output[:-1].astype(int)
