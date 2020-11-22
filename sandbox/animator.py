from matplotlib import pyplot as plt
import numpy as np
from utils import side_tools
import matplotlib.animation as animation


class WalkerAnimator(object):
    def __init__(self, model, walker_type, mass_center_size, mass_center_color,
                 link_width, link_color):
        self.model = model
        self.walker_type = walker_type
        self.mass_center_size = mass_center_size
        self.mass_center_color = mass_center_color
        self.link_width = link_width
        self.link_color = link_color


    def animate(self, x, stanceleg_coord, step_interval_sample_count, tf, save, display):
        # TODO: Finish the animation code
        ani, fig= self.generate_walker(x, stanceleg_coord, step_interval_sample_count, tf)
        if display is True:
            plt.show()
        if save is True:
            ani.save("3_link_Planar_Walker_Demo.gif")
            print('The demo video is successfully saved!')
            fig.savefig('demo')
        return None

    def generate_walker(self, x, stanceleg_coord, step_interval_sample_count, tf):
        n_sample = len(x)
        interval = (tf / n_sample) * 1000
        if self.walker_type == "3link":
            r = self.model.r  # leg length
            l = self.model.l  # hip to torso length
            th1 = (np.asarray([xi[0] for xi in x])).reshape((n_sample,))  # stance leg
            th2 = (np.asarray([xi[1] for xi in x])).reshape((n_sample,))  # swing leg
            th3 = (np.asarray([xi[2] for xi in x])).reshape((n_sample,))  # torso
            hip_x, hip_y, torso_x, torso_y, swing_x, swing_y, stance_x, stance_y = np.zeros((n_sample, )), \
                                                                                   np.zeros((n_sample, )), \
                                                                                   np.zeros((n_sample, )), \
                                                                                   np.zeros((n_sample, )), \
                                                                                   np.zeros((n_sample, )), \
                                                                                   np.zeros((n_sample, )), \
                                                                                   np.zeros((n_sample, )), \
                                                                                   np.zeros((n_sample, ))
            interval_start_index = 0
            # Compute the end coordinate of each link throughout the entire trajectory
            for i in range(len(step_interval_sample_count)):
                curr_stanceleg_coord = stanceleg_coord[i]
                interval_end_index = step_interval_sample_count[i]
                num_samples_per_interval = interval_end_index - interval_start_index
                curr_interval_th1 = th1[interval_start_index: interval_end_index]
                curr_interval_th2 = th2[interval_start_index: interval_end_index]
                curr_interval_th3 = th3[interval_start_index: interval_end_index]
                # compute cartesian coordinate of hip joint
                hip_x[interval_start_index: interval_end_index] = curr_stanceleg_coord[0] + \
                                                                  r * np.sin(curr_interval_th1)
                hip_y[interval_start_index: interval_end_index] = curr_stanceleg_coord[1] + \
                                                                  r * np.cos(curr_interval_th1)
                # compute cartesian coordinate of torso end
                torso_x[interval_start_index: interval_end_index] = hip_x[interval_start_index: interval_end_index] + \
                                                                    l * np.sin(curr_interval_th3)
                torso_y[interval_start_index: interval_end_index] = hip_y[interval_start_index: interval_end_index] + \
                                                                    l * np.cos(curr_interval_th3)
                # compute cartesian coordinate of swing leg end
                swing_xy = side_tools.get_swingleg_end_coord((curr_interval_th1, curr_interval_th2), curr_stanceleg_coord, r)
                swing_x[interval_start_index: interval_end_index] = swing_xy[0].reshape((num_samples_per_interval,))
                swing_y[interval_start_index: interval_end_index] = swing_xy[1].reshape((num_samples_per_interval,))
                # compute cartesian coordinate of stance leg end
                stance_x[interval_start_index: interval_end_index] = np.ones((num_samples_per_interval,)) * curr_stanceleg_coord[0]
                stance_y[interval_start_index: interval_end_index] = np.ones((num_samples_per_interval,)) * curr_stanceleg_coord[1]

                # update the start index
                interval_start_index = interval_end_index

            # define a frame update function
            fig1 = plt.figure(figsize=(10, 4))
            ax = plt.axes()
            hip_traj_x, hip_traj_y, stance_traj_x, stance_traj_y = [], [], [], []
            swing_traj_x, swing_traj_y, torso_traj_x, torso_traj_y = [], [], [], []
            def update(frame):
                ax.clear()
                # add link between stance leg and hip
                stance_coord = (stance_x[frame], stance_y[frame])
                hip_coord = (hip_x[frame], hip_y[frame])
                stance_to_hip_link = self.add_link(stance_coord, hip_coord)
                # add link between hip and torso
                torso_coord = (torso_x[frame], torso_y[frame])
                hip_to_torso_link = self.add_link(hip_coord, torso_coord)
                # add link between swing and hip
                swing_coord = (swing_x[frame], swing_y[frame])
                swing_to_hip_link = self.add_link(swing_coord, hip_coord)
                # add plot
                ax.add_line(stance_to_hip_link)
                ax.add_line(hip_to_torso_link)
                ax.add_line(swing_to_hip_link)
                ax.add_line(plt.Line2D((-100, 100), (0, 0), lw=1.0, color='k'))
                ax.add_patch(self.add_mass_center(hip_coord))  # add center of mass for the hip
                # add trajectory
                hip_traj_x.append(hip_x[frame]), hip_traj_y.append(hip_y[frame])
                stance_traj_x.append(stance_x[frame]), stance_traj_y.append(stance_y[frame])
                swing_traj_x.append(swing_x[frame]), swing_traj_y.append(swing_y[frame])
                torso_traj_x.append(torso_x[frame]), torso_traj_y.append(torso_y[frame])
                ax.scatter(hip_traj_x, hip_traj_y, s=0.2, c='r', marker='o')
                ax.scatter(stance_traj_x, stance_traj_y, s=0.2, c='r', marker='o')
                ax.scatter(swing_traj_x, swing_traj_y, s=0.2, c='r', marker='o')
                ax.scatter(torso_traj_x, torso_traj_y, s=0.2, c='r', marker='o')
                # set up some plot attributes.
                ax.set_xlim([-1, 7])
                ax.set_ylim([-0.5, 2])
                ax.set_xlabel('x-axis (m)')
                ax.set_ylabel('y-axis (m)')
                ax.set_aspect('equal')
                return plt

            ani = animation.FuncAnimation(fig1, update, frames=[i for i in range(n_sample)], blit=False, repeat=False,
                                          interval=interval)

            return ani, fig1

        if self.walker_type == "5link":
            # TODO: add 5 link
            pass

        return None


    def add_link(self, pt1, pt2):
        link = plt.Line2D((pt1[0], pt2[0]), (pt1[1], pt2[1]), lw=self.link_width,
                          color=self.link_color)
        return link

    def add_mass_center(self, pt):
        circle = plt.Circle((pt[0], pt[1]), radius=self.mass_center_size, fc=self.mass_center_color)
        return circle
