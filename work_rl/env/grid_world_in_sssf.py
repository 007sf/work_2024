__credits__ = ["Intelligent Unmanned Systems Laboratory at Westlake University."]

import copy
import os
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from work_rl.env.arguments import args
import json
from scipy.signal import find_peaks
class GridWorld():

    def __init__(self, env_size=args.env_size,
                 start_state=args.start_state,
                 target_state=args.target_state,

                 forbidden_states=args.forbidden_states):
        action_map = {0: (0, 1),
                      1: (1, 0),
                      2: (0, -1),
                      3: (-1, 0),
                      4: (0, 0),
                      5:"mark"
                      }
        self.action_map = action_map
        self.env_size = env_size
        self.num_states = env_size[0] * env_size[1]

        # start_state = json.loads(start_state)
        # target_state = json.loads(target_state)

        self.start_state = start_state
        self.target_state = target_state
        forbidden_states = json.loads(forbidden_states)
        self.forbidden_states = forbidden_states

        self.agent_state = start_state
        self.action_space = args.action_space  # 动作空间变为0,1,2,3,4,5
        self.reward_target = args.reward_target
        self.reward_forbidden = args.reward_forbidden
        self.reward_step = args.reward_step

        self.reward_mark = args.reward_mark
        self.reward_invalid_mark = args.reward_invalid_mark

        self.canvas = None
        self.animation_interval = args.animation_interval

        self.color_forbid = (0.9290, 0.6940, 0.125)
        self.color_target = (0.3010, 0.7450, 0.9330)
        self.color_policy = (0.4660, 0.6740, 0.1880)
        self.color_trajectory = (0, 1, 0)
        self.color_agent = (0, 0, 1)

    def reset(self):
        self.agent_state = copy.deepcopy(self.start_state)
        self.traj = [self.agent_state["position"]]
        self.agent_state["marks_pos"] = []
        # self.scatter = [self.agent_state["marks_pos"]]
        self.scatter = copy.deepcopy(self.start_state["marks_pos"])
        return self.agent_state

    def step(self, action):

        assert action in self.action_space, "Invalid action"

        next_state, reward = self._get_next_state_and_reward(self.agent_state, action)
        next_state = copy.deepcopy(next_state)
        if len(next_state["marks_pos"])>10:
            print(self.agent_state)
        done = self._is_done(next_state)


        x_store = next_state["position"][0] + 0.03 * np.random.randn()
        y_store = next_state["position"][1] + 0.03 * np.random.randn()
        if action!=5:

            action=self.action_map[action]
            state_store = tuple(np.array((x_store, y_store)) + 0.2 * np.array(action))  #--------------------------------------------------------
            state_store_2 = (next_state["position"][0], next_state["position"][1])
            self.agent_state = next_state

            self.traj.append(state_store)
            self.traj.append(state_store_2)
        else:
            state_store =  tuple(np.array((x_store, y_store)))
            state_store_2 = (next_state["position"][0], next_state["position"][1])

            self.traj.append(state_store)
            self.traj.append(state_store_2)
            # if state_store not in self.scatter:
            if next_state["position"] not in self.scatter and self.agent_state["marks_left"]>0:
                self.scatter.append(next_state["position"])
            self.agent_state = next_state
        return self.agent_state, reward, done, {}

    def __reward_mark_function(self,state):
        used_sensor  = self.start_state["marks_left"]-state["marks_left"]
        sensor_pos = state["marks_pos"]

        grid_data = np.zeros((self.env_size[0],self.env_size[1]))
        for sensor in sensor_pos:
            sensor_index = sensor[0]+sensor[1]*self.env_size[0]
            grid_data.flat[sensor_index]=1

        fft_data = np.fft.fft2(grid_data)
        fft_shifted = np.fft.fftshift(fft_data)
        magnitude = np.abs(fft_shifted)
        magnitude = magnitude.flatten()
        peaks, _ = find_peaks(magnitude)
        peak_values = np.array(magnitude[peaks])
        peak_values = np.unique(peak_values)[::-1]
        if len(peak_values)>1:
            r_loc =(peak_values[0]-peak_values[1])/peak_values[0]
        else:
            r_loc = -0.1


        r_num = used_sensor/self.start_state["marks_left"]
        r_time=-30/600
        sum = r_loc+r_num+r_time
        # return 20*(r_loc/sum)+0.05*(r_num/sum)+0.05*(r_time/sum)
        return 2

    def _get_next_state_and_reward(self, state, action):
        x, y = state["position"]
        marks_left = state["marks_left"]
        marks_pos = state["marks_pos"]

        new_state = {
            "position": (x, y),
            "marks_left": marks_left,
            "marks_pos": marks_pos[:],
        }
        self.reward_mark = self.__reward_mark_function(state)
        self.forbidden_states["position"] = [tuple(pos) for pos in self.forbidden_states["position"]]

        if action == 5:

            # if marks_left > 0 and (x, y) not in marks_pos:
            if marks_left>0:
                if (x,y) not in marks_pos :
                    new_state["marks_left"] = marks_left- 1
                    new_state["marks_pos"].append((x, y))
                    self.reward_mark = self.__reward_mark_function(new_state)
                    reward = self.reward_mark
                    # print("此时状态为{}，标记奖励值为：{}".format(new_state,reward))

                else:
                    reward = self.reward_forbidden
            else:
                reward = self.reward_forbidden
        else:
            action = self.action_map[action]

            new_state["position"] = tuple(np.array(state["position"]) + np.array(action))
            x, y = state["position"]
            if y + 1 > self.env_size[1] - 1 and action == (0, 1):  # down
                y = self.env_size[1] - 1
                reward = self.reward_forbidden
            elif x + 1 > self.env_size[0] - 1 and action == (1, 0):  # right
                x = self.env_size[0] - 1
                reward = self.reward_forbidden
            elif y - 1 < 0 and action == (0, -1):  # up
                y = 0
                reward = self.reward_forbidden
            elif x - 1 < 0 and action == (-1, 0):  # left
                x = 0
                reward = self.reward_forbidden
            elif new_state["position"] == self.target_state["position"]:  # stay
                x,y = new_state["position"]
                if new_state["marks_left"] == 0:

                    reward = self.reward_target
                else:
                    reward = -self.reward_target*3

            elif new_state["position"] in self.forbidden_states["position"]:  # stay
                x,y  = map(int,state["position"])
                # new_state["position"] = (x,y)


                reward = self.reward_forbidden
            else:
                x,y = new_state["position"]
                reward = self.reward_step
            new_state["position"] = (x, y)
        return new_state, reward

    def compute_expected_reward(self, policy_matrix):
        r_pi = np.zeros(self.num_states)

        for s in range(self.num_states):
            x = s % self.env_size[0]
            y = s // self.env_size[0]
            for index, probability in enumerate(policy_matrix[s]):
                action = self.action_space[index]
                (x1, y1), immediately_reward = self._get_next_state_and_reward((x, y), action)
                r_pi[s] += probability * immediately_reward
        return r_pi

    def compute_transition_probability(self, policy_matrix):
        p_pi = np.zeros((self.num_states, self.num_states))

        for s in range(self.num_states):
            x = s % self.env_size[0]
            y = s // self.env_size[0]
            for index, probability in enumerate(policy_matrix[s]):
                action = self.action_space[index]
                (x1, y1), immediately_reward = self._get_next_state_and_reward((x, y), action)
                next_s = x1 + y1 * self.env_size[0]
                p_pi[s][next_s] += probability
        return p_pi

    def _is_done(self, state):
        return state["position"] == self.target_state["position"] and state["marks_left"]==0


    def render(self, animation_interval=args.animation_interval):
        if self.canvas is None:
            plt.ion()
            self.canvas, self.ax = plt.subplots()
            self.ax.set_xlim(-0.5, self.env_size[0] - 0.5)
            self.ax.set_ylim(-0.5, self.env_size[1] - 0.5)
            self.ax.xaxis.set_ticks(np.arange(-0.5, self.env_size[0], 1))
            self.ax.yaxis.set_ticks(np.arange(-0.5, self.env_size[1], 1))
            self.ax.grid(True, linestyle="-", color="gray", linewidth="1", axis='both')
            self.ax.set_aspect('equal')
            self.ax.invert_yaxis()
            self.ax.xaxis.set_ticks_position('top')

            idx_labels_x = [i for i in range(self.env_size[0])]
            idx_labels_y = [i for i in range(self.env_size[1])]
            for lb in idx_labels_x:
                self.ax.text(lb, -0.75, str(lb + 1), size=10, ha='center', va='center', color='black')
            for lb in idx_labels_y:
                self.ax.text(-0.75, lb, str(lb + 1), size=10, ha='center', va='center', color='black')
            self.ax.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False,
                                labeltop=False)

            self.target_rect = patches.Rectangle((self.target_state["position"][0] - 0.5, self.target_state["position"][1] - 0.5), 1, 1,
                                                 linewidth=1, edgecolor=self.color_target, facecolor=self.color_target)
            self.ax.add_patch(self.target_rect)

            for forbidden_state in self.forbidden_states["position"]:
                rect = patches.Rectangle((forbidden_state[0] - 0.5, forbidden_state[1] - 0.5), 1, 1, linewidth=1,
                                         edgecolor=self.color_forbid, facecolor=self.color_forbid)
                self.ax.add_patch(rect)

            self.agent_star, = self.ax.plot([], [], marker='*', color=self.color_agent, markersize=20, linewidth=0.5)
            self.traj_obj, = self.ax.plot([], [], color=self.color_trajectory, linewidth=0.5)
            self.scatter_obj = self.ax.scatter([], [], color="red", label="mark", marker="x")
        # self.agent_circle.center = (self.agent_state[0], self.agent_state[1])
        self.agent_star.set_data([self.agent_state["position"][0]], [self.agent_state["position"][1]])
        traj_x, traj_y = zip(*self.traj)

        #我要跟随轨迹画出标记的点，下面是修改的代码

        if self.scatter and all(len(pos) == 2 for pos in self.scatter)  :
            marked_x, marked_y = zip(*self.scatter)
            self.scatter_obj.set_offsets(np.c_[marked_x, marked_y])
        self.traj_obj.set_data(traj_x, traj_y)

        plt.draw()
        '''保存图片'''
        # TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
        # # 保存路径
        # save_dir = "E:\\work\\work_rl\\picture_route"
        # save_path = os.path.join(save_dir, "pic_{}".format(TIMESTAMP))
        #
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # plt.savefig(save_path)
        plt.pause(animation_interval)

        if args.debug:
            input('press Enter to continue...')

    def add_policy(self, policy_matrix):
        for state, state_action_group in enumerate(policy_matrix):
            x = state % self.env_size[0]
            y = state // self.env_size[0]
            for i, action_probability in enumerate(state_action_group):
                if action_probability != 0:
                    dx_y = self.action_space[i]
                    dx,dy = self.action_map[dx_y]
                    if (dx, dy) != (0, 0):
                        self.ax.add_patch(patches.FancyArrow(x, y, dx=(0.1 + action_probability / 2) * dx,
                                                             dy=(0.1 + action_probability / 2) * dy,
                                                             color=self.color_policy, width=0.001, head_width=0.05))
                    else:
                        self.ax.add_patch(patches.Circle((x, y), radius=0.07, facecolor=self.color_policy,
                                                         edgecolor=self.color_policy, linewidth=1, fill=False))

    def add_state_values(self, values, precision=1):
        '''
            values: iterable
        '''
        values = np.round(values, precision)
        for i, value in enumerate(values):
            x = i % self.env_size[0]
            y = i // self.env_size[0]
            self.ax.text(x, y, str(value), ha='center', va='center', fontsize=10, color='black')

    def reset_canvas(self):
        """重置画布：清除轨迹数据并重置图像"""
        self.traj.clear()  # 清空轨迹数据
        self.traj_obj.set_data([], [])  # 清空轨迹图像
        plt.draw()

    def get_next_state_reward(self, state, action):
        next_state, reward = self._get_next_state_and_reward(state, action)
        return next_state[1] * self.env_size[0] + next_state[0], reward

    def get_state_from_index(self, index):
        x = index % self.env_size[0]
        y = index // self.env_size[0]
        return x, y