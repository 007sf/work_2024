import copy
import math
import random

from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from collections import deque, Counter
import os
from torch.utils.tensorboard import SummaryWriter
from work_rl.env.grid_world_in_sssf import GridWorld
import torch.nn.functional as F
import seaborn as sns


class MyNet(nn.Module):
    def __init__(self, device, output_dim=6, n_max_points=10, ):
        """
        初始化网络
        :param output_dim: 动作空间的维度
        :param n_max_points: 状态中最大标记点数量
        :param device: 设备 (CPU 或 GPU)
        """
        super().__init__()
        # 定义全连接层
        # self.fc1 = nn.Linear(3+2*n_max_points, 128)#修改一下
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

        # 保存最大点数（n_max_points）
        self.n_max_points = n_max_points

        self.device = device

    def preprocess_state(self, states):
        """
        预处理状态，将字典形式的状态解析并转换为张量
        :param states: List[Dict], 批量状态，每个字典包含 position, marks_left, marks_pos
        :return: Tensor, 形状为 [batch_size, 3 + 2 * n_max_points]
        """
        if isinstance(states, dict):
            states = [states]
        batch_size = len(states)

        positions = torch.tensor([s["position"] for s in states], dtype=torch.float32).to(self.device)  # [batch_size,2]
        marks_left = torch.tensor([[s["marks_left"]] for s in states], dtype=torch.float32).to(
            self.device)  # [batch_size,1]
        marks_pos = torch.full((batch_size, self.n_max_points, 2), -1, dtype=torch.float32).to(
            self.device)  # [batch_size,n_max_points,2]

        for i, s in enumerate(states):
            pos_list = s["marks_pos"]
            try:
                if len(pos_list) > 0:
                    marks_pos[i, :len(pos_list), :] = torch.tensor(pos_list[:len(pos_list)], dtype=torch.float32)
            except RuntimeError as e:
                print(f"Error: {e}")
                print(f"pos_list: {pos_list}")

                raise  # 重新抛出异常
            # if len(pos_list) > 0:
            #     marks_pos[i, :len(pos_list),:] = torch.tensor(pos_list[:len(pos_list)],dtype=torch.float32)
        marks_pos_flat = marks_pos.view(batch_size, -1)
        state_tensor = torch.cat((positions, marks_left, marks_pos_flat), dim=1)
        # 修改一下
        # return state_tensor
        return positions

    def forward(self, states):
        """
        前向传播
        :param state: List[Dict], 批量状态，每个字典包含 position, marks_left, marks_pos
        :return: Q值，形状为 [batch_size, output_dim]
        """

        state_tensor = self.preprocess_state(states)

        x = F.relu(self.fc1(state_tensor))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class ReplayBuffer():
    """
    经验回放缓冲区
    """

    def __init__(self, length):
        self.buffer = deque(maxlen=length)
        self.length = length

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        return (states,
                actions,
                rewards,
                next_states,
                dones
                )

    def len(self, ):
        return len(self.buffer)


class DQN():

    def __init__(self, env, save_path, save_model_path, main_net, target_net, replaybuffer, device, iterations,
                 gamma=0.95, batch_size=256, lr=0.01):
        """
        初始化 DQN 类
        :param env: 强化学习环境实例
        :param main_net: 主神经网络
        :param target_net: 目标神经网络
        :param replaybuffer: 经验回放缓存区
        :param save_path: tensorboard保存区域
        :param save_model_path:模型保存区
        :param device: 运行设备（CPU/GPU）
        :param iterations: 训练迭代次数
        :param gamma: 折扣因子
        :param batch_size: 批量大小
        :param lr: 学习率
        """
        self.env = env
        self.save_path = save_path
        self.save_model_path = save_model_path
        self.device = device
        self.main_net = (main_net).to(self.device)
        self.target_net = target_net.to(self.device)
        self.replaybuffer = replaybuffer

        self.gamma = gamma
        self.iterations = iterations
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(self.main_net.parameters(), lr=lr)

        self.writer = SummaryWriter(save_path)
        self.state_distribution = []  # 用于记录状态

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            action = random.choice(self.env.action_space)
        else:
            with torch.no_grad():
                action = torch.argmax(self.main_net(state)).item()
        return action

    # 收集数据

    def collect_data(self, steps=5):
        for _ in range(steps):
            state = self.env.reset()
            done = False
            count = 0
            while not done:
                # ε-贪婪策略，概率 epsilon 选择随机动作
                action = self.select_action(state)
                # 执行动作
                next_state, reward, done, _ = self.env.step(action)
                self.replaybuffer.add(state, action, reward, next_state, done)

                # 记录当前状态
                self.state_distribution.append(state["position"])

                state = next_state
                count+=1

                if done:
                    break
                if count >1000:
                    break

    def collect_data_step(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            action = np.random.choice(self.env.action_space)
        else:
            with torch.no_grad():
                action = torch.argmax(self.main_net(state)).item()
        next_state, reward, done, _ = self.env.step(action)
        self.replaybuffer.add(state, action, reward, next_state, done)

        # 记录当前状态
        self.state_distribution.append(state["position"])

        return next_state, done ,reward

    def update_network(self):
        states, actions, rewards, next_states, dones = self.replaybuffer.sample(self.batch_size)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        dones = torch.tensor(dones, dtype=torch.int64).to(self.device)
        with torch.no_grad():
            target_q_values = self.target_net(next_states)
            target_q_values_max, _ = torch.max(target_q_values, dim=1)
            y_t = rewards + self.gamma * target_q_values_max * (1 - dones)
        q_values = self.main_net(states)
        q_values_for_actions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # squeeeze(1)去掉维度为1的维度是为什么：

        loss = F.mse_loss(q_values_for_actions, y_t)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), 1)
        self.optimizer.step()

        return loss.item()

    def train(self):
        """
        训练主网络
        """
        train_steps = 0

        for i in range(self.iterations):
            state = self.env.reset()

            total_reward = 0
            T = 200
            for t in range(T):

                next_state, done ,reward= self.collect_data_step(state)

                total_reward += reward

                if self.replaybuffer.len() > 0.5 * self.replaybuffer.length:

                    loss = self.update_network()
                    train_steps += 1

                    # 记录损失
                    self.writer.add_scalar("Train_loss", loss, global_step=train_steps)

                    # 每 5000 步 打印损失
                    # if train_steps % 500 == 0:
                    #     print("第{}轮损失,走了{}步长，训练步数：{}，损失函数大小:{}".format(i, t, train_steps, loss))

                    if train_steps % 10 == 0:
                        self.target_net.load_state_dict(self.main_net.state_dict())

                state = next_state
                if done:
                    break
            print("第{}次迭代，在总步数{}下，实际走了{}步，得到总回报{}".format(i, T, t+1 , total_reward))
            if i % 20==0:
                self.plot_state()
        self.writer.close()
        torch.save(self.main_net.state_dict(), self.save_model_path)

    def display_policy_matrix(self):
        """
        显示策略矩阵
        """
        test_net = self.main_net
        policy_matrix = np.zeros((self.env.num_states, len(self.env.action_space)))
        test_net.load_state_dict(torch.load(self.save_model_path))
        test_net.eval()

        for i in range(self.env.num_states):
            # 环境中没有 get_state_from_index函数，重写获取state的代码

            state = self.env.get_state_from_index(i)
            # 上一步获取坐标，网络需要传入字典
            state = {
                "position": state,
                "marks_left": 0,
                "marks_pos": []
            }
            q_values = test_net(state).unsqueeze(0)

            action = torch.argmax(q_values).item()

            # 强行修改终止状态的策略
            if state["position"] == self.env.target_state["position"]:
                action = 4

            policy_matrix[i, action] = 1

        return policy_matrix

    def plot_state(self):

        states_counts = Counter(tuple(map(tuple, self.state_distribution)))
        heatmap = np.zeros((self.env.env_size[0], self.env.env_size[0]))
        for (x, y), count in states_counts.items():
            heatmap[x, y] = count

        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap, cmap='viridis', annot=True, cbar=True)
        plt.title("State Visit Heatmap")
        plt.xlabel("State Dimension 1 (Discrete)")
        plt.ylabel("State Dimension 2 (Discrete)")
        plt.show()


if __name__ == "__main__":

    # 创建环境
    env = GridWorld()
    env.reset()

    device = torch.device("cuda")
    # 创建神经网络
    main_net = MyNet(device=device)
    target_net = copy.deepcopy(main_net)

    # 创建经验回放缓冲区
    replaybuffer = ReplayBuffer(1000)

    save_path_tensorboard = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    # 创建 DQN
    dqn = DQN(env, save_path_tensorboard, "dqn.pth", main_net, target_net, replaybuffer, device=torch.device("cuda"),
              iterations=100)

    for _ in range(1):
        dqn.collect_data()
    print(dqn.replaybuffer.len())
    # 训练
    dqn.train()
    dqn.plot_state()

    state = env.reset()
    for t in range(1000):
        env.render()
        action = torch.argmax(dqn.main_net(state)).item()
        next_state, reward, done, info = env.step(action)

        print(f"Step: {t}, Action: {action}, State: {next_state},Reward: {reward}, Done: {done}")

    env.render(animation_interval=100)

    # 显示策略矩阵
    # policy_matrix = dqn.display_policy_matrix()
    # env.reset()
    # env.render()
    # env.add_policy(policy_matrix)
    # env.render(animation_interval=100)
