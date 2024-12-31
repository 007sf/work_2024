import copy
import math
import random

from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
from collections import deque, Counter
import os

from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from work_rl.env.grid_world_in_sssf import GridWorld
import torch.nn.functional as F
import seaborn as sns


class MyNet(nn.Module):
    def __init__(self, device, output_dim=8, n_max_points=10, ):
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
        state_tensor_3_dim = torch.cat((positions, marks_left), dim=1)
        # 修改一下
        # return state_tensor_3_dim
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


# 根据定义的网络，写一个DQN类

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
        self.state_distribution = []

    # 收集数据

    def collect_data(self, steps=100):

        state = env.reset()

        # 限制最大步数

        for _ in range(steps):
            done = False
            while not done:
                # 随机选择动作, 'list' object has no attribute 'sample'：
                action = random.choice(env.action_space)

                next_state, reward, done, _ = env.step(action)
                self.replaybuffer.add(state, action, reward, next_state, done)
                self.state_distribution.append(state["position"])
                state = next_state


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
        for i in range(self.iterations):

            loss = self.update_network()

            # 记录损失
            self.writer.add_scalar("Train_loss", loss, global_step=i)

            # 更新目标网络
            if i % 10 == 0:
                self.target_net.load_state_dict(self.main_net.state_dict())

            # 每 5000 步打印损失
            if i % 5000 == 0:
                print("第{}轮损失函数大小:{}".format(i, loss))

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

            state = self.env.get_state_from_index(i)
            # 上一步获取坐标，网络需要传入字典
            state = {
                "position": state,
                "marks_left": 0,
                "marks_pos": []
            }
            q_values = test_net(state).unsqueeze(0)

            action = torch.argmax(q_values).item()
            policy_matrix[i, action] = 1
        print(policy_matrix)
        return policy_matrix

    #路径追踪
    def display_policy_matrix2(self):
        """
        显示从初始状态到目标状态的策略路径
        """
        test_net = self.main_net
        policy_matrix = np.zeros((self.env.num_states, len(self.env.action_space)))
        test_net.load_state_dict(torch.load(self.save_model_path))
        test_net.eval()

        # 从初始状态开始追踪路径
        current_state = self.env.start_state
        path = []
        count = 0
        while True:
            # 将当前状态转为字典格式
            state = {
                "position": current_state["position"],
                "marks_left": current_state["marks_left"],
                "marks_pos": current_state["marks_pos"]
            }

            # 获取当前状态的最佳动作
            with torch.no_grad():
                q_values = test_net(state).unsqueeze(0)
                action = torch.argmax(q_values).item()

            # 将状态和动作记录到路径中
            path.append((current_state, action))
            policy_matrix[self.env.get_index_from_state(current_state), action] = 1

            # 执行动作，获取下一个状态
            next_state, _, done, _ = self.env.step(action)
            print("marks_left:{}".format(next_state["marks_left"]))

            if done:  # 到达目标状态停止
                break

            count +=1
            if count >100:
                break

            current_state = next_state

        # 将策略矩阵返回，只用于从路径上显示策略
        return policy_matrix, path
    def plot_state(self):

        states_counts = Counter(tuple(map(tuple, self.state_distribution)))
        heatmap = np.zeros((self.env.env_size[0], self.env.env_size[0]))
        for (x, y), count in states_counts.items():
            heatmap[y, x] = count

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
    replaybuffer = ReplayBuffer(10000)

    save_path_tensorboard = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    # 创建 DQN
    dqn = DQN(env, save_path_tensorboard, "dqn.pth", main_net, target_net, replaybuffer, device=torch.device("cuda"),
              iterations=8000)

    # 收集多组数据
    for _ in range(10):
        dqn.collect_data()



    dqn.train()



    #常规显示
    # policy_matrix = dqn.display_policy_matrix()
    # env.reset()
    # env.render()
    # env.add_policy(policy_matrix)

    # 显示轨迹策略矩阵
    env.reset()
    policy_matrix,path = dqn.display_policy_matrix2()
    env.render()
    env.add_policy2(policy_matrix,path)


    env.render(animation_interval=1000)
