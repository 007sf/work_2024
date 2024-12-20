import random
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import os
from torch.utils.tensorboard import SummaryWriter
from bishe_rl.env.grid_world_in_sssf import GridWorld
import torch.nn.functional as F


class MyNet(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.fc_fixed = nn.Linear(3, 128)  # 3:2D坐标+1D标量
        self.lstm = nn.LSTM(input_size=2, hidden_size=128, batch_first=True)
        self.fc1 = nn.Linear(128 + 128, 64)
        self.fc2 = nn.Linear(64, 6)
        self.device = device

    # def preprocess_states(self,states):
    #     positions = torch.tensor([s["position"] for s in states],dtype= torch.float32).to(self.device)
    #     marks_left = torch.tensor([[s["markd_left"]] for s in states])
    def process_marks_pos(self,marks_pos):
        # 遍历每个样本
        processed_marks_pos = []
        for seq in marks_pos:
            if len(seq) == 0:
                # 将空列表填充为 [-1, -1]
                processed_marks_pos.append([[-1, -1]])
            else:
                processed_marks_pos.append(seq)
        return processed_marks_pos

    def forward(self, position,marks_left,marks_pos):
        # position = torch.tensor([state["position"]], dtype=torch.float32).to(self.device)
        # marks_left = torch.tensor([[state["marks_left"]]], dtype=torch.int64).to(self.device)
        marks_pos = self.process_marks_pos(marks_pos)
        if len(marks_pos) == 0:
            # 使用一个全零的张量代替 lstm_output
            lstm_output = torch.zeros(1, 128, device=self.device, dtype=torch.float32).to(self.device)  # 假设 hidden_size=128
            aa=2
        else:
            print(marks_pos)
            placed_coords = torch.tensor(marks_pos, dtype=torch.float32).to(self.device)
            lstm_out, (hn, cn) = self.lstm(placed_coords)  # LSTM输出的隐状态
            lstm_output = hn[-1].unsqueeze(0)
            aa=1
        fixed_input = torch.cat((position, marks_left), dim=1)
        fixed_output = F.relu(self.fc_fixed(fixed_input))
        combined_input = torch.cat((fixed_output, lstm_output), dim=1)

        x = F.relu(self.fc1(combined_input))
        q_val = self.fc2(x)
        return q_val


class ReplayBuffer():
    def __init__(self, length):
        self.buffer = deque(maxlen=length)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def process_batch(self,states):
        positions = torch.tensor([s["position"] for s in states],dtype=torch.float32).to(device).view(-1,1)
        marks_lefts = torch.tensor([s["marks_left"] for s in states],dtype=torch.int64).to(device).view(-1,1)
        marks_pos =[s["marks_pos"] for s in states]
        batch = (positions,marks_lefts,marks_pos)
        return batch

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        states = self.process_batch(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = self.process_batch(next_states)
        dones = np.array(dones)
        return (states,
                actions,
                rewards,
                next_states,
                dones
                )

    def len(self, ):
        return len(self.buffer)


class DQN():
    def __init__(self, env, save_path_tensorboard, save_model_path, main_net, target_net, replaybuffer, device,
                 gamma=0.8, batch_size=128, lr=0.01, epsilon=0.1, epsilon_decay=0.995, min_epsilon=1e-8):
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
        self.save_path = save_path_tensorboard
        self.save_model_path = save_model_path
        self.device = device
        self.main_net = (main_net).to(self.device)
        self.target_net = target_net.to(self.device)
        self.replaybuffer = replaybuffer

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.optimizer = torch.optim.SGD(self.main_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss().to(self.device)
        self.writer = SummaryWriter(log_dir=self.save_path)  # 可选的日志记录器，例如 TensorBoard

    def train(self, episodes=1000, max_steps_per_episode=100, target_update_interval=5):
        """
        融合数据采集与训练的完整流程。
        :param episodes: 总训练回合数
        :param max_steps_per_episode: 每个episode最大步数上限
        :param target_update_interval: 每隔多少个episode更新目标网络参数
        """
        train_steps = 0
        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0
            for step in range(max_steps_per_episode):
                if np.random.rand() < self.epsilon:
                    action = random.sample(self.env.action_space, 1)[0]
                else:
                    pos= torch.tensor([state["position"]],dtype=torch.float32).to(self.device)
                    left = torch.tensor([[state["marks_left"]]], dtype=torch.int64).to(self.device)
                    action = torch.argmax(self.main_net(pos,left,state["marks_pos"])).item()


                next_state, reward, done, _ = self.env.step(action)

                total_reward += reward

                self.replaybuffer.add(state, action, reward, next_state, done)
                state = next_state

                if self.replaybuffer.len() > self.batch_size:
                    loss = self.update_network()
                    train_steps += 1
                    self.writer.add_scalar("Train-loss", loss, global_step=train_steps)

                if done:
                    break
            if ep % 50 == 0:
                print(
                    f"Episode {ep}/{episodes}, Steps: {step + 1}, Total Reward: {total_reward}, Epsilon: {self.epsilon}")

            self.writer.add_scalar("Total_Reward_per_Episode", total_reward, ep)
            self.writer.add_scalar("Epsilon", self.epsilon, ep)

            if ep % target_update_interval == 0:
                self.target_net.load_state_dict(self.main_net.state_dict())

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if ep % 100 == 0:
                file_path = os.path.join("./model", "{}".format(ep), )
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                    torch.save(self.main_net.state_dict(), os.path.join(file_path, self.save_model_path))
        self.writer.close()

    def update_network(self):
        states, actions, rewards, next_states, dones = self.replaybuffer.sample(self.batch_size)
        rewards = torch.tensor(rewards,dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions,dtype=torch.int64).to(self.device)
        dones = torch.tensor(dones,dtype=torch.float32).to(self.device)

        with torch.no_grad():
            # pos2 = next_states["position"].to(self.device)
            # left2 = next_states["marks_left"].to(self.device)
            target_q_values = self.target_net(*next_states)
            target_q_values_max, _ = torch.max(target_q_values, dim=1)
            y_t = rewards + self.gamma * target_q_values_max * (1 - dones)

        q_values = self.main_net(*states)
        q_values_for_actions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.loss_fn(q_values_for_actions, y_t)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.main_net.parameters(), max_norm=1)
        self.optimizer.step()
        return loss.item()

    def display_policy_matrix(self):
        """
        根据训练好的网络，显示策略矩阵
        """

        test_net = self.main_net
        test_net.load_state_dict(torch.load(self.save_model_path, weights_only=True))
        test_net.eval()
        policy_matrix = np.zeros((self.env.num_states, len(self.env.action_space)), dtype=int)

        for state_index in range(self.env.num_states):
            x = state_index % self.env.env_size[0]
            y = state_index // self.env.env_size[0]

            state = torch.tensor([x, y], dtype=torch.float32).to(self.device).unsqueeze(0)
            state = state / self.env.env_size[0]

            # 获取所有动作的 Q 值
            with torch.no_grad():
                q_values = test_net(state)

            # 选择 Q 值最大的动作
            best_action_index = q_values.argmax().item()
            policy_matrix[state_index, best_action_index] = 1

        print("策略矩阵:")
        print(policy_matrix)
        return policy_matrix


if __name__ == "__main__":
    "初始化参数"
    # --------------------dqn迭代次数-------------------------
    iterations = 7000

    # -------------------------------------------------------
    device = torch.device("cuda")
    main_net = MyNet(device=device)
    target_net = MyNet(device=device)
    replaybuffer = ReplayBuffer(length=10000)
    np.random.seed(40)
    env = GridWorld()

    base_log_dir = "dqn_logs"
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
    save_path = os.path.join(base_log_dir, TIMESTAMP, )
    save_model_path = "E:\\bishe\\bishe_rl\\save_model\\train_model.pth"

    device = torch.device("cuda")

    dqn = DQN(env, save_path, save_model_path, main_net, target_net, replaybuffer, device, )

    dqn.train(episodes=1000)
