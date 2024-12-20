# copilot: disable
import copy
import math
import random

from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import os
from torch.utils.tensorboard import SummaryWriter
from work_rl.env.grid_world_in_sssf import GridWorld
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self, device,output_dim=5, n_max_points=10,):
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
        if  isinstance(states, dict):
            states = [states]
        batch_size = len(states)


        positions = torch.tensor([s["position"] for s in states],dtype=torch.float32).to(self.device) #[batch_size,2]
        marks_left = torch.tensor([[s["marks_left"]] for s in states],dtype=torch.float32).to(self.device) #[batch_size,1]
        marks_pos = torch.full((batch_size, self.n_max_points, 2), -1, dtype=torch.float32).to(self.device) #[batch_size,n_max_points,2]

        for i,s in enumerate(states):
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
        state_tensor = torch.cat((positions,marks_left,marks_pos_flat),dim=1)
        #修改一下
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

class DQNAgent():
    def __init__(self,env, save_path, save_model_path, main_net, target_net, replaybuffer, device,
                 gamma=0.95, batch_size=128, lr=0.01, epsilon=0.1, epsilon_decay=0.995, min_epsilon=1e-8):
        """
        初始化 DQN 类
        :param env: 强化学习环境实例
        :param save_path:
        :param save_model_path:
        :param main_net:
        :param target_net:
        :param replaybuffer:
        :param device:
        :param gamma:
        :param batch_size:
        :param lr:
        :param epsilon:
        :param epsilon_decay:
        :param min_epsilon:
        """
        self.env = env
        self.save_path = save_path
        self.save_model_path = save_model_path
        self.device = device
        self.main_net = main_net
        self.target_net = target_net
        self.replaybuffer = replaybuffer

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.optimizer = torch.optim.SGD(self.main_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss().to(self.device)
        self.writer = SummaryWriter(log_dir=self.save_path)

    def train(self, episodes=4000,max_steps_per_episode=100,target_update_interval=5):

        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0
            train_steps=0
            for step in range(max_steps_per_episode):
                if np.random.rand() < self.epsilon:
                    action = random.sample(self.env.action_space, 1)[0]
                else:
                    action = torch.argmax(self.main_net(state)).item()

                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                self.replaybuffer.add(state, action, reward, next_state, done)
                state = next_state
                if self.replaybuffer.len() > self.batch_size:
                    loss = self.update_network()
                    train_steps += 1
                    if train_steps % 10==0:
                        self.target_net.load_state_dict(self.main_net.state_dict())
                    self.writer.add_scalar("Train-loss", loss, global_step=train_steps)
                    # if train_steps %10==0 and train_steps>0 :
                    #     self.target_net.load_state_dict(self.main_net.state_dict())
                if done:
                    print("此时达成目标，用了{}步长".format(step))
                    break
            if train_steps % 500==0:
                print(
                    f"Episode {ep}/{episodes}, Steps: {step + 1}, Total Reward: {total_reward}, Epsilon: {self.epsilon}")
            self.writer.add_scalar("Total_Reward_per_Episode", total_reward, ep)
            self.writer.add_scalar("Epsilon", self.epsilon, ep)



            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            if ep % 100 == 0:
                file_path = os.path.join("./model", "{}".format(ep), )
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                    torch.save(self.main_net.state_dict(), os.path.join(file_path, self.save_model_path))



    def update_network(self):
        states, actions, rewards, next_states, dones = self.replaybuffer.sample(self.batch_size)
        rewards = torch.tensor(rewards,dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions,dtype=torch.int64).to(self.device)
        dones = torch.tensor(dones,dtype=torch.int64).to(self.device)
        with torch.no_grad():
            target_q_values = self.target_net(next_states)
            target_q_values_max, _ = torch.max(target_q_values, dim=1)
            y_t = rewards+ self.gamma * target_q_values_max * (1-dones)
        q_values = self.main_net(states)
        q_values_for_actions = q_values.gather(1,actions.unsqueeze(1)).squeeze(1)     #squeeeze(1)去掉维度为1的维度是为什么：

        loss = self.loss_fn(q_values_for_actions,y_t)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_net.parameters(), 1)
        self.optimizer.step()
        return loss.item()

    def display_policy_matrix(self,test2):
        """
        根据训练好的网络，显示策略矩阵
        """
        #重写这个函数，增加训练好的智能体， 用于显示策略矩阵

        test_net = self.main_net
        test_net.load_state_dict(torch.load(os.path.join("model",test2, "model.pth"), weights_only=True))
        test_net.eval()
        policy_matrix = np.zeros((self.env.num_states, len(self.env.action_space)), dtype=int)

        for state_index in range(self.env.num_states):
            x = state_index % self.env.env_size[0]
            y = state_index // self.env.env_size[0]

            state = torch.tensor([x, y], dtype=torch.float32).to(self.device).unsqueeze(0)
            state = state / self.env.env_size[0]

            # 获取所有动作的 Q 值
            #传给网络的状态是一个字典，包含position,marks_left,marks_pos

            state_dict = {"position": [x,y],"marks_left": 0,"marks_pos": []}
            with torch.no_grad():
                q_values = test_net(state_dict)

            # 选择 Q 值最大的动作
            best_action_index = q_values.argmax().item()
            policy_matrix[state_index, best_action_index] = 1

        print("策略矩阵:")
        print(policy_matrix)
        return policy_matrix



if __name__== "__main__":
    env = GridWorld()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main_net = MyNet(device=device).to(device)
    target_net = MyNet(device=device).to(device)
    replaybuffer = ReplayBuffer(length=10000)
    save_path = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    save_model_path = "model.pth"
    agent = DQNAgent(env, save_path, save_model_path, main_net, target_net, replaybuffer, device)

    agent.train(episodes=6000,max_steps_per_episode=100,target_update_interval=10)

    # 验证智能体策略
    #启用保存的模型
    test2="5800"
    final_policy_matrix = agent.display_policy_matrix(test2)
    env.reset()
    env.render()
    env.add_policy(final_policy_matrix)
    env.render(animation_interval=100)



    agent.main_net.load_state_dict(torch.load(os.path.join("model", test2, "model.pth")))

    state = env.reset()
    for t in range(100):
        env.render()
        action = torch.argmax(agent.main_net(state)).item()
        state = env.agent_state
        next_state, reward, done, info = env.step(action)
        print(f"Step: {t}, Action: {action}, State: {next_state},Reward: {reward}, Done: {done}")



















