import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from agent_dir.agent import Agent
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        hidden_size = 512
        # 定义第一个线性层，从输入层到隐藏层
        self.fc1 = nn.Linear(input_size[0], hidden_size)
        # 定义第二个线性层，从隐藏层到输出层
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        # 将输入数据通过第一个线性层，并使用ReLU激活函数
        x = F.relu(self.fc1(inputs))
        # 将ReLU层的输出通过第二个线性层得到最终的Q值
        x = self.fc2(x)
        return x
    

class ReplayBuffer: #本质是一个存储经验数据的列表
    def __init__(self, buffer_size):
        # 初始化缓冲区大小和创建一个空的列表来存储经验元组
        self.buffer_size = buffer_size
        self.buffer = []

    def __len__(self):
        # 返回缓冲区当前存储的经验数量
        return len(self.buffer)

    def push(self, *transition):
        # 如果缓冲区未满，添加新的经验元组
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition)
        else:  # 如果缓冲区已满，用新的经验替换最旧的经验
            self.buffer.pop(0)
            self.buffer.append(transition)

    def sample(self, batch_size):  #从缓冲区中随机采样一个批次的过渡，用于训练。
        index = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in index]
        return zip(*batch)

    def clean(self):
        # 清空缓冲区，移除所有存储的经验
        self.buffer.clear()





class AgentDQN(Agent): #继承自agent类
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(AgentDQN, self).__init__(env)
        self.env = env
        self.hidden_size = args.hidden_size


        # 输入是状态空间，输出是动作空间
        self.input_size = env.observation_space.shape
        self.output_size = env.action_space.n
        print("-------------------------------------")
        print("observation_space: ", self.input_size)
        print("action_space: ", self.output_size)

        # 创建评估网络和目标网络，二者初始完全相同，并具有相同的和权重和参数更新评率
        self.eva_net = QNetwork(self.input_size, self.output_size).to(device)
        # 评估网络，创建一个Qnet作为主网络，用于选择和训练
        self.tar_net = QNetwork(self.input_size, self.output_size).to(device)
        # 目标网络，创建另一个Qnet作为目标网络，用于稳定学习的过程
        self.tar_net.load_state_dict(self.eva_net.state_dict())
        self.target_update_freq = args.target_update_freq

        # 定义神经网络的优化器和损失函数
        self.optim = optim.Adam(self.eva_net.parameters(), lr=args.lr)
        self.loss_func = nn.MSELoss()

        # DQN参数
        self.gamma = args.gamma  # 在计算reward时候使用的折扣因子

        # 经验回放缓冲区参数
        self.buffer_size = args.buffer_size
        self.buffer = ReplayBuffer(args.buffer_size)
        self.batch_size = args.batch_size

        # 贪婪策略，用以选择动作
        self.eps = 1  # 贪婪决策概率
        self.eps_start = 1  # 初始贪婪决策概率
        self.eps_end = 0.01  # 最终贪婪决策概率

        # 训练参数
        self.n_episodes = args.n_episodes  # 迭代次数
        self.test = args.test  # 是否处于测试模式
        self.grad_norm_clip = args.grad_norm_clip  # 梯度剪裁,防止梯度爆炸

        # 记录reward，用于作图
        self.episode_rewards = []

        
    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        self.eva_net.load_state_dict(torch.load('network_params.pth', map_location='cpu'))
        # 加载保存的模型参数文件,映射到CPU上
        self.tar_net.load_state_dict(self.eva_net.state_dict())
        # 将评估网络参数复制到目标网络

    def train(self):
        """
        Implement your training algorithm here
        """
        # 从经验回放缓冲区 self.buffer 中随机采样 self.batch_size 条经验。
        obs, action, reward, next_obs, dones = self.buffer.sample(self.batch_size)

        # 将采样得到的数据转换为PyTorch张量,并移动到device
        action = torch.tensor(np.array(action), dtype=torch.long).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device)
        reward = torch.tensor(np.array(reward), dtype=torch.float32).to(device)
        obs = torch.tensor(np.array(obs), dtype=torch.float32).to(device)
        next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32).to(device)

        # 使用评估网络self.eva_net根据当前观察状态obs和采取的动作actions评估Q值，并计算下一个状态的最大Q值
        q_eval = self.eva_net(obs).gather(-1, action.unsqueeze(-1)).squeeze(-1)
        q_next = self.tar_net(next_obs).detach()
        # 使用greedy策略计算目标值
        q_target = reward + self.gamma * (1 - dones) * torch.max(q_next, dim=-1)[0]

        # 计算损失
        Loss = self.loss_func(q_eval, q_target)
        self.optim.zero_grad()  # 清零之前的梯度
        Loss.backward()  # 反向传播计算梯度
        self.optim.step()  # 优化器进行参数w更新
        return Loss.item()  # 返回损失值

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:observation
        Return:action
        """
        #在测试模式中选择概率固定为0.01
        if self.test:
            self.eps = 0.01
        else:
            # 根据 eps 的起始值 eps_start 和最终目标值 eps_end 计算线性衰减的探索率，并确保它不会低于 eps_end
            self.eps = self.eps - (self.eps_start - self.eps_end)/100000
            self.eps = max(self.eps, self.eps_end)

        # 随机数小于 eps，则随机选择一个动作，这有助于探索
        # 如果随机数大于或等于 eps，则使用目标网络 tar_net 来评估观察状态 observation 的动作值，然后选择具有最大动作值的动作
        if np.random.uniform() <= self.eps: 
            action = np.random.randint(0, self.output_size)
        else:
            # 在最外层再添加一个维度
            observation = torch.tensor(observation, dtype=torch.float32).to(device)
            action_value = self.tar_net(observation)
            action = torch.max(action_value, dim=-1)[1].cpu().numpy()
        return int(action)

    def run(self):
        step = 0
        for i_episode in range(self.n_episodes):
            obs = self.env.reset()[0]
            episode_reward = 0
            done = False
            loss_avg = 0  # 每次迭代的平均loss
            loss = []  # 统计每步的loss
            # 在每次迭代的按步进行
            while not done:
                action = self.make_action(obs, self.test) # 动作选择
                next_obs, reward, done, truncated, info = self.env.step(action) # 创建四元组，表示每步的各个参数
                self.buffer.push(obs, action, reward, next_obs, done) # 将四元组放入经验回放池
                episode_reward += reward # 计算总reward
                obs = next_obs # 向后迭代

                # 如果步数过小则不需要训练模型，只是收集经验
                if step >= self.batch_size*500:
                    loss.append(self.train()) # 训练

                # 目标网络更新
                if step % 1000 == 0:  # 在一定步数中固定目标网络的参数，不进行更新
                    self.tar_net.load_state_dict(self.eva_net.state_dict())
                step += 1
                if done or truncated:
                    break

            # 将当前episode的奖励存储到数组中
            self.episode_rewards.append(episode_reward)

            if len(loss):
                loss_avg = sum(loss)/len(loss)
            print("-------------------------------")
            print("episode: " + str(i_episode))
            print("reward: " + str(episode_reward))
            print("eps: " + str(self.eps))
            print("loss_avg: ", loss_avg)

            # if self.episode_rewards:  # 检查列表是否不为空
            #     max_reward = max(self.episode_rewards)
            #     print("Max Reward so far:" + str(max_reward))

        # 绘制最终图表
        self.plot_rewards(final=True)
        # 保存网络参数
        torch.save(self.tar_net.state_dict(), 'network_params.pth')

    def plot_rewards(self, final=False):
        """
        绘制episode奖励曲线图。

        :param final: 是否是最终图表，如果是，则显示图表而不是立即关闭。
        """
        plt.figure(figsize=(10, 5))  # 设置图像大小
        plt.plot(self.episode_rewards, label='Episode Rewards')  # 绘制奖励曲线

        # 添加标题和坐标轴标签
        plt.title('Training Progress Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')

        plt.grid(True)

        # 添加图例
        plt.legend()

        # 显示图表
        if final:
            plt.show()
        else:
            plt.draw()  # 更新图表
            plt.pause(0.1)  # 短暂暂停以显示图表