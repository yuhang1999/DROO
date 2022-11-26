"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.

torch实现DDPG算法
"""
import torch
import numpy as np
import torch.nn as nn

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_default_dtype(torch.float)

print(torch.__version__)

# Actor Net
# Actor：输入是state，输出的是一个确定性的action
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # layer
        self.layer_1 = nn.Linear(state_dim, 120)
        self.layer_2 = nn.Linear(120, 80)
        self.layer_3 = nn.Linear(80, action_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, s):
        a = self.relu(self.layer_1(s))
        a = self.relu(self.layer_2(a))
        a = self.sigmoid(self.layer_3(a))
        return a


# Critic Net
# Critic输入的是当前的state以及Actor输出的action,输出的是Q-value
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim + action_dim, 120)
        self.layer_2 = nn.Linear(120, 80)
        self.layer_3 = nn.Linear(80, 1)
        self.relu = nn.ReLU()

    def forward(self, s, a):

        x = torch.cat([s, a], 1)
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        q_val = self.layer_3(x)
        return q_val


# Deep Deterministic Policy Gradient
class DDPG(object):
    def __init__(self, state_dim, action_dim, action_bound, replacement, memory_capacity=1000, gamma=0.9, lr_a=0.001,
                 lr_c=0.002, batch_size=100, training_interval=10,):
        super(DDPG, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_capacity = memory_capacity
        self.replacement = replacement
        self.t_replace_counter = 0
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.batch_size = batch_size
        self.training_interval = training_interval  # learn every #training_interval

        # 记忆库
        self.memory = np.zeros((memory_capacity, state_dim + action_dim))
        self.pointer = 0
        # 定义 Actor 网络
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        # 定义 Critic 网络
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        # 定义优化器
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        # 选取损失函数
        self.mse_loss = nn.MSELoss()

        # store training cost
        self.a_cost_his = []
        self.c_cost_his = []

    def sample(self):
        if self.pointer > self.memory_capacity:
            indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        else:
            indices = np.random.choice(self.pointer, size=self.batch_size)
        return self.memory[indices, :]

    def choose_action(self, s):
        s = torch.Tensor(s[np.newaxis, :])
        # print("s=", s.shape)
        # s = torch.FloatTensor(s)
        action = self.actor(s)
        return action.detach().numpy()[0][0]

    def store_transition(self, h, m):
        transition = np.hstack((h, m))
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition
        self.pointer += 1

    def encode(self, h, m):
        # encoding the entry
        self.store_transition(h, m)
        # train the DNN every 10 step
        #        if self.memory_counter> self.memory_size / 2 and self.memory_counter % self.training_interval == 0:
        if self.pointer % self.training_interval == 0:
            self.learn()

    def learn(self):

        # # soft replacement and hard replacement
        # # 用于更新target网络的参数
        # if self.replacement['name'] == 'soft':
        #     # soft的意思是每次learn的时候更新部分参数
        #     tau = self.replacement['tau']
        #     a_layers = self.actor_target.named_children()
        #     c_layers = self.critic_target.named_children()
        #     for al in a_layers:
        #         a = self.actor.state_dict()[al[0] + '.weight']
        #         al[1].weight.data.mul_((1 - tau))
        #         al[1].weight.data.add_(tau * self.actor.state_dict()[al[0] + '.weight'])
        #         al[1].bias.data.mul_((1 - tau))
        #         al[1].bias.data.add_(tau * self.actor.state_dict()[al[0] + '.bias'])
        #     for cl in c_layers:
        #         cl[1].weight.data.mul_((1 - tau))
        #         cl[1].weight.data.add_(tau * self.critic.state_dict()[cl[0] + '.weight'])
        #         cl[1].bias.data.mul_((1 - tau))
        #         cl[1].bias.data.add_(tau * self.critic.state_dict()[cl[0] + '.bias'])
        #
        # else:
        #     # hard的意思是每隔一定的步数才更新全部参数
        #     if self.t_replace_counter % self.replacement['rep_iter'] == 0:
        #         self.t_replace_counter = 0
        #         a_layers = self.actor_target.named_children()
        #         c_layers = self.critic_target.named_children()
        #         for al in a_layers:
        #             al[1].weight.data = self.actor.state_dict()[al[0] + '.weight']
        #             al[1].bias.data = self.actor.state_dict()[al[0] + '.bias']
        #         for cl in c_layers:
        #             cl[1].weight.data = self.critic.state_dict()[cl[0] + '.weight']
        #             cl[1].bias.data = self.critic.state_dict()[cl[0] + '.bias']
        #
        #     self.t_replace_counter += 1

        # 从记忆库中采样bacth data
        batch_memory = self.sample()
        h_train = torch.FloatTensor(batch_memory[:, 0: self.state_dim])
        m_train = torch.FloatTensor(batch_memory[:, self.state_dim:])

        # 训练Actor
        m = self.actor(h_train)
        q = self.critic(h_train, m)
        a_loss = -torch.mean(q)
        self.aopt.zero_grad()
        a_loss.backward(retain_graph=True)
        self.aopt.step()

        # 训练critic
        m_ = self.actor_target(h_train)
        q_target = self.critic_target(h_train, m_)

        q_eval = self.critic(h_train, m_train)
        td_error = self.mse_loss(q_target, q_eval)
        self.copt.zero_grad()
        td_error.backward()
        self.copt.step()

        self.a_cost = a_loss.item()
        assert (self.a_cost < 0)
        self.a_cost_his.append(self.a_cost)
        self.c_cost = td_error.item()
        assert (self.c_cost > 0)
        self.c_cost_his.append(self.c_cost)



    def plot_a_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.a_cost_his))*self.training_interval, self.a_cost_his)
        plt.ylabel('Actor Training Loss')
        plt.xlabel('Time Frames')
        plt.show()

    def plot_c_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.c_cost_his))*self.training_interval, self.c_cost_his)
        plt.ylabel('Critic Training Loss')
        plt.xlabel('Time Frames')
        plt.show()


import time

if __name__ == '__main__':

    # hyper parameters
    VAR = 3  # control exploration
    MAX_EPISODES = 500
    MAX_EP_STEPS = 200
    MEMORY_CAPACITY = 10000
    REPLACEMENT = [
        dict(name='soft', tau=0.005),
        dict(name='hard', rep_iter=600)
    ][0]  # you can try different target replacement strategies

    ENV_NAME = 'Pendulum-v1'
    RENDER = False

    # train
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    ddpg = DDPG(state_dim=s_dim,
                action_dim=a_dim,
                action_bound=a_bound,
                replacement=REPLACEMENT,
                memory_capacity=MEMORY_CAPACITY)

    t1 = time.time()
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()

            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, VAR), -2, 2)  # 在动作选择上添加随机噪声

            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r / 10, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                VAR *= .9995  # decay the action randomness
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS - 1:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % VAR, )
                if ep_reward > -300: RENDER = True
                break

    print('Running time: ', time.time() - t1)