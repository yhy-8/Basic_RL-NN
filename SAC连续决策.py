import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
import random


# ==========================================
#   核心配置参数 (Config)
# ==========================================
class Config:
    def __init__(self):
        self.env_name = 'Pendulum-v1'

        # 训练时随机初始化的范围 [角度范围, 角速度范围]
        self.train_theta_range = [-np.pi, np.pi]  # 角度 theta 范围 (弧度)
        self.train_thetadot_range = [-1.0, 1.0]  # 角速度 theta_dot 范围

        # 展示(Demo)时固定的初始状态基准 [theta, theta_dot]
        self.demo_fixed_state = [np.pi, 0.0]
        # 展示时的随机扰动范围 [角度扰动范围, 角速度扰动范围]
        # 如果设为 [0.2, 0.5]，则实际初始角度在 [np.pi-0.2, np.pi+0.2] 之间，角速度在 [-0.5, 0.5] 之间
        # 设为 [0.0, 0.0] 或 None 代表严格固定没有任何随机
        self.demo_random_range = [0.2, 0.5]

        # 神经网络参数
        self.hidden_dim = 256  # 隐藏层神经元数量

        # 训练超参数
        self.num_episodes = 150  # 训练总轮数
        self.max_steps = 200  # 每轮最大步数
        self.batch_size = 256  # 批大小
        self.buffer_size = 100000  # 经验回放池大小
        self.actor_lr = 3e-4  # Actor 学习率
        self.critic_lr = 3e-4  # Critic 学习率
        self.alpha_lr = 3e-4  # 温度参数 Alpha 学习率
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.005  # 目标网络软更新系数

        # SAC 特定参数
        self.alpha = 0.2  # 初始温度参数 (决定探索程度)
        self.adaptive_alpha = True  # 是否自动调整温度参数 alpha

        # 模型保存路径
        self.model_save_path = 'models/sac_pendulum.pth'

        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else "cpu")


# ==========================================
# 1. 环境包装器 (Wrapper) - 专门管理初始状态
# ==========================================
class CustomPendulumWrapper(gym.Wrapper):
    """接管 Pendulum 初始状态逻辑的包装器"""

    def __init__(self, env, is_train_mode=False, train_ranges=None, init_state=None, noise_range=None):
        super().__init__(env)
        self.is_train_mode = is_train_mode
        self.train_ranges = train_ranges  # 格式: [[theta_min, theta_max], [dot_min, dot_max]]
        self.init_state = init_state  # 格式: [theta, theta_dot]
        self.noise_range = noise_range  # 格式: [theta_noise, dot_noise]

    def reset(self, **kwargs):
        # 1. 先调用底层的 reset
        state, info = self.env.reset(**kwargs)

        # 2. 拦截并篡改内部状态
        if self.is_train_mode and self.train_ranges is not None:
            # 训练模式：在指定大范围内完全随机
            theta_range, dot_range = self.train_ranges
            theta = self.unwrapped.np_random.uniform(low=theta_range[0], high=theta_range[1])
            dot = self.unwrapped.np_random.uniform(low=dot_range[0], high=dot_range[1])
            self.unwrapped.state = np.array([theta, dot])

        elif self.init_state is not None:
            # 演示模式：基准点 + 随机扰动
            base_state = np.array(self.init_state)
            if self.noise_range is not None:
                noise = np.random.uniform(low=-1.0, high=1.0, size=2) * np.array(self.noise_range)
                self.unwrapped.state = base_state + noise
            else:
                self.unwrapped.state = base_state

        # 3. 根据修改后的状态，重新计算并返回给网络的 Observation
        theta, thetadot = self.unwrapped.state
        obs = np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

        return obs, info


# ==========================================
# 2. 经验回放池 (Replay Buffer)
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# ==========================================
# 3. 神经网络结构 (Actor & Critic)
# ==========================================
class ValueNetwork(nn.Module):
    """ Critic网络：评估 Q(s, a) """

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        q1 = F.relu(self.linear1(x))
        q1 = F.relu(self.linear2(q1))
        q1 = self.linear3(q1)

        q2 = F.relu(self.linear4(x))
        q2 = F.relu(self.linear5(q2))
        q2 = self.linear6(q2)
        return q1, q2


class PolicyNetwork(nn.Module):
    """ Actor网络：输出动作的均值和标准差 """

    def __init__(self, state_dim, action_dim, hidden_dim, action_scale=1.0):
        super(PolicyNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.action_scale = action_scale

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


# ==========================================
# 4. SAC 智能体 (Agent)
# ==========================================
class SACAgent:
    def __init__(self, state_dim, action_dim, action_scale, config):
        self.config = config
        self.device = config.device

        self.actor = PolicyNetwork(state_dim, action_dim, config.hidden_dim, action_scale).to(self.device)
        self.critic = ValueNetwork(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic_target = ValueNetwork(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        if config.adaptive_alpha:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=config.alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = config.alpha

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * self.actor.action_scale
        else:
            action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self, replay_buffer):
        state, action, reward, next_state, done = replay_buffer.sample(self.config.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            q_target = reward + (1 - done) * self.config.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        action_new, log_prob = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, action_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.config.adaptive_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)


# ==========================================
# 5. 训练封装函数 (train_sac)
# ==========================================
def train_sac(config):
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    raw_env = gym.make(config.env_name)
    # 使用包装器接管训练初始化
    env = CustomPendulumWrapper(
        raw_env,
        is_train_mode=True,
        train_ranges=[config.train_theta_range, config.train_thetadot_range]
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    agent = SACAgent(state_dim, action_dim, action_scale, config)
    memory = ReplayBuffer(config.buffer_size)

    best_reward = -float('inf')
    rewards_history = []

    base_name, ext = os.path.splitext(config.model_save_path)
    best_model_save_path = f"{base_name}_best{ext}"

    print(f"========== 准备开始训练 ==========")
    print(f"正在使用的计算设备: [{config.device}]")
    if config.device.type == 'cuda':
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"最大回合: {config.num_episodes}")
    print(f"==================================")

    for episode in range(1, config.num_episodes + 1):
        # 这里的 reset 会自动调用包装器里的随机逻辑，主循环变得非常干净！
        state, _ = env.reset()
        episode_reward = 0

        for step in range(config.max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            memory.push(state, action, reward, next_state, terminated)

            state = next_state
            episode_reward += reward

            if len(memory) > config.batch_size:
                agent.update(memory)

            if terminated or truncated:
                break

        rewards_history.append(episode_reward)

        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(agent.actor.state_dict(), best_model_save_path)
            print(f"Episode {episode:3d} | Reward: {episode_reward:7.1f} | [🌟 已保存最佳模型]")
        elif episode % 10 == 0:
            print(f"Episode {episode:3d} | Reward: {episode_reward:7.1f}")

    torch.save(agent.actor.state_dict(), config.model_save_path)
    print(f"=== 训练结束 ===")
    env.close()


# ==========================================
# 6. 展示封装函数 (demo_sac)
# ==========================================
def demo_sac(config, input_model_path):
    model_path = input_model_path if input_model_path else config.model_save_path
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return

    raw_env = gym.make(config.env_name, render_mode="human", max_episode_steps=config.max_steps)
    # 使用包装器接管演示初始化
    env = CustomPendulumWrapper(
        raw_env,
        init_state=config.demo_fixed_state,
        noise_range=config.demo_random_range
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    actor = PolicyNetwork(state_dim, action_dim, config.hidden_dim, action_scale).to(config.device)
    actor.load_state_dict(torch.load(model_path, map_location=config.device))
    actor.eval()

    print(f"========== 开始 SAC 连续决策演示 ==========")
    print(f"正在使用的计算设备: [{config.device}]")
    if config.device.type == 'cuda':
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"加载模型: {model_path}")
    print(f"====================================")

    for episode in range(3):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(config.max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.device)
            with torch.no_grad():
                mean, _ = actor(state_tensor)
                action = torch.tanh(mean) * action_scale
            action = action.cpu().numpy()[0]

            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        print(f"Demo 回合 {episode + 1} | 总奖励: {episode_reward:.1f}")

    env.close()


# ==========================================
# 7. 主程序执行入口
# ==========================================
if __name__ == '__main__':
    cfg = Config()

    # 1. 训练
    train_sac(cfg)

    # 3. 展示
    cfg.demo_fixed_state = [np.pi, 0.0] # 可以在这里快速配置测试初始环境
    cfg.demo_random_range = [0.2, 0.5]
    demo_sac(cfg, 'models/sac_pendulum_best.pth')