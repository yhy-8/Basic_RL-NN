import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import gymnasium as gym


# ==========================================
#    核心参数配置类
# ==========================================
class Config:
    def __init__(self):
        # 环境参数
        self.env_name = "Pendulum-v1"

        # 训练时随机初始化的范围 [角度范围, 角速度范围]
        self.train_theta_range = [-np.pi, np.pi]  # 角度 theta 范围 (弧度)
        self.train_thetadot_range = [-1.0, 1.0]  # 角速度 theta_dot 范围

        # 展示(Demo)时固定的初始状态基准 [theta, theta_dot]
        self.demo_fixed_state = [np.pi, 0.0]
        # 展示时的随机扰动范围 [角度扰动范围, 角速度扰动范围]
        # 如果设为 [0.2, 0.5]，则实际初始角度在 [np.pi-0.2, np.pi+0.2] 之间，角速度在 [-0.5, 0.5] 之间
        # 设为 [0.0, 0.0] 或 None 代表严格固定没有任何随机
        self.demo_random_range = [0.2, 0.5]

        # 神经网络超参数
        self.hidden_dim = 512  # 隐藏层神经元数量

        # PPO 算法超参数
        self.lr_actor = 3e-4  # Actor 学习率
        self.lr_critic = 1e-3  # Critic 学习率
        self.gamma = 0.99  # 折扣因子
        self.gae_lambda = 0.95  # GAE 参数
        self.clip_param = 0.2  # PPO 截断参数 (epsilon)
        self.K_epochs = 10  # 每次更新的网络迭代次数
        self.entropy_coef = 0.01  # 熵正则化系数 (鼓励探索)

        # 训练过程控制
        self.num_episodes = 1500  # 最大训练轮数
        self.max_steps = 200  # 每轮最大步数 (Pendulum默认200)
        self.update_timestep = 2000  # 收集多少步数据后进行一次PPO更新

        # 模型保存路径
        self.model_save_path = "models/ppo_pendulum.pth"

        # 设备
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
        self.train_ranges = train_ranges
        self.init_state = init_state
        self.noise_range = noise_range

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)

        if self.is_train_mode and self.train_ranges is not None:
            theta_range, dot_range = self.train_ranges
            theta = self.unwrapped.np_random.uniform(low=theta_range[0], high=theta_range[1])
            dot = self.unwrapped.np_random.uniform(low=dot_range[0], high=dot_range[1])
            self.unwrapped.state = np.array([theta, dot])

        elif self.init_state is not None:
            base_state = np.array(self.init_state)
            if self.noise_range is not None:
                noise = np.random.uniform(low=-1.0, high=1.0, size=2) * np.array(self.noise_range)
                self.unwrapped.state = base_state + noise
            else:
                self.unwrapped.state = base_state

        theta, thetadot = self.unwrapped.state
        obs = np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
        return obs, info


# ==========================================
# 2. 网络结构定义 (Actor & Critic)
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, config, state_dim, action_dim, max_action):
        super(ActorCritic, self).__init__()
        self.max_action = max_action

        self.actor = nn.Sequential(
            nn.Linear(state_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, action_dim),
            nn.Tanh()
        )
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))

        self.critic = nn.Sequential(
            nn.Linear(state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )

    def act(self, state):
        action_mean = self.actor(state) * self.max_action
        action_std = torch.exp(self.action_log_std).expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state) * self.max_action
        action_std = torch.exp(self.action_log_std).expand_as(action_mean)
        dist = Normal(action_mean, action_std)

        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


# ==========================================
# 3. 训练函数
# ==========================================
def train_ppo(config):
    raw_env = gym.make(config.env_name)
    env = CustomPendulumWrapper(
        raw_env,
        is_train_mode=True,
        train_ranges=[config.train_theta_range, config.train_thetadot_range]
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    model = ActorCritic(config, state_dim, action_dim, max_action).to(config.device)
    optimizer = optim.Adam([
        {'params': model.actor.parameters(), 'lr': config.lr_actor},
        {'params': model.action_log_std, 'lr': config.lr_actor},
        {'params': model.critic.parameters(), 'lr': config.lr_critic}
    ])

    memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []

    time_step = 0
    best_reward = -float('inf')

    base_name, ext = os.path.splitext(config.model_save_path)
    best_model_save_path = f"{base_name}_best{ext}"

    print(f"========== 准备开始训练 ==========")
    print(f"正在使用的计算设备: [{config.device}]")
    if config.device.type == 'cuda':
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"最大回合: {config.num_episodes}")
    print(f"==================================")

    for episode in range(1, config.num_episodes + 1):
        # 直接 reset 即可获得随机化初始状态
        state, _ = env.reset()
        ep_reward = 0

        for t in range(config.max_steps):
            time_step += 1

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.device)
            action, action_logprob = model.act(state_tensor)

            action_np = action.cpu().numpy().flatten()
            action_np = np.clip(action_np, -model.max_action, model.max_action)

            next_state, reward, done, truncated, _ = env.step(action_np)

            memory_states.append(state)
            memory_actions.append(action.cpu().numpy().flatten())
            memory_logprobs.append(action_logprob.cpu().item())
            memory_rewards.append((reward + 8.0) / 8.0)

            state = next_state
            ep_reward += reward

            # PPO 更新逻辑
            if time_step % config.update_timestep == 0:
                old_states = torch.FloatTensor(np.array(memory_states)).to(config.device)
                old_actions = torch.FloatTensor(np.array(memory_actions)).to(config.device)
                old_logprobs = torch.FloatTensor(memory_logprobs).to(config.device)

                rewards = []
                discounted_reward = 0
                for r in reversed(memory_rewards):
                    discounted_reward = r + (config.gamma * discounted_reward)
                    rewards.insert(0, discounted_reward)

                rewards = torch.FloatTensor(rewards).to(config.device)

                for _ in range(config.K_epochs):
                    logprobs, state_values, dist_entropy = model.evaluate(old_states, old_actions)
                    state_values = torch.squeeze(state_values)

                    advantages = rewards - state_values.detach()
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

                    ratios = torch.exp(logprobs - old_logprobs)
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - config.clip_param, 1 + config.clip_param) * advantages
                    loss_actor = -torch.min(surr1, surr2).mean()

                    loss_critic = nn.MSELoss()(state_values, rewards)
                    loss = loss_actor + 0.5 * loss_critic - config.entropy_coef * dist_entropy.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []

            if done or truncated:
                break

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(model.state_dict(), best_model_save_path)
            print(f"Episode {episode}: 发现新 Best Reward: {best_reward:.2f} -> 已保存")
        elif episode % 10 == 0:
            print(f"Episode: {episode} \t Reward: {ep_reward:.2f}")

    torch.save(model.state_dict(), config.model_save_path)
    env.close()


# ==========================================
# 4. 演示函数 (展示训练好的策略)
# ==========================================
def demo_ppo(config, input_model_path=None):
    model_path = input_model_path if input_model_path else config.model_save_path
    if not os.path.exists(model_path):
        print(f"未找到模型: {model_path}")
        return

    print(f"========== 开始 PPO 连续决策演示 ==========")
    print(f"正在使用的计算设备: [{config.device}]")
    if config.device.type == 'cuda':
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"加载模型: {model_path}")
    print(f"====================================")

    raw_env = gym.make(config.env_name, render_mode="human", max_episode_steps=config.max_steps)
    env = CustomPendulumWrapper(
        raw_env,
        init_state=config.demo_fixed_state,
        noise_range=config.demo_random_range
    )

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    model = ActorCritic(config, state_dim, action_dim, max_action).to(config.device)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.eval()

    for ep in range(3):
        # 演示模式重置，自带扰动逻辑
        state, _ = env.reset()
        ep_reward = 0

        for t in range(config.max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.device)
            with torch.no_grad():
                action_mean = model.actor(state_tensor) * model.max_action

            action_np = action_mean.cpu().numpy().flatten()
            action_np = np.clip(action_np, -model.max_action, model.max_action)

            state, reward, done, truncated, _ = env.step(action_np)
            ep_reward += reward

            if done or truncated:
                break

        print(f"演示轮次 {ep + 1} 结束, 获得奖励: {ep_reward:.2f}")

    env.close()

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == '__main__':
    # 实例化配置
    cfg = Config()

    # 1. 训练
    train_ppo(cfg)

    # 2. 演示
    cfg.demo_fixed_state = [np.pi, 0.0]  # 可以在这里快速配置测试初始环境
    cfg.demo_random_range = 0.1
    demo_ppo(cfg,'models/ppo_pendulum_best.pth')