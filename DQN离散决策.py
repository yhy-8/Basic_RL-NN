import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import math
import os
from collections import deque


# ==========================================
# 0. 超参数配置中心 (方便统一调节和阅读)
# ==========================================
class Config:
    def __init__(self):
        # 1. 环境与状态参数
        self.env_name = "CartPole-v1"  # Gym 环境名称
        self.position_limit = 2.0  # 小车左右移动的极限距离
        self.train_random_range = 0.1  # 训练时的初始状态随机范围 (增加随机性，防止过拟合)
        self.test_fixed_state = [0.0, 0.0, np.pi, 0.0]  # 测试/演示时的固定初始状态 (正下方)
        self.test_random_range = 0.1  # 测试/演示时的随机干扰范围 (设为0即为纯固定)
        self.max_steps_per_episode = 1000  # 每回合最大步数

        # 2. 神经网络参数
        self.hidden_dim = 512  # 隐藏层神经元数量
        self.learning_rate = 1e-3  # 学习率
        self.grad_clip = 100  # 梯度裁剪阈值 (防止梯度爆炸，极大提升稳定性)

        # 3. DQN 核心参数
        self.batch_size = 64  # 每次从回放池取出的样本数
        self.gamma = 0.99  # 折扣因子 (越接近1越看重长远利益)
        self.buffer_capacity = 50000  # 经验回放池容量
        self.target_update_freq = 10  # 目标网络更新频率 (每 N 个回合更新一次)

        # 4. 探索率 (Epsilon Greedy)
        self.eps_start = 1.0  # 初始探索率
        self.eps_end = 0.1  # 最低探索率
        self.eps_decay_steps = 50000  # 衰减步数 (在这个总步数时，衰减到约 36%)

        # 5. 训练参数
        self.num_episodes = 1000  # 最大训练回合数
        self.model_save_path = "models/dqn_cartpole.pth" # 保存路径

        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else "cpu")


# ==========================================
# 1. 环境魔改 Wrapper (Swing-up 魔改)
# ==========================================
class CartPoleSwingUpWrapper(gym.Wrapper):
    """魔改 CartPole：统一管理状态随机性、角度归一化、限制范围与奖励函数"""

    def __init__(self, env, config, is_training=True):
        super().__init__(env)
        self.cfg = config
        self.is_training = is_training
        # 同步修改底层 env 的 x_threshold，这样 render 的时候画面比例才正确
        self.env.unwrapped.x_threshold = self.cfg.position_limit

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        base_state = np.array(self.cfg.test_fixed_state, dtype=np.float32)

        if self.is_training:
            # 训练模式：加入训练范围的随机扰动
            random_noise = np.random.uniform(
                low=-self.cfg.train_random_range,
                high=self.cfg.train_random_range,
                size=(4,)
            )
            self.env.unwrapped.state = base_state + random_noise
        else:
            # 演示模式：加入测试范围的随机扰动
            if getattr(self.cfg, 'test_random_range', 0.0) > 0:
                random_noise = np.random.uniform(
                    low=-self.cfg.test_random_range,
                    high=self.cfg.test_random_range,
                    size=(4,)
                )
                self.env.unwrapped.state = base_state + random_noise
            else:
                self.env.unwrapped.state = base_state

        self.env.unwrapped.steps_beyond_terminated = None
        return np.array(self.env.unwrapped.state, dtype=np.float32), {}

    def step(self, action):
        state, _, _, truncated, info = self.env.step(action)
        x, x_dot, theta, theta_dot = state

        # 终止条件：只判断小车是否超出了我们自定义的极限距离
        terminated = bool(x < -self.cfg.position_limit or x > self.cfg.position_limit)

        # 角度归一化 [-π, π]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        state[2] = theta

        # 0. 存活奖励：只要没出界，每一步都给 1 分
        survival_reward = 1.0

        # 1. 角度奖励：越向上越接近 1，越向下越接近 -1
        upright_reward = np.cos(theta)

        # 2. 位置惩罚：鼓励保持在中央，越靠近边缘惩罚越大
        center_penalty = 0.1 * (x / self.cfg.position_limit)

        # 3. 速度惩罚：防止小车像疯了一样无限加速狂转，促使它在顶部悬停
        vel_penalty = 0.01 * (theta_dot ** 2)

        # 总奖励
        reward = survival_reward + upright_reward - center_penalty - vel_penalty

        if terminated:
            reward += -10.0  # 边界惩罚

        return state, reward, terminated, truncated, info


# ==========================================
# 2. 定义神经网络 (Q-Network)
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ==========================================
# 3. 定义经验回放池 (Replay Buffer)
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# ==========================================
# 4. 训练过程
# ==========================================
def train_dqn(config: Config):
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
    print(f"========== 准备开始训练 ==========")
    print(f"正在使用的计算设备: [{config.device}]")
    if config.device.type == 'cuda':
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"最大回合: {config.num_episodes}")
    print(f"==================================")

    # 接入 Wrapper 进行训练环境配置
    base_env = gym.make(config.env_name).unwrapped
    base_env.theta_threshold_radians = float('inf')
    env = CartPoleSwingUpWrapper(base_env, config, is_training=True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = QNetwork(state_dim, action_dim, config.hidden_dim).to(config.device)
    target_net = QNetwork(state_dim, action_dim, config.hidden_dim).to(config.device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=config.learning_rate)
    memory = ReplayBuffer(config.buffer_capacity)

    global_steps = 0
    best_avg_reward = -float('inf')
    recent_rewards = deque(maxlen=10)

    def select_action(state, current_steps):
        eps_threshold = config.eps_end + (config.eps_start - config.eps_end) * \
                        math.exp(-1. * current_steps / config.eps_decay_steps)
        if random.random() > eps_threshold:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.device)
                return policy_net(state_tensor).argmax().item()
        else:
            return env.action_space.sample()

    def optimize_model():
        if len(memory) < config.batch_size:
            return

        states, actions, rewards, next_states, dones = memory.sample(config.batch_size)

        states = torch.FloatTensor(states).to(config.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(config.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(config.device)
        next_states = torch.FloatTensor(next_states).to(config.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(config.device)

        curr_q_values = policy_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * config.gamma * max_next_q_values

        loss = nn.MSELoss()(curr_q_values, target_q_values)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), config.grad_clip)
        optimizer.step()

    # --- 主训练循环 ---
    for episode in range(config.num_episodes):
        # 直接 reset 即可，Wrapper 已经接管了随机状态的生成逻辑
        state, _ = env.reset()
        total_reward = 0
        step = 0

        for step in range(config.max_steps_per_episode):
            action = select_action(state, global_steps)
            global_steps += 1

            # 直接步进环境，所有的角度归一化、自定义奖励和越界均在 Wrapper 中完成了
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            optimize_model()

            if done:
                break

        recent_rewards.append(total_reward)
        avg_reward = sum(recent_rewards) / len(recent_rewards)

        if len(recent_rewards) == recent_rewards.maxlen and avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            best_model_path = config.model_save_path.replace(".pth", "_best.pth")
            torch.save(policy_net.state_dict(), best_model_path)
            print(f"🌟 [新突破] Episode: {episode:4d} | 最近10局均分达 {avg_reward:.1f}，模型已备份至 {best_model_path}")

        if episode % config.target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 20 == 0:
            current_eps = config.eps_end + (config.eps_start - config.eps_end) * math.exp(
                -1. * global_steps / config.eps_decay_steps)
            print(f"Episode: {episode:4d} | 存活步数: {(step+1):3d} | 当局奖励: {total_reward:6.1f} | 10局均分: {avg_reward:6.1f} | Epsilon: {current_eps:.3f}")

    env.close()

    torch.save(policy_net.state_dict(), config.model_save_path)
    print(f"==================================")
    print(f"训练彻底结束！")
    print(f"最终状态模型已保存至: {config.model_save_path}")
    print(f"🏆 历史表现最好的模型保存在: {config.model_save_path.replace('.pth', '_best.pth')} (历史最高均分: {best_avg_reward:.1f})")
    print(f"==================================")


# ==========================================
# 5. 演示过程
# ==========================================
def demo_dqn(config: Config, input_model_path: str = None):
    model_path = input_model_path if input_model_path else config.model_save_path

    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return

    print(f"========== 开始 DQN 离散决策演示 ==========")
    print(f"正在使用的计算设备: [{config.device}]")
    if config.device.type == 'cuda':
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"加载模型: {model_path}")
    print(f"====================================")

    # 接入 Wrapper 进行演示环境配置 (is_training=False)
    base_env = gym.make(config.env_name, render_mode="human").unwrapped
    base_env.theta_threshold_radians = float('inf')
    env = CartPoleSwingUpWrapper(base_env, config, is_training=False)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = QNetwork(state_dim, action_dim, config.hidden_dim).to(config.device)
    policy_net.load_state_dict(torch.load(model_path, map_location=config.device, weights_only=True))
    policy_net.eval()

    for ep in range(3):
        state, _ = env.reset()

        done = False
        total_reward = 0
        step_count = 0

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.device)
                action = policy_net(state_tensor).argmax().item()

            state, reward, terminated, truncated, _ = env.step(action)
            step_count += 1

            # 额外的最大步数检测
            if getattr(config, 'max_steps_per_episode', None) and step_count >= config.max_steps_per_episode:
                print(f"⏱️ 达到最大步数限制 ({config.max_steps_per_episode}步)，强制结束。")
                truncated = True

            if terminated and reward == -10.0:
                 print(f"⚠️ 小车触碰边界，当前回合结束。")

            done = terminated or truncated
            total_reward += reward
            time.sleep(0.02)

        print(f"🏁 演示回合 {ep + 1} 结束，总步数: {step_count}，得分: {total_reward:.1f}")

    env.close()
    print("========== 演示结束 ==========")


# ==========================================
# 6. 启动入口
# ==========================================
if __name__ == "__main__":
    cfg = Config()

    # 1. 训练
    train_dqn(cfg)

    # 2. 演示
    cfg.test_fixed_state = [0.0, 0.0, np.pi, 0.0] # 可以在这里快速配置测试初始环境
    cfg.test_random_range = 0.1
    demo_dqn(cfg, "models/dqn_cartpole_best.pth")