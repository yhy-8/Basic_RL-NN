import gymnasium as gym
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os


# ==========================================
# 第一步：全局参数类封装 (Configuration)
# ==========================================
class Config:
    def __init__(self):
        # 1. 环境与状态参数
        self.env_name = "CartPole-v1"  # Gym 环境名称
        self.max_steps_per_episode = 1000  # 每回合最大步数限制
        self.position_limit = 2.0  # 小车左右移动的极限距离
        self.train_random_range = 0.1  # 训练时的初始状态随机范围 (增加随机性，防止过拟合)
        self.test_fixed_state = [0.0, 0.0, np.pi, 0.0]  # 测试/演示时的固定初始状态 (正下方)
        self.test_random_range = 0.1  # 测试/演示时的随机干扰范围 (设为0即为纯固定)

        # 2. 神经网络参数
        self.hidden_dim = 512  # 隐藏层神经元数量
        self.lr_actor = 0.0003  # 策略网络 (Actor) 学习率
        self.lr_critic = 0.0003  # 价值网络 (Critic) 学习率

        # 3. PPO 核心参数
        self.gamma = 0.99  # 折扣因子
        self.k_epochs = 15  # 每次 PPO 更新时，复用这批数据的迭代次数
        self.eps_clip = 0.2  # PPO 裁剪系数
        self.entropy_coef = 0.01  # 熵正则化系数

        # 4. 训练与系统参数
        self.update_timestep = 2000  # 收集多少步(steps)的数据后进行一次 PPO 网络更新
        self.num_episodes = 1000  # 最大训练回合数
        self.model_save_path = "models/ppo_cartpole.pth"  # 保存路径

        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if torch.backends.mps.is_available() else "cpu")


# ==========================================
# 第二步：环境魔改 Wrapper (Swing-up 魔改)
# ==========================================
class CartPoleSwingUpWrapper(gym.Wrapper):
    """魔改 CartPole：控制初始状态随机性，限制小车范围，修改奖励函数"""

    def __init__(self, env, config, is_training=True):
        super().__init__(env)
        self.cfg = config
        self.is_training = is_training
        self.env.unwrapped.x_threshold = self.cfg.position_limit

    def reset(self, **kwargs):
        self.env.reset(**kwargs)

        if self.is_training:
            # 训练模式：纯正下方 + 随机扰动
            random_noise = np.random.uniform(
                low=-self.cfg.train_random_range,
                high=self.cfg.train_random_range,
                size=(4,)
            )
            base_state = np.array([0.0, 0.0, np.pi, 0.0], dtype=np.float32)
            self.env.unwrapped.state = base_state + random_noise
        else:
            # 演示模式
            base_state = np.array(self.cfg.test_fixed_state, dtype=np.float32)
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

        # 1. 角度奖励：越向上越接近1，越向下越接近-1
        upright_reward = np.cos(theta)

        # 2. 位置惩罚：鼓励保持在中央，越靠近边缘惩罚越大（平方项让边缘惩罚更剧烈）
        center_penalty = 0.5 * (x / self.cfg.position_limit) ** 2

        # 3. 速度惩罚：防止小车像疯了一样无限加速狂转，促使它在顶部悬停
        vel_penalty = 0.01 * (theta_dot ** 2)

        reward = upright_reward - center_penalty - vel_penalty

        return state, reward, terminated, truncated, info


# ==========================================
# 第三步：网络与组件定义区
# ==========================================
class RolloutBuffer:
    def __init__(self):
        self.actions, self.states, self.logprobs = [], [], []
        self.rewards, self.is_terminals, self.values = [], [], []

    def clear(self):
        del self.actions[:], self.states[:], self.logprobs[:]
        del self.rewards[:], self.is_terminals[:], self.values[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Actor 输出层使用极小的 gain，让初始动作概率接近均匀分布，鼓励早期探索
        nn.init.orthogonal_(self.actor[-2].weight, gain=0.01)

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        value = self.critic(state)
        return action.detach(), dist.log_prob(action).detach(), value.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        return dist.log_prob(action), self.critic(state), dist.entropy()


# ==========================================
# 第四步：PPO Agent 类
# ==========================================
class PPOAgent:
    def __init__(self, config, state_dim, action_dim):
        self.cfg = config
        self.policy = ActorCritic(state_dim, action_dim, config.hidden_dim).to(config.device)
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': config.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': config.lr_critic}
        ])
        self.policy_old = ActorCritic(state_dim, action_dim, config.hidden_dim).to(config.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = RolloutBuffer()
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.cfg.device)
            action, action_logprob, value = self.policy_old.act(state_tensor)

        self.buffer.states.append(state_tensor)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.values.append(value.item())
        return action.item()

    def update(self, next_state_value, next_done):
        # 使用 GAE (广义优势估计)
        returns = []
        advantages = []
        gae = 0

        rewards = self.buffer.rewards
        is_terminals = self.buffer.is_terminals
        values = self.buffer.values

        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_val = next_state_value
                next_non_terminal = 1.0 - int(next_done)
            else:
                next_val = values[step + 1]
                next_non_terminal = 1.0 - int(is_terminals[step + 1])

            # 计算 TD 误差
            delta = rewards[step] + self.cfg.gamma * next_val * next_non_terminal - values[step]
            # 计算 GAE (此处硬编码 lambda = 0.95 提升稳定性)
            gae = delta + self.cfg.gamma * 0.95 * next_non_terminal * gae

            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])

        returns = torch.tensor(returns, dtype=torch.float32).to(self.cfg.device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.cfg.device)

        old_states = torch.stack(self.buffer.states, dim=0).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).detach()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach()

        # 对 Advantage 进行标准化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        for _ in range(self.cfg.k_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * self.loss_fn(state_values, returns)

            loss = actor_loss + critic_loss - self.cfg.entropy_coef * dist_entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            # 增加梯度裁剪防止网络崩溃
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()


# ==========================================
# 第五步：训练与评估函数
# ==========================================
def train_ppo(config):
    # 确保保存目录存在
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)

    base_env = gym.make(config.env_name, max_episode_steps=config.max_steps_per_episode)
    base_env.unwrapped.theta_threshold_radians = float('inf')
    env = CartPoleSwingUpWrapper(base_env, config, is_training=True)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(config, state_dim, action_dim)
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

    for ep in range(1, config.num_episodes + 1):
        state, _ = env.reset()
        ep_reward = 0
        temp_step = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            agent.buffer.rewards.append(reward)
            # 只有出界失败才算 Terminal。超时截断不属于物理世界的失败。
            agent.buffer.is_terminals.append(terminated)

            state = next_state
            ep_reward += reward
            time_step += 1
            temp_step += 1

            if time_step % config.update_timestep == 0:
                if terminated:
                    next_state_value = 0.0
                    next_done = True
                else:
                    with torch.no_grad():
                        st_tensor = torch.FloatTensor(next_state).to(config.device)
                        next_state_value = agent.policy_old.critic(st_tensor).item()
                    next_done = False

                agent.update(next_state_value, next_done)

            if done:
                break

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(agent.policy_old.state_dict(), best_model_save_path)
            print(f"🌟 回合 {ep}: 发现新最高分 {best_reward:.2f}，最优模型已保存到 {best_model_save_path}！")

        if ep % 20 == 0:
            print(f"回合: {ep}\t 累计步数: {time_step}\t 本回合步数：{temp_step}\t 本回合得分: {ep_reward:.2f}")

    torch.save(agent.policy_old.state_dict(), config.model_save_path)
    print(f"✅ 训练全部完成！最终迭代模型已保存到 {config.model_save_path}")
    env.close()


def demo_ppo(config, input_model_path: str = None):
    model_path = input_model_path if input_model_path else config.model_save_path

    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return

    print(f"========== 开始 PPO 离散决策演示 ==========")
    print(f"正在使用的计算设备: [{config.device}]")
    if config.device.type == 'cuda':
        print(f"GPU 型号: {torch.cuda.get_device_name(0)}")
    print(f"加载模型: {model_path}")
    print(f"====================================")

    base_env = gym.make(
        config.env_name,
        render_mode="human",
        max_episode_steps=config.max_steps_per_episode,
        disable_env_checker=True
    )
    base_env.unwrapped.theta_threshold_radians = float('inf')

    env = CartPoleSwingUpWrapper(base_env, config, is_training=False)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(config, state_dim, action_dim)
    agent.policy_old.load_state_dict(torch.load(model_path, map_location=config.device, weights_only=True))
    agent.policy_old.eval()

    for ep in range(3):
        state, _ = env.reset()
        ep_reward = 0
        step_count = 0

        while True:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(config.device)
                action_probs = agent.policy_old.actor(state_tensor)
                action = torch.argmax(action_probs).item()

            state, reward, terminated, truncated, _ = env.step(action)

            ep_reward += reward
            step_count += 1

            if getattr(config, 'max_steps_per_episode', None) and step_count >= config.max_steps_per_episode:
                print(f"⏱️ 达到最大步数限制 ({config.max_steps_per_episode}步)，强制结束。")
                truncated = True

            time.sleep(0.02)

            if terminated or truncated:
                print(f"🏁 演示回合 {ep + 1} 结束，总步数: {step_count}，得分: {ep_reward:.2f}")
                break

    env.close()
    print("========== 演示结束 ==========")


if __name__ == '__main__':
    cfg = Config()

    # 1. 训练
    train_ppo(cfg)

    # 2. 演示
    cfg.test_fixed_state = [0.0, 0.0, np.pi, 0.0]  # 可以在这里快速配置测试初始环境
    cfg.test_random_range = 0.1
    demo_ppo(cfg, 'models/ppo_cartpole_best.pth')