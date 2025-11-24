import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def shaped_reward(state, reward):
    goal_state = 63
    if reward == 1:
        return 1.0
    dist = abs(goal_state - state)
    shaped = (1 - dist / 63) * 0.05
    shaped -= 0.005
    return shaped

def run(episodes, is_training=True, render=False):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True,
                   render_mode='human' if render else None)

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        with open('frozen_lake8x8(1).pkl', 'rb') as f:
            q = pickle.load(f)

    learning_rate_a = 0.9
    discount_factor_g = 0.9
    epsilon = 1.0
    epsilon_decay_rate = 0.0001
    min_epsilon = 0.05

    rng = np.random.default_rng()

    rewards = np.zeros(episodes)
    avg_rewards = np.zeros(episodes)
    success = np.zeros(episodes)
    success_rate = np.zeros(episodes)

    window = 50

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    (reward_line,) = ax1.plot([], [], label="Avg Shaped Reward (Last 50)", linewidth=2)
    (succ_line,) = ax1.plot([], [], label="Success Rate (Last 50)", linewidth=2)
    ax1.set_title("FrozenLake 8×8 — Training Performance (50-episode window)")
    ax1.set_ylabel("Value (0 - 1)")
    ax1.set_xlim(0, episodes)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="upper left")
    ax1.grid(True)

    (eps_line,) = ax2.plot([], [], label="Epsilon (exploration)", linewidth=2)
    ax2.set_title("Epsilon Decay")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Epsilon")
    ax2.set_xlim(0, episodes)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True)

    plt.tight_layout()

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        ep_reward = 0.0
        reached_goal = False

        while not (terminated or truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)

            final_reward = shaped_reward(state, reward)
            ep_reward += final_reward

            if reward == 1:
                reached_goal = True

            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (
                    final_reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state

        rewards[i] = ep_reward
        success[i] = 1 if reached_goal else 0

        start = max(0, i - window + 1)
        avg_rewards[i] = np.mean(rewards[start: i + 1])
        success_rate[i] = np.mean(success[start: i + 1])

        epsilon = max(min_epsilon, epsilon - epsilon_decay_rate)
        epsilon = max(min_epsilon, epsilon * np.exp(-0.000001))

        xdata = np.arange(i + 1)

        reward_line.set_xdata(xdata)
        reward_line.set_ydata(avg_rewards[:i + 1])

        succ_line.set_xdata(xdata)
        succ_line.set_ydata(success_rate[:i + 1])

        eps_line.set_xdata(xdata)
        eps_line.set_ydata(np.minimum(1.0, np.ones(i + 1) * epsilon))

        ax1.relim()
        ax1.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()

        plt.pause(0.001)

    env.close()

    plt.ioff()
    plt.savefig("frozen_lake8x8_training_two_subplots.png")
    plt.show()

    if is_training:
        with open("frozen_lake8x8(1).pkl", "wb") as f:
            pickle.dump(q, f)

if __name__ == '__main__':
    run(episodes=15000, is_training=True, render=False)
