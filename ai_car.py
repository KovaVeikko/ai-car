import gym
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle


# settings
render = True
print_policy = True
resume = True
file_name = 'lr001d20e001.p'

# hyper parameters
D = 20
MAX_ITERATIONS = 500000
LEARNING_RATE_INITIAL = 0.5
LEARNING_RATE_MINIMUM = 0.01
EPSILON_INITIAL = 0.1
EPSILON_MINIMUM = 0.01

def init_model(environment):
    q_table = np.zeros((D, D, environment.action_space.n))
    model = {
        'q_table': q_table,
        'iteration': 0,
        'rewards': [],
        'max_reward': -200,
    }
    if resume:
        try:
            model = pickle.load(open(file_name, 'rb'))
        except:
            pass
    return model

def observation_to_table_index(environment, observation):
    low = environment.observation_space.low
    high = environment.observation_space.high
    delta_x = (high - low) / D
    s1 = int((observation[0] - low[0])/delta_x[0])
    s2 = int((observation[1] - low[1])/delta_x[1])
    return s1, s2

def sample_action(environment, q_table, s1, s2, iteration):
    decay = 0.9 ** (iteration // 100)
    epsilon = max(EPSILON_INITIAL * decay, EPSILON_MINIMUM)
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice(environment.action_space.n)
    else:
        q_values_exp = np.exp(q_table[s1][s2])
        probabilities = q_values_exp / np.sum(q_values_exp)
        action = np.random.choice(environment.action_space.n, p=probabilities)
    return action

def update_table(environment, observation, q_table, s1, s2, action, reward, learning_rate):
    s1_next, s2_next = observation_to_table_index(environment, observation)
    q_table[s1][s2][action] = (1 - learning_rate) * q_table[s1][s2][action] + learning_rate * (reward + np.max(q_table[s1_next][s2_next]))
    return learning_rate

def compute_learing_rate(iteration):
    decay = 1/(LEARNING_RATE_INITIAL * np.sqrt(iteration * 0.01) + 1)
    learning_rate = max(LEARNING_RATE_INITIAL * decay, LEARNING_RATE_MINIMUM)
    return learning_rate

def main():
    environment = gym.make('MountainCar-v0')
    model = init_model(environment)
    q_table = model['q_table']
    iteration = model['iteration']
    rewards = model['rewards']
    max_avg_reward = model['max_reward']
    reward_sum = 0
    step = 0
    for i in range(iteration, MAX_ITERATIONS):
        step += 1
        observation = environment.reset()
        while True:
            if render and i % 1000 == 0:
                environment.render()
            s1, s2 = observation_to_table_index(environment, observation)
            action = sample_action(environment, q_table, s1, s2, i)
            observation, reward, done, _ = environment.step(action)
            learning_rate = compute_learing_rate(i)
            update_table(environment, observation, q_table, s1, s2, action, reward, learning_rate)
            reward_sum += reward
            if done:
                break
        if i % 100 == 0 and step != 1:
            avg_reward = reward_sum/100
            if avg_reward > max_avg_reward:
                max_avg_reward = avg_reward
            rewards.append(avg_reward)
            reward_sum = 0
            print('Iteration: {}, high: {},  average reward: {}, lr: {}'.format(i, max_avg_reward, avg_reward, learning_rate))
            if (print_policy):
                policy = np.argmax(q_table, axis=2)
                print("Policy table:")
                print(policy[::2, ::2])
            model['q_table'] = q_table
            model['iteration'] = i
            model['rewards'] = rewards
            model['max_reward'] = max_avg_reward
            pickle.dump(model, open(file_name, 'wb'))
            plt.plot(rewards)
            plt.savefig(file_name + 'ng')

if __name__ == '__main__':
    main()
