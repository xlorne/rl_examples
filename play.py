import gym

env = gym.make('CartPole-v1', render_mode='human')
env.reset()

done = False
while not done:
    env.render()
    action = env.action_space.sample()
    next_state, reward, done, _, _ = env.step(action)

env.close()
