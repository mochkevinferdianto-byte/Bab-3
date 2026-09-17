# Contoh yang disesuaikan untuk Gymnasium (mengatasi error NumPy 2.0 / Python 3.12)
import gymnasium as gym

# Gunakan CartPole-v1 karena v0 sudah usang
env = gym.make('CartPole-v1', render_mode='human')

for i_episode in range(20):
    # Di Gymnasium, reset mengembalikan (observation, info)
    observation, info = env.reset()
    
    for t in range(100):
        env.render()
        print(observation)
        
        action = env.action_space.sample()
        
        # Di Gymnasium, step mengembalikan 5 nilai: observation, reward, terminated, truncated, info
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Kondisi selesai (done) gabungan dari terminated atau truncated
        done = terminated or truncated
        
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()