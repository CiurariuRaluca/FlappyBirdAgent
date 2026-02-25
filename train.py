

import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import cv2
import gc


log_file = f"data_{__import__('datetime').datetime.now().strftime('%H%M%S')}.txt"

EPISODE_NUMBER = 10000
GAMMA = 0.99
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.01
#EXPLORATION_STEPS = 20000
EPSILON_DECAY = 0.9991  # decay exponential per episod nu per steps
REPLAY_MEMORY_BUFFER = 60000# de la 50000 la 60000
BATCH_SIZE = 64  # marire batch 32->64
LEARNING_RATE = 3e-4
TARGET_UPDATE_FREQUENCY = 100  # micsorat de la 100 la 50 deoare dqn e mai mic acum/ vreau sa fac update mai rar
PRB_ALPHA=0.6# nivel de prioritate
PRB_BETA_START=0.4#
PRB_BETA_END=1.0

#RANDOM_FLAP_PROB = 0.08  # când random, flap doar 8%


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # optimizare


def preprocess_image(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # rgb->gray

    image = cv2.resize(image, (84, 84))  # redimensionare
    #_, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)  # alb-negru
    image = image.astype(np.float32) / 255.0

    return image.astype(np.float32)

class PrioritizedReplayBuffer:

    def __init__(self,capacity,alpha=PRB_ALPHA):
      self.capacity=capacity
      self.alpha=alpha
      self.buffer=[]
      self.priorities=[]
      self.position=0

    def push_exp(self,exp,td_error=1.0):
      priority=(abs(td_error)+0.01)**self.alpha

      if len(self.buffer)<self.capacity:
        self.buffer.append(exp)
        self.priorities.append(priority)
      else:
        self.buffer[self.position]=exp
        self.priorities[self.position]=priority

      self.position=(self.position+1)%self.capacity

    def sample_exp(self,batch_size,beta=PRB_BETA_START):
      priorities=np.array(self.priorities)
      probabilities=priorities/priorities.sum()

      indices=np.random.choice(len(self.buffer),batch_size,p=probabilities)
      samples=[self.buffer[i] for i in indices]

      weights=(len(self.buffer)*probabilities[indices])**(-beta)
      weights/=weights.max()

      return samples,indices,torch.tensor(weights,dtype=torch.float32,device=device)

    def update_priorities(self,indices,td_errors):
      for index,td_error in zip(indices,td_errors):
        self.priorities[index]=(abs(td_error)+0.01)**self.alpha

    def __len__(self):
      return len(self.buffer)


class DeepQNetwork(nn.Module):

    def __init__(self, height, width, output_size):
        super(DeepQNetwork, self).__init__()

        #self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)  # 4 canale,32 filters de 8x8 si stride=4
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # 32 canale,64 filtre de 4x4 si stride=2
        #self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # 64 canale, 64 filtre de 3x3 si stride=1
        #self.fc1 = nn.Linear(64 * 7 * 7, 512)
        #self.fc2 = nn.Linear(512, output_size)#incerc alta arhitectura poate asta e prea mare

        self.conv1=nn.Conv2d(4,16,kernel_size=8,stride=4)
        self.conv2=nn.Conv2d(16,32,kernel_size=4,stride=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        #x = torch.relu(self.conv1(x))
        #x = torch.relu(self.conv2(x))
        #x = torch.relu(self.conv3(x))
        #x = torch.relu(self.fc1(x.view(x.size(0), -1)))#vechiul feed forward
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)


class FlappyBirdAgent:

    def __init__(self):
        self.memory = PrioritizedReplayBuffer(capacity=REPLAY_MEMORY_BUFFER)
        self.policy_network = DeepQNetwork(84, 84, 2).to(device)

        self.target_network = DeepQNetwork(84, 84, 2).to(device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=LEARNING_RATE)
        self.epsilon = INITIAL_EPSILON
        self.criterion = nn.SmoothL1Loss(reduction='none')# huber loss pt stabilitate
        self.episode_losses = []
        self.beta=PRB_BETA_START



    def sample_action(self, state):
        if random.random() < self.epsilon:
          flap_prob = max(0.05, (1.0/6.0) * self.epsilon)  
          return 1 if random.random() < flap_prob else 0

        with torch.no_grad():
            return self.policy_network(state.to(device)).max(1)[1].item()

    def add_exp(self,state,action,reward,next_state,final):
      max_priority=max(self.memory.priorities) if self.memory.priorities else 1.0
      exp=(state,action,reward,next_state,final)
      self.memory.push_exp(exp,td_error=max_priority)
      #exp=(state,action,reward,next_state,final)
      #self.memory.push_exp(exp,td_error=1.0)

    def backward_update(self, episode_memory):
        if len(episode_memory) < 2:
            return

        reversed_memory = list(reversed(episode_memory))

        for state, action, reward, next_state, final in reversed_memory:
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            next_state_t = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            action_t = torch.tensor([[action]], dtype=torch.long, device=device)
            reward_t = torch.tensor([reward], dtype=torch.float32, device=device)
            final_t = torch.tensor([float(final)], device=device)

            current_q = self.policy_network(state_t).gather(1, action_t)

            with torch.no_grad():
                best_action = self.policy_network(next_state_t).argmax(dim=1,keepdim=True)#double dqn
                next_q = self.target_network(next_state_t).gather(1,best_action).squeeze()

            target = reward_t + GAMMA * next_q * (1 - final_t)  # bellman eq
            loss = self.criterion(current_q, target.unsqueeze(1)).mean()
            self.episode_losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)  # gradient clipping
            self.optimizer.step()

    def step(self):
        if len(self.memory) < 1000:  # minimum replay buffer size
            return

        batch,indices,weights = self.memory.sample_exp(BATCH_SIZE,beta=self.beta)
        # esantionare aleatorie din replay buffer
        states_batch, actions_batch, rewards_batch, next_states_batch, final_batch = zip(*batch)

        states_batch = torch.tensor(np.stack(states_batch), dtype=torch.float32, device=device)
        actions_batch = torch.tensor(actions_batch, dtype=torch.long, device=device).unsqueeze(1)
        rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32, device=device)
        next_states_batch = torch.tensor(np.stack(next_states_batch), dtype=torch.float32, device=device)
        finals_batch = torch.tensor(final_batch, dtype=torch.float32, device=device)

        current_q_values = self.policy_network(states_batch).gather(1, actions_batch)
        with torch.no_grad():
            best_actions=self.policy_network(next_states_batch).argmax(dim=1,keepdim=True)#double dqn
            next_q_values = self.target_network(next_states_batch).gather(1, best_actions).squeeze()

        target = rewards_batch + (GAMMA * next_q_values * (1 - finals_batch))  # bellman eq

        td_errors=(target-current_q_values.squeeze()).detach().cpu().numpy()
        element_wise_loss=self.criterion(current_q_values,target.unsqueeze(1)).squeeze()
        loss=(weights*element_wise_loss).mean()
        self.episode_losses.append(loss.item())

        self.optimizer.zero_grad()  # backprop
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)

        self.optimizer.step()
        self.memory.update_priorities(indices,td_errors)

    def update_epsilon(self):
        self.epsilon = max(FINAL_EPSILON, self.epsilon * EPSILON_DECAY)

    def update_beta(self,episode,total_episodes):
        self.beta=PRB_BETA_START+(PRB_BETA_END - PRB_BETA_START)*(episode/total_episodes)

    def get_average_loss(self):
        if len(self.episode_losses) > 0:
            avg = np.mean(self.episode_losses)
            self.episode_losses = []
            return avg
        return 0.0
def eval_policy(agent, episodes=10):
    old_eps = agent.epsilon
    agent.epsilon = 0.0
    env = gym.make("FlappyBird-v0", render_mode=None, audio_on=False)

    scores = []
    for _ in range(episodes):
        obs, _ = env.reset()
        frame = preprocess_image(obs)

        state = torch.tensor(
            np.stack([frame for _ in range(4)], axis=0),
            dtype=torch.float32
        ).unsqueeze(0).to(device)

        total = 0.0
        while True:
            with torch.no_grad():
                action = agent.sample_action(state)

            obs, reward, finished, truncated, _ = env.step(action)
            total += float(reward)
            final = finished or truncated

            next_frame = preprocess_image(obs)

            next_frame_t = torch.tensor(next_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            state = torch.cat((state[:, 1:, :, :], next_frame_t), dim=1)

            if final:
                break

        scores.append(total)

    env.close()
    agent.epsilon = old_eps
    return float(np.mean(scores)), float(np.std(scores)), float(np.max(scores))


def train_flappy_bird():
    env = gym.make("FlappyBird-v0", render_mode=None, audio_on=False)
    agent = FlappyBirdAgent()
    best_score = 0
    scores_window = deque(maxlen=100)  # pentru media ultimelor 100 episoade

    with open(log_file, 'w') as f:
        f.write(
            f"eps decay:{EPSILON_DECAY}-final eps:{FINAL_EPSILON}-initial eps:{INITIAL_EPSILON}"
            f"-replay memory buffer:{REPLAY_MEMORY_BUFFER}-episodes:{EPISODE_NUMBER}"
            #f"-random_flap_prob:{RANDOM_FLAP_PROB}\n"
        )

        for episode in range(EPISODE_NUMBER):
            obs, _ = env.reset()
            processed_frame = preprocess_image(obs)
            state = torch.tensor(np.stack([processed_frame for _ in range(4)], axis=0), dtype=torch.float32).unsqueeze(0)
            total_reward = 0
            steps = 0
            episode_memory = []

            while True:
                steps += 1
                action = agent.sample_action(state)
                obs, reward, finished, truncated, _ = env.step(action)
                final = finished or truncated

                #if reward >= 1.0:  # trece pipe#scos pentru nu ajuta, inrautatea
                    #shaped_reward = 1.0
                #elif final:  # moare
                    #shaped_reward = -1.0
                #else:
                  #shaped_reward = 0.1  # supravietuire

                total_reward += reward

                processed_next_frame = preprocess_image(obs)
                next_frame_t = torch.tensor(processed_next_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                next_state = torch.cat((state[:, 1:, :, :], next_frame_t), dim=1)

                state_np = state.squeeze(0).cpu().numpy()
                next_state_np = next_state.squeeze(0).cpu().numpy()

                episode_memory.append((state_np, action, reward, next_state_np, final))
                agent.add_exp(state_np, action, reward, next_state_np, final)

                state = next_state

                if final:
                    break


            # batch training (mărit de la 5 la 10 pentru a compensa lipsa backward_update)
            for _ in range(15):
                agent.step()

            scores_window.append(total_reward)
            avg_score = np.mean(scores_window)

            if total_reward > best_score:
                best_score = total_reward

            agent.update_epsilon()
            agent.update_beta(episode,EPISODE_NUMBER)

            if episode > 0 and episode % TARGET_UPDATE_FREQUENCY == 0:
                agent.target_network.load_state_dict(agent.policy_network.state_dict())

            loss = agent.get_average_loss()
            log = f"{episode:4d} | score:{total_reward:6.1f} | avg100:{avg_score:6.2f} | best:{best_score:6.1f} | loss:{loss:.4f} | steps:{steps:4d} | eps:{agent.epsilon:.4f}"
            print(log)
            f.write(log + "\n")
            f.flush()

            if episode > 0 and episode % 200 == 0:
              eval_avg, eval_std, eval_best = eval_policy(agent, episodes=10)
              elog = f"      [EVAL eps=0] avg:{eval_avg:6.2f} | std:{eval_std:6.2f} | best:{eval_best:6.1f}"
              print(elog)
              f.write(elog + "\n")
              f.flush()


            if episode % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


    torch.save(agent.policy_network.state_dict(), "flappy_dqn.pt")
    env.close()

def test_flappy_bird(model_path="flappy_dqn.pt", episodes=20, render=True):
    env = gym.make("FlappyBird-v0", render_mode="human" if render else None, audio_on=False)

    net = DeepQNetwork(84, 84, 2).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    scores = []
    steps_list = []

    for ep in range(episodes):
        obs, _ = env.reset()
        frame = preprocess_image(obs)
        state = torch.tensor(np.stack([frame for _ in range(4)], axis=0), dtype=torch.float32).unsqueeze(0).to(device)

        total_reward = 0.0
        steps = 0

        while True:
            steps += 1

            with torch.no_grad():
                action = net(state).argmax(dim=1).item()

            obs, reward, finished, truncated, _ = env.step(action)
            final = finished or truncated
            total_reward += float(reward)

            next_frame = preprocess_image(obs)
            next_frame_t = torch.tensor(next_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            state = torch.cat((state[:, 1:, :, :], next_frame_t), dim=1)

            if final:
                break

        scores.append(total_reward)
        steps_list.append(steps)
        print(f"[TEST] ep {ep+1:02d} | score={total_reward:.1f} | steps={steps}")

    env.close()

    print("\n=== TEST SUMMARY ===")
    print(f"episodes: {episodes}")
    print(f"avg score: {np.mean(scores):.2f}")
    print(f"std score: {np.std(scores):.2f}")
    print(f"best score: {np.max(scores):.1f}")
    print(f"avg steps: {np.mean(steps_list):.1f}")



#if __name__ == "__main__":
    #train_flappy_bird()
    #test_flappy_bird("flappy_dqn.pt", episodes=100, render=False)
