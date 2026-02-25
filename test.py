import os, time, json, datetime
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import flappy_bird_gymnasium
import numpy as np
import torch

from train import DeepQNetwork, preprocess_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = DeepQNetwork(84, 84, 2).to(device)
net.load_state_dict(torch.load("flappy_dqn.pt", map_location=device))
net.eval()

os.makedirs("videos", exist_ok=True)
log_path = f"test_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

env = gym.make("FlappyBird-v0", render_mode="rgb_array", audio_on=False)

best = -1e9
best_seed = None
best_ep = None

scores = []  

for ep in range(500):
    seed = int(time.time() * 1000) % (2**31 - 1)
    obs, _ = env.reset(seed=seed)

    frame = preprocess_image(obs)
    state = torch.tensor(np.stack([frame]*4, axis=0), dtype=torch.float32).unsqueeze(0).to(device)

    total = 0.0
    while True:
        with torch.no_grad():
            action = net(state).argmax(dim=1).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        done = terminated or truncated

        next_frame = preprocess_image(obs)
        next_frame_t = torch.tensor(next_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        state = torch.cat((state[:, 1:, :, :], next_frame_t), dim=1)

        if done:
            break

    scores.append(total)  

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    rec = {"ts": ts, "ep": ep + 1, "seed": seed, "score": float(total)}
    with open(log_path, "a") as f:
        f.write(json.dumps(rec) + "\n")

    if total > best:
        best = total
        best_seed = seed
        best_ep = ep + 1

    print(f"ep {ep+1:02d}: score={total:.1f} | seed={seed}")

env.close()

avg = float(np.mean(scores))
std = float(np.std(scores))
print(f"\nAVG over {len(scores)} eps: {avg:.2f} | STD: {std:.2f} | BEST: {best:.1f} (ep={best_ep}, seed={best_seed})")
print(f"Log saved to: {log_path}")

video_env = gym.make("FlappyBird-v0", render_mode="rgb_array", audio_on=False)
video_env = RecordVideo(
    video_env,
    video_folder="videos",
    name_prefix=f"best_ep{best_ep}_score{int(best)}_seed{best_seed}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    episode_trigger=lambda i: True,
)

obs, _ = video_env.reset(seed=best_seed)
frame = preprocess_image(obs)
state = torch.tensor(np.stack([frame]*4, axis=0), dtype=torch.float32).unsqueeze(0).to(device)

total2 = 0.0
while True:
    with torch.no_grad():
        action = net(state).argmax(dim=1).item()

    obs, reward, terminated, truncated, _ = video_env.step(action)
    total2 += float(reward)
    done = terminated or truncated

    next_frame = preprocess_image(obs)
    next_frame_t = torch.tensor(next_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    state = torch.cat((state[:, 1:, :, :], next_frame_t), dim=1)

    if done:
        break

video_env.close()

print(f"Re-run best score={total2:.1f}")
print("Video saved under folder: videos/")
