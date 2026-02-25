import os, time, json, datetime
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import flappy_bird_gymnasium
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from train import DeepQNetwork, preprocess_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_SEED = 42
np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(BASE_SEED)

net = DeepQNetwork(84, 84, 2).to(device)
net.load_state_dict(torch.load("flappy_dqn.pt", map_location=device))
net.eval()

os.makedirs("videos", exist_ok=True)
os.makedirs("test_results", exist_ok=True)
log_path = f"test_results/test_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

env = gym.make("FlappyBird-v0", render_mode="rgb_array", audio_on=False)

best = -1e9
best_seed = None
best_ep = None
scores = []
seeds_used = []

NUM_EPISODES = 500

print(f"Testing model on {NUM_EPISODES} episodes...")
print(f"Device: {device}")
print(f"Base seed: {BASE_SEED}\n")


for ep in tqdm(range(NUM_EPISODES), desc="Testing episodes"):
    seed = BASE_SEED + ep 
    seeds_used.append(seed)
    obs, _ = env.reset(seed=seed)

    frame = preprocess_image(obs)
    state = torch.tensor(np.stack([frame]*4, axis=0), dtype=torch.float32).unsqueeze(0).to(device)

    total = 0.0
    steps = 0
    
    while True:
        steps += 1
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
    rec = {"ts": ts, "ep": ep + 1, "seed": seed, "score": float(total), "steps": steps}
    with open(log_path, "a") as f:
        f.write(json.dumps(rec) + "\n")

    if total > best:
        best = total
        best_seed = seed
        best_ep = ep + 1

env.close()

assert len(set(seeds_used)) == NUM_EPISODES, "ERROR: Duplicate seeds detected!"


avg = float(np.mean(scores))
std = float(np.std(scores))
median = float(np.median(scores))
min_score = float(np.min(scores))
max_score = float(np.max(scores))

p25 = float(np.percentile(scores, 25))
p50 = float(np.percentile(scores, 50))
p75 = float(np.percentile(scores, 75))
p90 = float(np.percentile(scores, 90))
p95 = float(np.percentile(scores, 95))

print("\n" + "="*60)
print("TESTING RESULTS")
print("="*60)
print(f"Episodes:     {len(scores)}")
print(f"Mean:         {avg:.2f}")
print(f"Median:       {median:.2f}")
print(f"Std:          {std:.2f}")
print(f"Min:          {min_score:.1f}")
print(f"Max:          {max_score:.1f}")
print(f"\nPercentiles:")
print(f"  25th:       {p25:.1f}")
print(f"  50th:       {p50:.1f}")
print(f"  75th:       {p75:.1f}")
print(f"  90th:       {p90:.1f}")
print(f"  95th:       {p95:.1f}")
print(f"\nBest Episode: {best_ep}")
print(f"Best Seed:    {best_seed}")
print(f"Best Score:   {best:.1f}")
print("="*60)


summary = {
    "timestamp": datetime.datetime.now().isoformat(),
    "num_episodes": NUM_EPISODES,
    "base_seed": BASE_SEED,
    "statistics": {
        "mean": avg,
        "median": median,
        "std": std,
        "min": min_score,
        "max": max_score,
        "percentiles": {
            "25": p25,
            "50": p50,
            "75": p75,
            "90": p90,
            "95": p95
        }
    },
    "best": {
        "episode": best_ep,
        "seed": best_seed,
        "score": best
    }
}

summary_path = f"test_results/summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nDetailed log saved to: {log_path}")
print(f"Summary saved to: {summary_path}")


print("\nGenerating plots...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))


axes[0, 0].hist(scores, bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].axvline(avg, color='r', linestyle='--', linewidth=2, label=f'Mean: {avg:.1f}')
axes[0, 0].axvline(median, color='g', linestyle='--', linewidth=2, label=f'Median: {median:.1f}')
axes[0, 0].set_xlabel('Score', fontsize=12)
axes[0, 0].set_ylabel('Frequency', fontsize=12)
axes[0, 0].set_title('Score Distribution', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)


axes[0, 1].plot(scores, alpha=0.6, linewidth=1)
axes[0, 1].axhline(avg, color='r', linestyle='--', linewidth=2, label=f'Mean: {avg:.1f}')
axes[0, 1].axhline(median, color='g', linestyle='--', linewidth=2, label=f'Median: {median:.1f}')
axes[0, 1].scatter([best_ep-1], [best], color='gold', s=200, marker='*', 
                   edgecolors='black', linewidths=1.5, zorder=5, label=f'Best: {best:.1f}')
axes[0, 1].set_xlabel('Episode', fontsize=12)
axes[0, 1].set_ylabel('Score', fontsize=12)
axes[0, 1].set_title('Scores Over Episodes', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)


window = 20
moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
axes[1, 0].plot(range(window-1, NUM_EPISODES), moving_avg, linewidth=2, label=f'MA({window})')
axes[1, 0].axhline(avg, color='r', linestyle='--', linewidth=2, label=f'Overall Mean: {avg:.1f}')
axes[1, 0].set_xlabel('Episode', fontsize=12)
axes[1, 0].set_ylabel('Moving Average Score', fontsize=12)
axes[1, 0].set_title(f'Moving Average (window={window})', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)


axes[1, 1].boxplot(scores, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2),
                   whiskerprops=dict(linewidth=1.5),
                   capprops=dict(linewidth=1.5))
axes[1, 1].set_ylabel('Score', fontsize=12)
axes[1, 1].set_title('Score Distribution (Box Plot)', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')


stats_text = f'Mean: {avg:.2f}\nMedian: {median:.2f}\nStd: {std:.2f}\nMin: {min_score:.1f}\nMax: {max_score:.1f}'
axes[1, 1].text(1.15, np.median(scores), stats_text, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plot_path = f"test_results/test_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")
plt.close()


print(f"\n{'='*60}")
print("RECORDING BEST EPISODE")
print(f"{'='*60}")
print(f"Episode: {best_ep}")
print(f"Seed:    {best_seed}")
print(f"Score:   {best:.1f}")

video_env = gym.make("FlappyBird-v0", render_mode="rgb_array", audio_on=False)
video_env = RecordVideo(
    video_env,
    video_folder="videos",
    name_prefix=f"best_ep{best_ep}_score{int(best)}_seed{best_seed}",
    episode_trigger=lambda i: True,
)

obs, _ = video_env.reset(seed=best_seed)
frame = preprocess_image(obs)
state = torch.tensor(np.stack([frame]*4, axis=0), dtype=torch.float32).unsqueeze(0).to(device)

total_rerun = 0.0
steps_rerun = 0

while True:
    steps_rerun += 1
    with torch.no_grad():
        action = net(state).argmax(dim=1).item()

    obs, reward, terminated, truncated, _ = video_env.step(action)
    total_rerun += float(reward)
    done = terminated or truncated

    next_frame = preprocess_image(obs)
    next_frame_t = torch.tensor(next_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    state = torch.cat((state[:, 1:, :, :], next_frame_t), dim=1)

    if done:
        break

video_env.close()

print(f"\nRe-run score:     {total_rerun:.1f}")
print(f"Original score:   {best:.1f}")
print(f"Difference:       {abs(total_rerun - best):.6f}")

if abs(total_rerun - best) < 0.1:
    print("✅ Determinism verified: Re-run matches original")
else:
    print("⚠️  WARNING: Re-run score differs from original!")
    print("   This may indicate non-deterministic behavior in the environment.")

print(f"\nVideo saved to: videos/")
print(f"{'='*60}\n")

print("DETERMINISM CHECK")
print("="*60)
print("Re-running first 10 episodes to verify reproducibility...\n")

test_env = gym.make("FlappyBird-v0", render_mode="rgb_array", audio_on=False)
determinism_ok = True

for ep in range(10):
    seed = BASE_SEED + ep
    obs, _ = test_env.reset(seed=seed)
    
    frame = preprocess_image(obs)
    state = torch.tensor(np.stack([frame]*4, axis=0), dtype=torch.float32).unsqueeze(0).to(device)
    
    total_check = 0.0
    
    while True:
        with torch.no_grad():
            action = net(state).argmax(dim=1).item()
        
        obs, reward, terminated, truncated, _ = test_env.step(action)
        total_check += float(reward)
        done = terminated or truncated
        
        next_frame = preprocess_image(obs)
        next_frame_t = torch.tensor(next_frame, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        state = torch.cat((state[:, 1:, :, :], next_frame_t), dim=1)
        
        if done:
            break
    
    match = abs(total_check - scores[ep]) < 1e-6
    status = "✅" if match else "⚠️ "
    
    if not match:
        determinism_ok = False
    
    print(f"{status} Episode {ep+1:2d} (seed={seed}): "
          f"original={scores[ep]:.1f}, rerun={total_check:.1f}, "
          f"diff={abs(total_check - scores[ep]):.6f}")

test_env.close()

print(f"\n{'='*60}")
if determinism_ok:
    print("✅ ALL CHECKS PASSED: Model is deterministic and reproducible")
else:
    print("⚠️  SOME CHECKS FAILED: Non-deterministic behavior detected")
print(f"{'='*60}\n")

print("Testing complete!")
print(f"\nGenerated files:")
print(f"  - {log_path}")
print(f"  - {summary_path}")
print(f"  - {plot_path}")
print(f"  - videos/best_ep{best_ep}_score{int(best)}_seed{best_seed}_*.mp4")