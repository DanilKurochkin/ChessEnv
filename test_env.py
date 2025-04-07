
from chess_env import ChessEnv
import random

env = ChessEnv()
obs = env.reset()
done = False
step_count = 0

while not done and step_count < 3:
    env.render()
    legal_moves = env.legal_actions()
    if not legal_moves:
        break
    action = random.choice(legal_moves)
    print(f"\n→ Ход: {action}")
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}, Done: {done}, Info: {info}")
    step_count += 1

print("Игра завершена.")
env.render()