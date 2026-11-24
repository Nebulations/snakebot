from collections import deque
from game import SnakeGame
from agent import Agent

TRAINING = False
agent = Agent(training=TRAINING, model_file="model.pth")
game = SnakeGame(render=(not TRAINING), fps=10, grid=(20, 16, 24, 4))

best_score = 0
total_score = 0

last_50_games = deque(maxlen=50)

while True:
    state = game.get_state()
    action = agent.get_action(state)
    reward, score, done = game.step(action)

    if TRAINING:
        next_state = game.get_state()

        agent.train_short(state, action, reward, next_state, done)
        agent.remember(state, action, reward, next_state, done)

        if done:
            if best_score < score:
                best_score = score
                agent.model.save()

            last_50_games.append(score)

            game.reset()
            agent.games_played+=1
            agent.train_long()
            agent.update_epsilon()
            print(f"Game #{agent.games_played}. Score: {score}, Best: {best_score}, Average: {round(sum(last_50_games)/50, 2)}")
    elif done:
        game.reset()
        agent.games_played+=1
        print(f"Game #{agent.games_played}. Score: {score}, Best: {best_score}")
