import numpy as np
import retro
from random import randrange

def main():
    # print(retro.data.list_games())

    # env = retro.make(game='Airstriker-Genesis')
    env = retro.make(game='Breakout-Atari2600')
    
    print(f"The size of our frame is: {env.observation_space}")
    print(f"The action size is : {env.action_space.n}")

    possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())
    print(f"possible actions:\n{possible_actions}")

    for act in possible_actions:
        print(env.get_action_meaning(act))

    valid_actions = possible_actions[-2:]
    valid_actions = np.append(valid_actions, [possible_actions[0]], axis=0)
    print(valid_actions)

    obs = env.reset()
    while True:
        action = valid_actions[randrange(valid_actions.shape[0])]
        # action = env.action_space.sample()
        print(action)
        print(env.get_action_meaning(action))
        obs, reword, done, info = env.step(action)
        
        print(reword)
        print(info)
        env.render()
        
        if done:
            break

    env.close()


if __name__ == "__main__":
    main()

