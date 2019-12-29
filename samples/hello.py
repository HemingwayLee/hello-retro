import numpy as np
import retro

def main():
    print(retro.data.list_games())

    env = retro.make(game='Airstriker-Genesis')
    print(f"The size of our frame is: {env.observation_space}")
    print(f"The action size is : {env.action_space.n}")

    possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())
    print(f"possible actions:\n{possible_actions}")

    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            break

    env.close()


if __name__ == "__main__":
    main()

