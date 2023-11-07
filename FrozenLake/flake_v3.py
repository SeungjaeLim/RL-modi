import gym
import numpy as np
import time, pickle, os
import parameters as params
from q_updater import Qlearning_updater as Q_up




class trainer():
    def __init__(self, params):
        self.size_idx = params["size_idx"]
        self.map_name = params["map_name"]
        self.epsilon = params["epsilon"]
        self.epochs = params["epochs"]
        self.max_steps = params["max_steps"]
        self.lr = params["lr"]
        self.gamma = params["gamma"]

        self.Train_render = params["Train_render"]
        self.Test_Visrender = params["Test_Visrender"]
        self.save_result = params["save_result"]

        self.env = gym.make(params["env"], is_slippery = params["is_slippery"], 
            map_name = self.map_name)

    def train(self): 
        size = self.size_idx[self.map_name]

        l_idx = ['left', 'down', 'right', 'up']
        print("TRAINING...")
        updater = Q_up(self.env, self.epsilon, self.gamma, self.lr)

        for episode in range(self.epochs):
            state = self.env.reset()
            t = 0
            reward_sum = 0
            while t < self.max_steps:
                if self.Train_render:
                    self.env.render()
                action = updater.action_strategy(state)
                state2, reward, done, info = self.env.step(action)  
                Q = updater.Bellman_Q_updater(state, state2, reward, action)
                state = state2
                t += 1
                reward_sum += reward
                if done:
                    break
            
            # if episode != 0 and episode % 1000 == 0:
            # if episode != 0:

            #     print("EPISODE: %d, CURRENT_REWARD: %d" % (episode + 1, reward_sum))

        print("TRAINING DONE.")
        res_path = np.argmax(Q, axis = 1)
        res_wpath = np.array([l_idx[x] for x in res_path]).reshape(-size, size)
        
        print("Q-tables: ")
        print(Q)

        if self.Test_Visrender:
            dataDic = self.env.our_render()
            import griddrawer as gd

            rend = gd.OUR_RENDERER(dataDic, size)
            rend.renderer(res_wpath)

        if self.save_result:
            with open("FrozenLake" + self.map_name + "_" + str(self.epochs) + "iters.pkl", 'wb') as f:
                pickle.dump(Q, f)


if __name__ == "__main__":
    params = params.PARAMS
    if params["Test_Visrender"]:
        try:
            from pip import main as pipmain
        except:
            from pip._internal import main as pipmain
        def install(package):
            pipmain(['install', package])
        install("pygame")
    trainer = trainer(params)
    trainer.train()