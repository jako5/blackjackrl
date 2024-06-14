from Blackjack import *
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs
from torchrl.data import BoundedTensorSpec, CompositeSpec,BinaryDiscreteTensorSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
from tensordict import TensorDict, TensorDictBase
import torch


class BlackjackEnv(EnvBase):
    def __init__(self, seed=None, device="cpu"): # TBD seed
        super().__init__(device=device) # TBD batching
        self.agent = Blackjack(DECKS=4,
                               LOG=False,
                               PLOT=False,
                               DRAW_STRATEGY="Manual",
                               BET_STRATEGY="Constant",
                               THRESHOLD=0,
                               MAXROUNDS=np.inf)
        self._make_spec()

        self.actionmap = ["hit", "stand", "double", "split"]

    
    def getReturnTensordict(self):
        return TensorDict({**self.agent.get_state(),
                           "reward": self.agent.get_reward(),
                           "done": self.agent.get_done()})
    
    def _reset(self,batch_size=None):
        while True:
            self.agent.shuffle_cardbank()
            self.agent.prepare_game()
            instant_win = self.agent.react_to_roundstart()
            if not instant_win:
                return TensorDict({**self.agent.get_state(),
                           #"reward": self.agent.get_reward(),
                           "done": self.agent.get_done()})
    
    
    # Apply action and return next_state, reward, done, info
    def _step(self, tensordict):

        # Execute player action, adjust state
        self.agent.react_to_action(self.actionmap[tensordict["action"]])

        ### --- Round ends here --- ###

        # If Player has not finished the round, initiate new round
        if not self.agent.get_done():
            self.agent.react_to_roundstart()

        return self.getReturnTensordict()
    
    def _make_spec(self):
        # Define the shape and type of observations that the agent receives from the environment.
        self.observation_spec = CompositeSpec(
            playerhandval = DiscreteTensorSpec(21), # possible states: 0-21
            dealerhandval = DiscreteTensorSpec(21), # possible states: 0-21
            playerace = DiscreteTensorSpec(2), # possible states: False, True
            playerpair = DiscreteTensorSpec(2), # possible states: False, True
        )

        # Define the shape and type of actions that the agent can take in the environment
        self.action_spec = DiscreteTensorSpec(4)

        # Define the shape and type of rewards that the agent receives from the environment
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1,1))

        # Define the shape and type of the info that the agent receives from the environment
        self.done_spec = BinaryDiscreteTensorSpec(1) # 

    
    def _set_seed(self, seed: int | None):
        rng = torch.manual_seed(seed)
        self.rng = rng


    

if __name__ == '__main__':
    env = BlackjackEnv()
    #env.reset()

    check_env_specs(env)
    # state = env.reset()
    # action = 1
    # next_state, reward, done, info = env.step(action)
    # print(state, next_state, reward, done, info)
    # env.close()