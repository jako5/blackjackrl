from Blackjack import *
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs
from torchrl.data import BoundedTensorSpec, CompositeSpec,BinaryDiscreteTensorSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
from tensordict import TensorDict, TensorDictBase
import torch


class BlackjackEnv(EnvBase):
    def __init__(self, device="cpu"): # TBD seed
        super().__init__(device=device) # TBD batching
        self.agent = Blackjack(DECKS=4,
                               LOG=False,
                               PLOT=False,
                               DRAW_STRATEGY="Manual",
                               BET_STRATEGY="Constant",
                               THRESHOLD=0,
                               MAXROUNDS=np.inf)
        self._make_spec()

        self.actionmap = ["h", "s", "d"] # TODO split (v)

    
    def _reset(self,batch_size=None):

        #print("RESET")

        while True:
            self.agent.shuffle_cardbank()
            self.agent.prepare_game()
            instant_win = self.agent.react_to_roundstart()
            if not instant_win:
                return TensorDict({"observation":self.agent.get_state(),
                                   "done": self.agent.get_done()})
    
    
    # Apply action and return next_state, reward, done, info
    def _step(self, tensordict):
        
        #print("STEP")

        # Execute player action, adjust state
        #print("ACTION:", tensordict["action"])
        self.agent.react_to_action(self.actionmap[tensordict["action"]])

        ### --- Round ends here --- ###

        # If Player has not finished the round, initiate new round
        if not self.agent.get_done():
            self.agent.react_to_roundstart()

        return TensorDict({"observation":self.agent.get_state(),
                           #"action": tensordict["action"],
                           "reward": self.agent.get_reward(),
                           "done": self.agent.get_done()})
    
    def _make_spec(self):
        # Define the shape and type of observations that the agent receives from the environment.
        self.observation_spec = CompositeSpec({
            "observation": CompositeSpec(
                playerhandval = DiscreteTensorSpec(31, shape=(1,)), # possible states: 0-30 (20 + Draw King)
                dealerhandval = DiscreteTensorSpec(31, shape=(1,)), # possible states: 0-30
                playerace  = DiscreteTensorSpec(3, shape=(1,)), # possible states: Number of Aces
                playerpair = DiscreteTensorSpec(2, shape=(1,)), # possible states: False, True
            )}
        )

        # Define the shape and type of actions
        self.action_spec = DiscreteTensorSpec(3, shape=(1,))

        # Define the shape and type of rewards that the agent receives from the environment
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1))

        # Define the shape and type of the done space
        #self.done_spec = BinaryDiscreteTensorSpec(1, shape=(1,)) #
        self.done_spec = DiscreteTensorSpec(2, dtype=torch.bool, shape=(1,))

        #print(self.observation_spec)

    
    def _set_seed(self, seed: int | None):
        rng = torch.manual_seed(seed)
        self.rng = rng


if __name__ == '__main__':

    env = BlackjackEnv()
    #obspecrand = env.observation_spec.rand()

    #print(obspecrand[("observation", "playerhandval")],obspecrand[("observation", "dealerhandval")],obspecrand[("observation", "playerace")],obspecrand[("observation", "playerpair")])

    # print("\nFAKE TENSOR DICT:\n")
    # print(env.fake_tensordict())

    # print("\nREAL TENSOR DICT:\n")
    # ro = env.rollout(3)
    # print(ro)

    for _ in range(10):
        check_env_specs(env)

    #print(env.done_spec)

    #ro = env.rollout(150)
    # print(ro)