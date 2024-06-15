import torch
from BlackjackEnv import BlackjackEnv
from tensordict.nn import TensorDictModule, TensorDictSequential as Seq

# Environment Initialization
env = BlackjackEnv()

# Model Initialization
module = torch.nn.LazyLinear(out_features=1)

# Policy Initialization
policy = TensorDictModule(
    module,
    in_keys=["observation"],
    out_keys=["action"],
)

# Rollout
rollout = env.rollout(max_steps=10, policy=policy) # 