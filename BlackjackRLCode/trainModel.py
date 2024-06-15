import torch
import torch.nn as nn
import torch.nn.functional as F
from BlackjackEnv import BlackjackEnv
from tensordict.nn import TensorDictModule, TensorDictSequential as Seq

env = BlackjackEnv()

env.rollout(max_steps=10)

module = torch.nn.LazyLinear(out_features=1)

policy = TensorDictModule(
    module,
    in_keys=["observation"],
    out_keys=["action"],
)

rollout = env.rollout(max_steps=10, policy=policy) # 