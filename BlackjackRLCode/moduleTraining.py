from torchrl.data import DiscreteTensorSpec, CompositeSpec
from torchrl.envs import CatTensors, DTypeCastTransform
from tensordict.nn import TensorDictModule
import torch
from BlackjackEnv import BlackjackEnv
from matplotlib import pyplot as plt


# spec = CompositeSpec({("observation", "a"): DiscreteTensorSpec(21, shape=(1,)), ("observation", "b"): DiscreteTensorSpec(21, shape=(1,))})
# td = spec.rand()

env = BlackjackEnv()

t = CatTensors(in_keys=[("observation", "playerhandval"), ("observation", "dealerhandval"), ("observation", "playerace"),("observation", "playerpair")], out_key=("observation", "aggregate"))
t_dtype = DTypeCastTransform(dtype_in=torch.long, dtype_out=torch.float32)
# print(spec.rand())
# print(t(spec.rand()))

module = TensorDictModule(torch.nn.LazyLinear(4), in_keys=[("observation", "aggregate")], out_keys=["action_value"])

env = env.append_transform(t).append_transform(t_dtype)
spec = env.observation_spec
infer = module(spec.rand())

# OLD; WITHOUT TRANSFORMS
# spec = env.observation_spec
# infer = module(t_dtype(t(spec.rand())))


m = torch.nn.Softmax(dim=0)
probs = m(infer["action_value"])
print(probs)
# plt.bar(range(4), probs.detach().numpy())
# plt.show()


from torchrl.modules import EGreedyModule, MLP, QValueModule
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq

policy = Seq(module, QValueModule(spec=env.action_spec))
exploration_module = EGreedyModule(
    env.action_spec, annealing_num_steps=100_000, eps_init=0.5
)
policy_explore = Seq(policy, exploration_module)


from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer

init_rand_steps = 5000
frames_per_batch = 100
optim_steps = 10
collector = SyncDataCollector(
    env,
    policy,
    frames_per_batch=frames_per_batch,
    total_frames=-1,
    init_random_frames=init_rand_steps,
)
rb = ReplayBuffer(storage=LazyTensorStorage(100_000))

from torch.optim import Adam
