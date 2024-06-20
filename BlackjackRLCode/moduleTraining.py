from torchrl.data import DiscreteTensorSpec, CompositeSpec
from torchrl.envs import CatTensors, DTypeCastTransform, StepCounter
from tensordict.nn import TensorDictModule
import torch
from BlackjackEnv import BlackjackEnv
from matplotlib import pyplot as plt
import time
from torchrl.modules import EGreedyModule, MLP, QValueModule
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer


# spec = CompositeSpec({("observation", "a"): DiscreteTensorSpec(21, shape=(1,)), ("observation", "b"): DiscreteTensorSpec(21, shape=(1,))})
# td = spec.rand()

env = BlackjackEnv()

t = CatTensors(in_keys=[("observation", "playerhandval"), ("observation", "dealerhandval"), ("observation", "playerace"),("observation", "playerpair")], out_key=("observation", "aggregate"))

t_dtype = DTypeCastTransform(dtype_in=torch.long, dtype_out=torch.float32)

t_step = StepCounter(max_steps=10)

module = TensorDictModule(torch.nn.LazyLinear(3), in_keys=[("observation", "aggregate")], out_keys=["action_value"])

env = env.append_transform(t).append_transform(t_dtype).append_transform(t_step)

# Manual test
# spec = env.observation_spec
# infer = module(spec.rand())
# m = torch.nn.Softmax(dim=0)
# probs = m(infer["action_value"])
# print(probs)

policy = Seq(module, QValueModule(spec=env.action_spec))

exploration_module = EGreedyModule(
    env.action_spec, annealing_num_steps=100_000, eps_init=0.5
)
policy_explore = Seq(policy, exploration_module)

init_rand_steps = 5000
frames_per_batch = 100
optim_steps = 10
collector = SyncDataCollector(
    env,
    policy,
    frames_per_batch=frames_per_batch,
    total_frames=-1,
    #init_random_frames=init_rand_steps,
)

rb = ReplayBuffer(storage=LazyTensorStorage(100_000))
from torch.optim import Adam
from torchrl.objectives import DQNLoss, SoftUpdate

loss = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True)
optim = Adam(loss.parameters(), lr=0.02)
updater = SoftUpdate(loss, eps=0.99)

# training loop

total_count = 0
total_episodes = 0
t0 = time.time()

for i, data in enumerate(collector):
    # Write data in replay buffer
    rb.extend(data)
    max_length = rb[:]["next", "step_count"].max()
    if len(rb) > init_rand_steps:
        print(total_episodes)
        # Optim loop (we do several optim steps
        # per batch collected for efficiency)
        for _ in range(optim_steps):
            
            sample = rb.sample(128)
            loss_vals = loss(sample)
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad()

            #print(sample["next"]["reward"].sum())
            
            # Update exploration factor
            exploration_module.step(data.numel())
            
            # Update target params
            updater.step()
            total_count += data.numel()
            total_episodes += data["next", "done"].sum()



