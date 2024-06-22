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
from torch.optim import Adam
from torchrl.objectives import DQNLoss, SoftUpdate
from RLDecisionTable import compareDecisionTables
from Blackjack import load_decision_table


# Init environment
env = BlackjackEnv()
t = CatTensors(in_keys=[("observation", "playerhandval"), ("observation", "dealerhandval"), ("observation", "playerace"),("observation", "playerpair")], out_key=("observation", "aggregate"))
t_dtype = DTypeCastTransform(dtype_in=torch.long, dtype_out=torch.float32)
t_step = StepCounter(max_steps=10)
env = env.append_transform(t).append_transform(t_dtype).append_transform(t_step)

# Init module
module = TensorDictModule(torch.nn.LazyLinear(3), in_keys=[("observation", "aggregate")], out_keys=["action_value"])
#module = TensorDictModule(MLP(in_features=4,out_features=3,depth=1), in_keys=[("observation", "aggregate")], out_keys=["action_value"])


# Init policy
policy = Seq(module, QValueModule(spec=env.action_spec))
exploration_module = EGreedyModule(env.action_spec,annealing_num_steps=100_000, eps_init=0.9)
policy_explore = Seq(policy, exploration_module)


# Init collector
init_rand_steps = 10000
frames_per_batch = 100
optim_steps = 10
collector = SyncDataCollector(
    env,
    policy,
    frames_per_batch=frames_per_batch,
    total_frames=-1,
    #init_random_frames=init_rand_steps,
)

# Init replay buffer
rb = ReplayBuffer(storage=LazyTensorStorage(100_000))
loss = DQNLoss(value_network=policy, action_space=env.action_spec, delay_value=True)
optim = Adam(loss.parameters(), lr=0.02)
updater = SoftUpdate(loss, eps=0.99)

# training loop
total_count = 0
total_episodes = 0
t0 = time.time()

# ground truth decision table
dt = load_decision_table("./data/decisiontable_singles.csv")

lastacc = 0
for i, data in enumerate(collector):
    # Write data in replay buffer
    rb.extend(data)
    print(len(rb))
    max_length = rb[:]["next", "step_count"].max()
    if len(rb) > init_rand_steps:
        # Optim loop (we do several optim steps
        # per batch collected for efficiency)
        for _ in range(optim_steps):
            
            sample = rb.sample(64)
            loss_vals = loss(sample)
            loss_vals["loss"].backward()
            optim.step()
            optim.zero_grad()
            
            # Update exploration factor
            exploration_module.step(data.numel())
            
            # Update target params
            updater.step()
            total_count += data.numel()
            total_episodes += data["next", "done"].sum()

    if len(rb)>20000:
        break

equal, unequal = compareDecisionTables(dt,env,policy,plot=True)

print("Equal: ",equal)
print("Unequal: ",unequal)
