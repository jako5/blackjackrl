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
import pandas as pd
from copy import deepcopy

def compare_dataframes(df1, df2):
    # Ensure the DataFrames have the same shape
    if df1.shape != df2.shape:
        raise ValueError("DataFrames do not have the same shape.")
    
    # Initialize an empty DataFrame with the same shape
    result_df = pd.DataFrame('', index=df1.index, columns=df1.columns)

    equal, unequal = 0, 0
    
    # Iterate over each cell to compare values
    for col in df1.columns:
        for row in df1.index:
            if df1.at[row, col] != df2.at[row, col]:
                result_df.at[row, col] = 'x'
                unequal += 1
            else:
                equal += 1
    
    return result_df, equal, unequal

def compareDecisionTables(dt,env,module,plot=False):

    dt_gt = dt
    dt_rl = deepcopy(dt_gt)

    for playerhand in dt_rl.keys():
        for dealerhand in dt_rl[playerhand].keys():
            for playerace in [0]:
                for playerpair in [0]:
                    playerhandval = int(playerhand)
                    dealerhandval = int(dealerhand if dealerhand.isnumeric() else 10)

                    #print(f"Player Hand Value: {playerhandval}, Dealer Hand Value: {dealerhandval}, Player Ace: {playerace}, Player Pair: {playerpair}")
                    state = env.observation_spec.rand()
                    state[("observation","aggregate")] = torch.tensor([playerhandval,dealerhandval,playerace,playerpair],dtype=torch.float32)
                    infer = module(state)
                    m = torch.nn.Softmax(dim=0)
                    probs = m(infer["action_value"])
                    
                    dt_rl[playerhand][dealerhand] = env.actionmap[probs.argmax()]
    
    dt_comp, equal, unequal = compare_dataframes(pd.DataFrame(dt_gt).T,pd.DataFrame(dt_rl).T)
    if plot:
        print("\nGROUND TRUTH:\n",pd.DataFrame(dt_gt).T)
        print("\nRL DECISION TABLE:\n",pd.DataFrame(dt_rl).T)
        print("\nCOMPARISON:\n",dt_comp)

    # export to csv
    acc = equal/(equal+unequal)
    pd.DataFrame(dt_rl).T.to_csv(f"data/tblout/{acc:.2f}_dt_rl.csv")

    return equal, unequal
