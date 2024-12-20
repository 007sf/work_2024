__credits__ = ["Intelligent Unmanned Systems Laboratory at Westlake University."]
'''
Specify parameters of the env
'''
from typing import Union
import numpy as np
import argparse
import json
parser = argparse.ArgumentParser("Grid World Environment")

## ==================== User settings ===================='''
# specify the number of columns and rows of the grid world
parser.add_argument("--env-size", type=Union[list, tuple, np.ndarray], default=(7,7) )

# specify the start state
parser.add_argument("--start-state", type=dict, default={
    "position":(0,6),
    "marks_left":10,
    "marks_pos":[]
})

# specify the target state
parser.add_argument("--target-state", type=dict, default={
    "position":(6,6),
    "marks_left":0,

})

# sepcify the forbidden states
parser.add_argument("--forbid"
                    "den-states", type=str, default='{"position": [[3,2],[3,3],[3,4],[2,5],[1,6],[5,5],[6,5],[4,3]]}')
#-------------------------------------------------------------------------------------------------------------------
# sepcify the reward when reaching target
parser.add_argument("--reward-target", type=float, default =400)

# sepcify the reward when entering into forbidden area
parser.add_argument("--reward-forbidden", type=float, default = -50)

# sepcify the reward for each step
parser.add_argument("--reward-step", type=float, default = -1)

#做标记的奖励函数
parser.add_argument("--reward-mark",type=float,default=10)
#标记无效的奖励函数
parser.add_argument("--reward_invalid_mark",type=float,default=-50)
#---------------------------------------------------------------------------------------------------------------------
## ==================== End of User settings ====================


## ==================== Advanced Settings ====================
action_map = {0:(0,1),
              1:(1,0),
              2:(0,-1),
              3:(-1,0),
              4:(0,0),
              }
parser.add_argument("--action-space", type=list, default=list(action_map.keys()) )  # down, right, up, left, stay,做标记
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--animation-interval", type=float, default = 0.3)
## ==================== End of Advanced settings ====================


args = parser.parse_args()     
def validate_environment_parameters(env_size, start_state, target_state, forbidden_states):
    if not (isinstance(env_size, tuple) or isinstance(env_size, list) or isinstance(env_size, np.ndarray)) and len(env_size) != 2:
        raise ValueError("Invalid environment size. Expected a tuple (rows, cols) with positive dimensions.")
    
    # for i in range(2):
    #     assert list(start_state["position"])[i] < env_size[i]
    #     assert list(target_state["position"])[i] < env_size[i]
    #     for j in range(len(forbidden_states)):
    #         assert forbidden_states[j][i] < env_size[i]
try:
    forbidden_states = json.loads(args.forbidden_states)
    validate_environment_parameters(args.env_size, args.start_state["position"], args.target_state["position"], forbidden_states["position"])
except ValueError as e:
    print("Error:", e)