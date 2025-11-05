# How to evaluate a trained agent.


In total we want to evaluate 3 types of agents.
- MPC+RL
- MPC
-- MPC with 'oracle' information 
-- MPC with 'realistic' infomation
- Greedy

## Evaluating MPC+RL

After training an RL agent, you can run the `eval_sac_mpc.py` script. 

`python eval_sac_mpc.py --model_folder 'SAC1006'`


## Evaluating greedy

For this we need the `eval_greedy_baseline.py`

Make sure that the arguments are correct. 

## Evaluating with MPC

For this we need the `eval_mpc_baselines.py`

There are the following 'agents' all located in the `mpc_baseline_agents.py` file:
- oracle
- front_turbine
- simple_estimator

To use call:
`python eval_mpc_baselines --agent_type oracle`