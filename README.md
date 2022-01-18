# QuantumReinforcementLearning
Quantum Reinforcement Learning project using VQC's and REINFORCE based on Jerbi et al. (2021)

Library requirements:
TensorFlow 2.4.1
TensorFlow-Quantum 0.5.1

To run:
python PQC_handler [n_vehicles] [n_layers] [n_episodes] [batch_size] [alpha-Theta] [alpha-Lambda] [alpha-Omega] [folds]

Example:
Running with three vehicles, seven layers, 1500 episodes, 10 episodes per batch, LR theta 0.01, LR lamda 0.01, LR omega 0.005, five folds:
python PQC_handler.py 3 7 1500 10 0.01 0.01 0.005 5

