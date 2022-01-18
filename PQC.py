
"""Now import TensorFlow and the module dependencies:"""

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_quantum as tfq

import gym, cirq, sympy, highway_env
import numpy as np
from functools import reduce
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit
import sys
import os

from scipy.stats import entropy
tf.get_logger().setLevel('ERROR')

"""## 1. Build a PQC with data re-uploading

At the core of both RL algorithms you are implementing is a PQC that takes as input the agent's state $s$ in the environment (i.e., a numpy array) and outputs a vector of expectation values. These expectation values are then post-processed, either to produce an agent's policy $\pi(a|s)$ or approximate Q-values $Q(s,a)$. In this way, the PQCs are playing an analog role to that of deep neural networks in modern deep RL algorithms.

A popular way to encode an input vector in a PQC is through the use of single-qubit rotations, where rotation angles are controlled by the components of this input vector. In order to get a [highly-expressive model](https://arxiv.org/abs/2008.08605), these single-qubit encodings are not performed only once in the PQC, but in several "[re-uploadings](https://quantum-journal.org/papers/q-2020-02-06-226/)", interlayed with variational gates. The layout of such a PQC is depicted below:

<img src="https://github.com/tensorflow/quantum/blob/master/docs/tutorials/images/pqc_re-uploading.png?raw=1" width="700">

As discussed in [[1]](https://arxiv.org/abs/2103.05577) and [[2]](https://arxiv.org/abs/2103.15084), a way to further enhance the expressivity and trainability of data re-uploading PQCs is to use trainable input-scaling parameters $\boldsymbol{\lambda}$ for each encoding gate of the PQC, and trainable observable weights $\boldsymbol{w}$ at its output.

### 1.1 Cirq circuit for ControlledPQC

The first step is to implement in Cirq the quantum circuit to be used as the PQC. For this, start by defining basic unitaries to be applied in the circuits, namely an arbitrary single-qubit rotation and an entangling layer of CZ gates:
"""

def one_qubit_rotation(qubit, symbols):
    """
    Returns Cirq gates that apply a rotation of the bloch sphere about the X,
    Y and Z axis, specified by the values in `symbols`.
    """
    return [cirq.rx(symbols[0])(qubit),
            cirq.ry(symbols[1])(qubit),
            cirq.rz(symbols[2])(qubit)]

def entangling_layer(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops

"""Now, use these functions to generate the Cirq circuit:"""
def generate_circuit(qubits, n_layers, features):
    """Prepares a data re-uploading circuit on `qubits` with `n_layers` layers."""
    # Number of qubits
    n_qubits = len(qubits)
    n_vec = int(features/n_qubits)
    # Sympy symbols for variational angles Theta
    params = sympy.symbols(f'theta(0:{3*(n_layers+1)*n_qubits})')
    params = np.asarray(params).reshape((n_layers + 1, n_qubits, 3))

    # Sympy symbols for encoding angles Lambda
    #inputs = sympy.symbols(f'x(0:{n_qubits})'+f'(0:{n_layers})')
    #inputs = np.asarray(inputs).reshape((n_layers, n_qubits))

    #inputs = sympy.symbols(f'x(0:{features})')
    #inputs = np.asarray(inputs).reshape((int(features / n_qubits), n_qubits))
    inputs = sympy.symbols(f'x(0:{n_layers})'+f'_(0:{n_qubits*n_vec})')
    inputs = np.asarray(inputs).reshape((n_layers, n_qubits*n_vec))

    # Define circuit
    circuit = cirq.Circuit()
    for l in range(n_layers):
        # Variational layer
        circuit += cirq.Circuit(one_qubit_rotation(q, params[l, i]) for i, q in enumerate(qubits))
        circuit += entangling_layer(qubits)
        # Encoding layer
        #for shift in range(int(features / n_qubits)):
        #    circuit += cirq.Circuit(cirq.rx(inputs[shift, i])(q) for i, q in enumerate(qubits))
        for shift in range(n_vec):
            circuit += cirq.Circuit(cirq.rx(inputs[l, i+(shift*n_qubits)])(q) for i, q in enumerate(qubits))

    # Last varitional layer
    circuit += cirq.Circuit(one_qubit_rotation(q, params[n_layers, i]) for i,q in enumerate(qubits))

    return circuit, list(params.flat), list(inputs.flat)


"""### 1.2 ReUploadingPQC layer using ControlledPQC

To construct the re-uploading PQC from the figure above, you can create a custom Keras layer. This layer will manage the trainable parameters (variational angles $\boldsymbol{\theta}$ and input-scaling parameters $\boldsymbol{\lambda}$) and resolve the input values (input state $s$) into the appropriate symbols in the circuit.
"""

class ReUploadingPQC(tf.keras.layers.Layer):
    """
    Performs the transformation (s_1, ..., s_d) -> (theta_1, ..., theta_N, lmbd[1][1]s_1, ..., lmbd[1][M]s_1,
        ......., lmbd[d][1]s_d, ..., lmbd[d][M]s_d) for d=input_dim, N=theta_dim and M=n_layers.
    An activation function from tf.keras.activations, specified by `activation` ('linear' by default) is
        then applied to all lmbd[i][j]s_i.
    All angles are finally permuted to follow the alphabetical order of their symbol names, as processed
        by the ControlledPQC.
    """

    def __init__(self, n_features, qubits, n_layers, observables, activation="linear", name="re-uploading_PQC"):
        super(ReUploadingPQC, self).__init__(name=name)
        circuit, theta_symbols, input_symbols = generate_circuit(qubits, n_layers, n_features)
        self.n_layers = n_layers
        self.n_qubits = len(qubits)
        self.n_features = n_features
        self.n_thetas = len(theta_symbols)
        self.init_parameters()

        # Define explicit symbol order.
        symbols = [str(symb) for symb in theta_symbols + input_symbols]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)]) #Sofienes new code

        self.activation = activation
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)

    def call(self, inputs):
        # inputs[0] = encoding data for the state.
        batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1])
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.n_layers])
        scaled_inputs = tf.einsum("i,ji->ji", self.lmbd, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        joined_vars = tf.concat([tiled_up_thetas, squashed_inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)

        return self.computation_layer([tiled_up_circuits, joined_vars])

    """Set Model Learnable Parameters"""
    def init_parameters(self):
        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, self.n_thetas), dtype="float32"),
            trainable=True, name="thetas"
        )

        lmbd_init = tf.ones(shape=(self.n_features * self.n_layers,))
        self.lmbd = tf.Variable(
            initial_value=lmbd_init, dtype="float32", trainable=True, name="lambdas"
        )

"""## 2. Policy-gradient RL with PQC policies

In this section, you will implement the policy-gradient algorithm presented in <a href="https://arxiv.org/abs/2103.05577" class="external">[1]</a>. For this, you will start by constructing, out of the PQC that was just defined, the `softmax-VQC` policy (where VQC stands for variational quantum circuit):
$$ \pi_\theta(a|s) = \frac{e^{\beta \langle O_a \rangle_{s,\theta}}}{\sum_{a'} e^{\beta \langle O_{a'} \rangle_{s,\theta}}} $$
where $\langle O_a \rangle_{s,\theta}$ are expectation values of observables $O_a$ (one per action) measured at the output of the PQC, and $\beta$ is a tunable inverse-temperature parameter.

You can adopt the same observables used in <a href="https://arxiv.org/abs/2103.05577" class="external">[1]</a> for CartPole, namely a global $Z_0Z_1Z_2Z_3$ Pauli product acting on all qubits, weighted by an action-specific weight for each action. To implement the weighting of the Pauli product, you can use an extra `tf.keras.layers.Layer` that stores the action-specific weights and applies them multiplicatively on the expectation value $\langle Z_0Z_1Z_2Z_3 \rangle_{s,\theta}$.
"""
#Replacement suggested for Alternating by Sofiene
class Rescaling(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(Rescaling, self).__init__()
        self.input_dim = input_dim
        self.w = tf.Variable(
            initial_value=tf.ones(shape=(1,input_dim)), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.math.multiply((inputs+1)/2, tf.repeat(self.w,repeats=tf.shape(inputs)[0],axis=0))

"""class Alternating(tf.keras.layers.Layer):
    def __init__(self, output_dim):
        super(Alternating, self).__init__()
        self.w = tf.Variable(
            initial_value=tf.constant([[1. for i in range(output_dim)]]), dtype="float32",
            trainable=True, name="obs-weights")

    def call(self, inputs):
        return tf.matmul(inputs, self.w)
"""

"""Define a `tf.keras.Model` that applies, sequentially, the `ReUploadingPQC` layer previously defined, followed by a post-processing layer that computes the weighted observables using `Alternating`, which are then fed into a `tf.keras.layers.Softmax` layer that outputs the `softmax-VQC` policy of the agent."""

def generate_model_policy(features, qubits, n_layers, n_actions, beta, observables):
    """Generates a Keras model for a data re-uploading PQC policy."""

    input_tensor = tf.keras.Input(shape=(features, ), dtype=tf.dtypes.float32, name='input')
    re_uploading_pqc = ReUploadingPQC(features, qubits, n_layers, observables)([input_tensor])
    process = tf.keras.Sequential([
        #Alternating(n_actions),
        Rescaling(n_actions),
        tf.keras.layers.Lambda(lambda x: x * beta),
        tf.keras.layers.Softmax()
    ], name="observables-policy")
    policy = process(re_uploading_pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=policy)

    return model

"""You can now train the PQC policy on CartPole-v1, using, e.g., the basic `REINFORCE` algorithm (see Alg. 1 in <a href="https://arxiv.org/abs/2103.05577" class="external">[1]</a>). Pay attention to the following points:
1. Because scaling parameters, variational angles and observables weights are trained with different learning rates, it is convenient to define 3 separate optimizers with their own learning rates, each updating one of these groups of parameters.
2. The loss function in policy-gradient RL is
    $$ \mathcal{L}(\theta) = -\frac{1}{|\mathcal{B}|}\sum_{s_0,a_0,r_1,s_1,a_1, \ldots \in \mathcal{B}} \left(\sum_{t=0}^{H-1} \log(\pi_\theta(a_t|s_t)) \sum_{t'=1}^{H-t} \gamma^{t'} r_{t+t'} \right)$$
for a batch $\mathcal{B}$ of episodes $(s_0,a_0,r_1,s_1,a_1, \ldots)$ of interactions in the environment following the policy $\pi_\theta$. This is different from a supervised learning loss with fixed target values that the model should fit, which make it impossible to use a simple function call like `model.fit` to train the policy. Instead, using a `tf.GradientTape` allows to keep track of the computations involving the PQC (i.e., policy sampling) and store their contributions to the loss during the interaction. After running a batch of episodes, you can then apply backpropagation on these computations to get the gradients of the loss with respect to the PQC parameters and use the optimizers to update the policy-model.

Start by defining a function that gathers episodes of interaction with the environment:
"""

def gather_episodes(state_bounds, n_actions, model, n_episodes, env_info):
    """Interact with environment in batched fashion."""

    trajectories = [defaultdict(list) for _ in range(n_episodes)]
    envs = [gym.make(env_info[0]) for _ in range(n_episodes)]
    for e in envs:
        e.config["observation"] = env_info[1]
        #e.seed(1814) #TURN OFF LATER!!!

    done = [False for _ in range(n_episodes)]
    states = [e.reset().flatten() for e in envs]

    while not all(done):

        unfinished_ids = [i for i in range(n_episodes) if not done[i]]
        #normalized_states = [s/state_bounds for i, s in enumerate(states) if not done[i]]
        normalized_states = [s.flatten() for i, s in enumerate(states) if not done[i]]

        for i, state in zip(unfinished_ids, normalized_states):
            trajectories[i]['states'].append(state)

        # Compute policy for all unfinished envs in parallel
        states = tf.convert_to_tensor(normalized_states)
        action_probs = model([states])

        # Store action and transition all environments to the next state
        states = [None for i in range(n_episodes)]
        for i, policy in zip(unfinished_ids, action_probs.numpy()):
            action = np.random.choice(n_actions, p=policy) #maybe check for AVAILABLE actions? Unavaiable means selecting IDLE
            states[i], reward, done[i], _ = envs[i].step(action)
            trajectories[i]['actions'].append(action)
            trajectories[i]['rewards'].append(reward)

    collisions = 0
    for i, e in enumerate(envs):
        if e._info(None, None)["crashed"]:
            collisions += 1

    return trajectories, collisions

"""and a function that computes discounted returns $\sum_{t'=1}^{H-t} \gamma^{t'} r_{t+t'}$ out of the rewards $r_t$ collected in an episode:"""

def compute_returns(rewards_history):
    """Compute discounted returns with discount factor `gamma`."""
    returns = []
    discounted_sum = 0
    for r in rewards_history[::-1]: #No baseline subtraction++
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)

    # Normalize them for faster and more stable learning
    returns = np.array(returns)
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
    #returns = returns.tolist()

    return returns

def value_approx(returns, batch_size):
    return(np.sum(returns) / batch_size)

"""Implement a function that updates the policy using states, actions and returns:"""
@tf.function
def reinforce_update(states, actions, returns, batch_size, optimizer_in, optimizer_var, optimizer_out, model):
    states = tf.convert_to_tensor(states)
    actions = tf.convert_to_tensor(actions)
    returns = tf.convert_to_tensor(returns)

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        logits = model(states)
        p_actions = tf.gather_nd(logits, actions)
        log_probs = tf.math.log(p_actions)
        loss = tf.math.reduce_sum(-log_probs * returns) / batch_size

    grads = tape.gradient(loss, model.trainable_variables)

    for optimizer, w in zip([optimizer_in, optimizer_var, optimizer_out], [w_in, w_var, w_out]):
        optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])


def create_model(n_qubits, n_vehicles, n_layers, n_actions, beta=1.0):
    qubits = cirq.GridQubit.rect(1, n_qubits)
    observables = [cirq.Z(q) for q in qubits] #To change each observable to have its own independent Z rotation
    #observables = [reduce((lambda x, y: x * y), ops)] # Z_0*Z_1*Z_2*Z_3
    return generate_model_policy(n_vehicles * n_qubits, qubits, n_layers, n_actions, beta, observables)

"""Model loading and saving functions"""
def save_model(loc, fname, model, n_vehicles, n_layers, epochs, n_episodes, batch_size, a1, a2, a3, folds):
    s_model = [[n_vehicles, n_layers, epochs, n_episodes, batch_size, a1, a2, a3, folds]]
    s_model.append(model.layers[1].get_weights())
    s_model.append(model.layers[2].get_weights())
    s_model = np.array(s_model)
    np.save(loc + fname, s_model)

def load_model(file):
    if type(file) is str:
        file = np.load(file)
    model = create_model(n_qubits, file[0][0], file[0][1], n_actions)
    model.layers[1].set_weights(file[1])
    model.layers[2].set_weights(file[2])
    return model

# Assign the model parameters to each optimizer
w_in, w_var, w_out = 1, 0, 2
gamma = 0.99
n_qubits = 5 # Dimension of the state vectors in Highway
n_actions = 5 # Number of actions in Highway

"""Main training loop of the agent"""
def train_model(model, stats, save_loc, curFold, n_epochs, batch_size, n_episodes, n_vehicles, n_layers, alpha1, alpha2, alpha3, kfold):
    env_info = ['highway-v0', { "type": "Kinematics", "vehicles_count": n_vehicles}]
    n_actions = 5 # Number of actions in Highway
    n_qubits = 5 # Number of features per vehicle

    save_name = 'Metrics-nVehicles-' + str(n_vehicles) + 'nLayers-' + str(n_layers) + '-Alphas-' + str(alpha1) + '-' + str(alpha2) + '-' + str(alpha3) + '.npy'

    # Start training the agent
    episode_reward_history = []
    batch_collisions = []
    batch_slow = []



    if type(stats) is np.ndarray:
        stats = stats.tolist()

    for f in range(curFold, kfold):
        #Set up the optimizers
        optimizer_in = tf.keras.optimizers.Adam(learning_rate=alpha1, amsgrad=True) #Lambda
        optimizer_var = tf.keras.optimizers.Adam(learning_rate=alpha2, amsgrad=True) #Theta
        optimizer_out = tf.keras.optimizers.Adam(learning_rate=alpha3, amsgrad=True) #Observation Weights

        print("\n\nFold " + str (f) + "/" + str(kfold) + ":")
        model_save_name = 'Model-nVehicles-' + str(n_vehicles) + 'nLayers-' + str(n_layers) + '-Alphas-' + str(alpha1) + '-' + str(alpha2) + '-' + str(alpha3) + '-Fold-' + str(f) + '.npy'
        if f > len(stats) -1:
            stats.append([[], []])
        for batch in range(n_episodes // batch_size):

            # Gather episodes
            episodes, col = gather_episodes(None, n_actions, model, batch_size, env_info)
            batch_collisions.append(col)
            # Group states, actions and returns in numpy arrays
            states = np.concatenate([ep['states'] for ep in episodes])
            actions = np.concatenate([ep['actions'] for ep in episodes])
            rewards = [ep['rewards'] for ep in episodes]
            returns = np.concatenate([compute_returns(ep_rwds) for ep_rwds in rewards])
            returns = np.array(returns, dtype=np.float32)

            returns = returns - value_approx(returns, batch_size)

            id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

            # Update model parameters.
            reinforce_update(states, id_action_pairs, returns, batch_size, optimizer_in, optimizer_var, optimizer_out, model)

            # Store collected rewards
            for ep_rwds in rewards:
                episode_reward_history.append(np.sum(ep_rwds))
                stats[f][0].append(np.sum(ep_rwds))

            avg_rewards = np.mean(episode_reward_history[-10:])

            #Save statistics
            stats[f][1].append(col)
            np.save(save_loc + save_name, np.array(stats))

            n_epochs += batch_size
            #Save Model
            save_model(save_loc, model_save_name, model, n_vehicles, n_layers, n_epochs, n_episodes, batch_size, alpha1, alpha2, alpha3, kfold)

            print('Finished episode', n_epochs - (curFold * n_episodes), '/', n_episodes,
                  '\t Average rewards: ', avg_rewards,
                  '\t Crashes: ', batch_collisions[-1],
                  '\t (Fold: ', f, ')')

            if len(stats[-1][0]) >= n_episodes:
                break

        #reset the model for the next fold (Doesn't work yet)
        tf.keras.backend.clear_session()
        model = create_model(n_qubits, n_vehicles, n_layers, n_actions)
