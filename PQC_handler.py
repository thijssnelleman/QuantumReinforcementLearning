import PQC
import sys
import os
import numpy as np

"""-------------------- MAIN ------------------------------------------------"""

"""Define the hyperparameters:"""

#state_bounds = np.array([2.4, 2.5, 0.21, 2.5])

#Pre-create environment variables

n_vehicles = -1
n_layers = -1

#Pre-create unvarying environment variables
n_qubits = 5 # Dimension of the state vectors in Highway
n_actions = 5 # Number of actions in Highway

#Standard setting
n_epochs = 0 #Epoch statistic counter

print("Enter the subdirectory in which to save/load the results (empty for the current directory):")
save_loc = input()

if len(save_loc) > 0 and [-1] != '/': #Formatting to add the filename to it later
    save_loc += ('/')
else:
    save_loc = ""

if len(save_loc) > 0 and not os.path.exists(save_loc):
    print("Path does not exist. Using current directory.")

if sys.argv[1] == "load":
    print("Select which file you wish to load as your model:")
    dir = os.listdir(None if len(save_loc) == 0 else save_loc)
    files = []
    for f in dir:
        if f.endswith(".npy"):
            files.append(f)

    for index, f in enumerate(files):
        print(str(index) + ". " + f)
    selection = input()
    #Load a previously saved model
    model_file = np.load(save_loc + files[int(selection)], allow_pickle=True)
    model = PQC.load_model(model_file)
    n_vehicles = model_file[0][0]
    n_layers = model_file[0][1]
    n_epochs = model_file[0][2]
    n_episodes = model_file[0][3]
    batch_size = model_file[0][4]
    alpha1 = model_file[0][5]
    alpha2 = model_file[0][6]
    alpha3 = model_file[0][7]
    kfold = model_file[0][8]

    print("And its corresponding statistics file:")
    selection = input()
    stats = np.load(save_loc + files[int(selection)], allow_pickle=True)
else:
    print("Creating New Model...")
    """Prepare the definition of your PQC:"""

    n_vehicles = int(sys.argv[1])
    n_layers = int(sys.argv[2]) # Number of layers in the PQC
    n_episodes = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    alpha1 = float(sys.argv[5])
    alpha2 = float(sys.argv[6])
    alpha3 = float(sys.argv[7])
    kfold = int(sys.argv[8])

    """and its observables:"""
    model = PQC.create_model(n_qubits, n_vehicles, n_layers, n_actions)
    print("Created model with " +str(n_vehicles) + " vehicles and " + str(n_layers) + " layers.\n\n" )
    stats = [[[], []]]

curFold = (len(stats) -1)
if (sys.argv[1] == "load" and len(stats[curFold][0]) >= n_episodes): #Fold is already done, go to next one
    curFold += 1
    #qubits = cirq.GridQubit.rect(1, n_qubits)
    #observables = [cirq.Z(q) for q in qubits] #To change each observable to have its own independent Z rotation
    #observables = [reduce((lambda x, y: x * y), ops)] # Z_0*Z_1*Z_2*Z_3
    #model = generate_model_policy(n_vehicles * n_qubits, qubits, n_layers, n_actions, 1.0, observables)
    model = PQC.create_model(n_qubits, n_vehicles, n_layers, n_actions)

PQC.train_model(model, stats, save_loc, curFold, n_epochs, batch_size, n_episodes, n_vehicles, n_layers, alpha1, alpha2, alpha3, kfold)
