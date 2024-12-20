import deepxde as dde
import numpy as np
from deepxde.backend import tf
from scipy import io
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
import os
from itertools import cycle

np.random.seed(42)

# Define the ODE system
def harmonic_oscillator(t, z, omega):
    x, v = z  # z = [x, v], where v = dx/dt
    dxdt = v
    dvdt = -omega**2 * x
    return [dxdt, dvdt]

# Make dataset
def get_data(load=True, save=True, q0=None, p0=None,tmax=25):
    if load and os.path.exists('harmonic_oscillator_datasets.pkl'):
        with open('harmonic_oscillator_datasets.pkl', 'rb') as f:
            (x_train, y_train, x_test, y_test) = pickle.load(f)
    else:
        n_branch = 1000
        n_trunk = 1000
        if q0 is None:
            q0 = np.random.uniform(-1, 1, size=(n_branch, 1))  # Initial position (x(0))
        else: 
            q0 = np.ones((n_branch,1)) * q0
        if p0 is None: 
            p0 =  np.random.uniform(-1, 1, size=(n_branch, 1))  # Initial velocity (dx/dt at t=0)
        else:
            p0 = np.ones((n_branch,1)) * p0
        omega = np.random.uniform(0, 1, size=(n_branch, 1)) # Angular frequency
        trunk_data =  np.random.uniform(0, tmax, size=(n_trunk,)) #n_trunk
        for array in [q0, p0, omega]:
            np.random.shuffle(array)


        branch_data = np.concatenate((q0,p0,omega), axis=1) #n_branch x 3
        branch_train = branch_data[0:int(0.8*branch_data.shape[0])]
        branch_test = branch_data[int(0.8*branch_data.shape[0]):]
        trunk_train = trunk_data[0:int(0.8*trunk_data.shape[0])]
        trunk_test = trunk_data[int(0.8*trunk_data.shape[0]):]  

        # Define time span for numerical solver
        t_span = (0, tmax)

        #generate y_train
        y_train = []
        for i in range(branch_train.shape[0]):
            print(f'starting on row {i} in train data')
            row = []
            for j in range(trunk_train.shape[0]):
                # Initial conditions
                z0 = branch_train[i,0:2] #initial phase space
                (omega) = branch_train[i,2] #angular velocity
                t = trunk_train[j].flatten()
                # Solve the ODE
                solution = solve_ivp(harmonic_oscillator, t_span, z0, args=(omega,), t_eval=t)
                # Extract the solution
                row.append(solution.y[0]) 
            y_train.append(row)
        
        y_train = np.squeeze(np.array(y_train))

        #generate y_test
        y_test = []
        for i in range(branch_test.shape[0]):
            print(f'starting on row {i} in test data')
            row = []
            for j in range(trunk_test.shape[0]):
                # Initial conditions
                z0 = branch_test[i,0:2]
                (omega) = branch_test[i,2]
                t = trunk_test[j].flatten()
                # Solve the ODE
                solution = solve_ivp(harmonic_oscillator, t_span, z0, args=(omega,), t_eval=t)
                # Extract the solution
                row.append(solution.y[0])  # Position x(t)
            y_test.append(row)
        y_test = np.squeeze(np.array(y_test))
        x_train = (branch_train.astype(np.float32), trunk_train.astype(np.float32)[:, np.newaxis])
        x_test = (branch_test.astype(np.float32), trunk_test.astype(np.float32)[:, np.newaxis])
        #save data for reuse
        if save:
            with open('harmonic_oscillator_datasets.pkl', 'wb') as f:
                pickle.dump([x_train, y_train, x_test, y_test], f)

    return x_train, y_train, x_test, y_test

def train(model, lr, epochs):
    decay = ("inverse time", epochs // 5, 0.5)
    model.compile("adam", lr=lr, metrics=["mean l2 relative error"], decay=decay)
    losshistory, train_state = model.train(epochs=epochs, batch_size=None)
    visualize_loss(losshistory)
    print("\nTraining done ...\n")


def main():
    tmax = 25
    x_train, y_train, x_test, y_test = get_data(tmax=tmax)

    print(f'Shapes are: {x_train[0].shape}, {x_train[1].shape}, {y_train.shape}, {x_test[0].shape}, {x_test[1].shape}, {y_test.shape}')

    net = dde.maps.DeepONetCartesianProd(
        [3, 128, 128, 128, 128], [1, 128, 128, 128], "tanh", "Glorot normal"
    )

    data = dde.data.TripleCartesianProd(x_train, y_train, x_test, y_test)
    model = dde.Model(data, net)

    lr = 0.001
    epochs = 10000
    train(model, lr, epochs)
    print(f'generating example data')

    tmax_pred = 50
    examples = ([0.5,0,0.5], [0,0.5,0.5], [0,0,0.5])
    example_t = t = np.linspace(0,50,500)[:, np.newaxis]
    output = []
    ground_truth = []
    for x in examples:
        example_u = np.array(x)[:, np.newaxis]
        output.append(np.squeeze(model.predict((example_u.T, example_t))))
        ground_truth.append(solve_ivp(harmonic_oscillator, (0, tmax_pred), np.squeeze(example_u[0:2]), args=(example_u[2].item(),), t_eval=np.squeeze(example_t)).y[0])
    visualize_example(examples, example_t, output, ground_truth)

    

def visualize_loss(losshistory):
    train_loss = losshistory.loss_train
    test_loss = losshistory.loss_test
    steps = losshistory.steps

    # Plotting the curves
    plt.figure(figsize=(8, 6))
    plt.plot(steps, train_loss, label='Train Loss', marker='o')
    plt.plot(steps, test_loss, label='Test Loss', marker='s')
    plt.xlabel('Steps', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Loss Curves', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')  # Save as a PNG file
    plt.show()

    return None

def visualize_example(labels,x,y_hat,y):
    # Plotting the curves
    plt.figure(figsize=(8, 6))
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])  # or define your own colors

    for i, label in enumerate(labels):
        color = next(colors)
        plt.plot(x, y[i], label=f'{label} – ground truth', linestyle='--', color=color)
        plt.plot(x, y_hat[i], label=f'{label} – prediction', linestyle='solid', color=color)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('time', fontsize=14)
    plt.title('Prediction vs Ground Truth', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig('predictions.png', dpi=300, bbox_inches='tight')  # Save as a PNG file
    plt.show()

    return None


if __name__ == "__main__":
    main()







