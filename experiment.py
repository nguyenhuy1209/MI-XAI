import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
np.random.seed(0)
from experimental_setup import ZivInformationPlane, BufferedSequential, mutual_information_for_network_family, ReshapeLayer

def gen_data():
    X = np.random.randint(5, size=(1000, 12)) - 1
    y = (np.mean(X, axis=1) + np.random.normal(scale=0.1, size=1000) > 0.7).astype(int)
    Y = y.reshape(-1, 1)
    return X, Y

def experiment(base_network, X, y, 
               epochs=1, lr=.2, batch_size=101, 
               network_copies=1, 
               plot='dynamic', 
               activation_bins=np.linspace(-1,1,50)):
    loss = torch.nn.MSELoss()
    network_copies = [copy.deepcopy(base_network) for _ in range(network_copies)]
    solvers = [torch.optim.SGD(params=network.parameters(), lr=lr) for network in network_copies]
    infoplane = ZivInformationPlane(X, y, bins=activation_bins)
    
    if plot == 'dynamic' or plot == 'once':
        fig = plt.figure(figsize=(12,5))
        graph = fig.add_subplot(111)
        graph.set_xlabel('I(X;T)')
        graph.set_ylabel('I(Y;T)')

        sgd_fig = plt.figure(figsize=(12,5))
        sgd_graph = sgd_fig.add_subplot(111)
        sgd_graph.set_xlabel('# Epochs')
        sgd_graph.set_ylabel('Normalized Mean and STD')
        
    if plot == 'once':
        mi_history = [[] for _ in range(base_network.n_buffers)]
    
    means = []
    stds = []
    for epoch in tqdm(range(epochs)):
        for network, solver in zip(network_copies, solvers):
            slice_ = np.random.permutation(range(len(X)))[:batch_size]
            X_batch = Variable(torch.from_numpy(X[slice_])).float()
            y_batch = Variable(torch.from_numpy(y[slice_])).float()

            solver.zero_grad()
            pred_batch = network(X_batch)

            loss(pred_batch, y_batch).backward()
            grad = network.layers[0].weight.grad.cpu().detach().numpy()
            mean_grad = np.mean(np.absolute(grad))
            std_grad = np.std(grad)
            means.append(mean_grad)
            stds.append(std_grad)
            solver.step()
        
        mi = mutual_information_for_network_family(infoplane, network_copies)
        
        if plot == 'dynamic':
            graph.scatter(*zip(*mi), s=10, c=np.linspace(0, 1, base_network.n_buffers), alpha=epoch/epochs)
            # display.clear_output(wait=True)
            # display.display(fig)
            # fig.show()
            fig.savefig(f"plot_{epoch}.png", dpi=500, format="png")
        elif plot == 'once':
            for history, new_point in zip(mi_history, mi):
                history.append(new_point)
        
    if plot == 'once':
        # for i, history in enumerate(mi_history):
        # #     graph.plot(*zip(*history), marker="o", markersize=4)
        #     graph.scatter(*zip(*history), c=range(epochs), cmap='gnuplot')
        # plt.savefig(f"plot_layer.png", dpi=1000, format="png")
        sgd_graph.plot(range(len(means)), means, linestyle="-")
        sgd_graph.plot(range(len(stds)), stds, linestyle="--")
        plt.savefig(f"sgd_plot.png", dpi=1000, format="png")
            
    return network_copies

if __name__ == '__main__':
    layers = [
        nn.Linear(12, 10),
        nn.Tanh(),
        nn.Linear(10, 7),
        nn.Tanh(),
        nn.Linear(7, 5),
        nn.Tanh(),
        nn.Linear(5, 4),
        nn.Tanh(),
        nn.Linear(4, 3),
        nn.Tanh(),
        nn.Linear(3, 1),
        nn.Tanh()
    ]
    buffer_mask = [False, True, False, True, False, True, False, True, False, True, False, True]
    tishby_architecture = BufferedSequential(layers, buffer_mask)
    X, Y = gen_data()
    result_nets = experiment(tishby_architecture, X, Y, epochs=10, network_copies=1, plot='once')
