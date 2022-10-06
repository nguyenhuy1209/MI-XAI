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
    
    layers_means = []
    layers_stds = []

    for epoch in tqdm(range(epochs)):
        for network, solver in zip(network_copies, solvers):
            slice_ = np.random.permutation(range(len(X)))[:batch_size]
            X_batch = Variable(torch.from_numpy(X[slice_])).float()
            y_batch = Variable(torch.from_numpy(y[slice_])).float()

            solver.zero_grad()
            pred_batch = network(X_batch)

            loss(pred_batch, y_batch).backward()
            means = []
            stds = []
            for i in range(len(network.layers)):
                if not network.buffer_or_not[i]:
                    grad = network.layers[i].weight.grad.cpu().detach().numpy()
                    mean_grad = np.mean(np.absolute(grad))
                    std_grad = np.std(grad)
                    means.append(mean_grad)
                    stds.append(std_grad)
            layers_means.append(means)
            layers_stds.append(stds)
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
        layers_means = list(zip(*layers_means))
        layers_stds = list(zip(*layers_stds))
        cmap = plt.cm.get_cmap('hsv', len(layers_means))

        mean_plots = []
        std_plots = []
        for i, (layer_mean, layer_std) in enumerate(zip(layers_means, layers_stds)):
            m, = sgd_graph.plot(range(len(layer_mean)), layer_mean, linestyle="-", c=cmap(i), label=f'layer_{i}_mean')
            s, = sgd_graph.plot(range(len(layer_std)), layer_std, linestyle="--", c=cmap(i), label=f'layer_{i}_std')
            mean_plots = mean_plots + [m]
            std_plots = std_plots + [s]

        sgd_fig.subplots_adjust(right=0.8)
        first_legend = plt.legend(handles=mean_plots, bbox_to_anchor=[1.05, 0.8], loc='center left')
        plt.gca().add_artist(first_legend)
        plt.legend(handles=std_plots, bbox_to_anchor=[1.05, 0.4], loc='center left')
        plt.savefig(f"sgd_plot_layer.png", dpi=1000, format="png")
        plt.cla()
            
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
    result_nets = experiment(tishby_architecture, X, Y, epochs=2, network_copies=1, plot='once')
