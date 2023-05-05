import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch, Patch
from matplotlib.colors import TwoSlopeNorm

from utils.metrics import spread_pt, efficiency_pt
from utils.plot import plot_image, add_arrow

def evaluate_model(model, test_dataloader, test_labels, test_labels_df, mean, std):
    val_pt = torch.tensor([])
    val_eta = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for data, target in tqdm(test_dataloader, leave=False, mininterval=1):
            batch_out = model(data, target.reshape(-1, 2))
            val_pt = torch.cat((val_pt, batch_out['pred_pt']), 0)
            val_eta = torch.cat((val_eta, batch_out['pred_eta']), 0)

    y_hat_pt = val_pt.detach().numpy()
    y_hat_eta = val_eta.detach().numpy()
    val_spr = spread_pt(test_labels[:,0].reshape(-1, 1) * std + mean, y_hat_pt.reshape(-1, 1) * std + mean)
    val_eff = efficiency_pt(test_labels[:,0].reshape(-1, 1) * std + mean, y_hat_pt.reshape(-1, 1) * std + mean)
    print(f'Spread: {val_spr:0.6f}, Efficiency: {val_eff:0.6f}')

    pred_df = pd.DataFrame(data={'pt_pred': y_hat_pt * std + mean, 'eta_pred': y_hat_eta})
    frames = [test_labels_df, pred_df]
    compare_df = pd.concat(frames, axis=1)
    print(compare_df.head(10))
    return compare_df


def train_and_evaluate(model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       train_dataloader: DataLoader,
                       test_dataloader: DataLoader = None,
                       save_path = '',
                       epochs: int = 5,
                       verbose: bool = True,
                       mean=0,
                       std=0,
                       ):
                       
    train_loss_history = []
    train_spread_history = []
    train_efficiency_history = []

    valid_loss_history = []
    valid_spread_history = []
    valid_efficiency_history = []

    max_val_eff = 0.0

    for epoch in range(epochs):
        # TRAINING ------------------------------------------
        # Batches of the training set
        epoch_losses = []
        epoch_spread = []
        epoch_eff = []
        model.train()
        for data, target in tqdm(train_dataloader, leave=False, mininterval=1):
            optimizer.zero_grad()
            batch_out = model(data, target.reshape(-1, 2))
            loss = batch_out['loss']
            epoch_losses.append(loss)
            output = batch_out['pred_pt'].detach().numpy()
            spr = spread_pt(target[:,0].reshape(-1, 1) * std + mean, output.reshape(-1, 1) * std + mean)
            eff = efficiency_pt(target[:,0].reshape(-1, 1) * std + mean, output.reshape(-1, 1) * std + mean)
            epoch_spread.append(spr.numpy())
            epoch_eff.append(eff.numpy())
            
            # Computes the gradient of the loss
            loss.backward()
            # Updates parameters based on the gradient information
            optimizer.step()

        model.global_epoch += 1
        mean_loss = sum(epoch_losses) / len(epoch_losses)
        train_loss_history.append(mean_loss.item())
        mean_spread = sum(epoch_spread) / len(epoch_spread)
        train_spread_history.append(mean_spread)
        mean_eff = sum(epoch_eff) / len(epoch_eff)
        train_efficiency_history.append(mean_eff)

        if verbose or epoch == epochs - 1:
            print(f'  Epoch {model.global_epoch:3d} => Loss: {mean_loss:0.6f} Spread: {mean_spread:0.6f} Efficiency: {mean_eff:0.6f}')
        
        if verbose and test_dataloader:
            losses = []
            spreads = []
            effs = []
            model.eval()
            with torch.no_grad():
                for data, target in tqdm(test_dataloader, leave=False, mininterval=1):
                    batch_out = model(data, target.reshape(-1, 2))
                    loss = batch_out['loss']
                    losses.append(loss)
                    output = batch_out['pred_pt'].detach().numpy()
                    spr = spread_pt(target[:,0].reshape(-1, 1) * std + mean, output.reshape(-1, 1) * std + mean)
                    eff = efficiency_pt(target[:,0].reshape(-1, 1) * std + mean, output.reshape(-1, 1) * std + mean)
                    spreads.append(spr.numpy())
                    effs.append(eff.numpy())
                
                mean_loss = sum(losses) / len(losses)
                valid_loss_history.append(mean_loss.item())
                mean_spread = sum(spreads) / len(spreads)
                valid_spread_history.append(mean_spread)
                mean_eff = sum(effs) / len(effs)
                valid_efficiency_history.append(mean_eff)

            print(f'    Validation => Loss: {mean_loss:0.6f}, Spread: {mean_spread:0.6f}, Efficiency: {mean_eff:0.6f}')
        
            # Save model if its validation efficiency is increasing
            if mean_eff > max_val_eff:
                torch.save(model.state_dict(), save_path)
                if verbose:
                    print(f'    Validation efficiency increased ({max_val_eff:0.6f} --> {mean_eff:0.6f}): Model saved')
                max_val_eff = mean_eff
    
    return {
        'train_loss_history': train_loss_history,
        'train_spread_history': train_spread_history,
        'train_efficiency_history': train_efficiency_history,
        'valid_loss_history': valid_loss_history,
        'valid_spread_history': valid_spread_history,
        'valid_efficiency_history': valid_efficiency_history,
    }


# Soft Decision Tree Model
class SDT(nn.Module):
    def __init__(self, input_dim, output_dim, depth=5, lamda=1e-3):
        super(SDT, self).__init__()

        assert depth > 0 and lamda > 0

        self.depth = depth
        self.num_internal_nodes = 2 ** self.depth - 1
        self.num_leaf_nodes = 2 ** self.depth

        # Penalty coefficients based on layer's depth
        self.penalty_coeff = [lamda * (2 ** -d) for d in range(self.depth)]

        # Initialize internal nodes (+1 used as bias)
        self.inner_nodes = nn.Sequential(
            nn.Linear(input_dim + 1, self.num_internal_nodes, bias=False),
            nn.Sigmoid()
        )

        # Initialize leaf nodes
        self.leaf_nodes = nn.Linear(self.num_leaf_nodes, output_dim, bias=False)
        nn.init.normal_(self.leaf_nodes.weight)

    def forward(self, X, is_training_data=False):
        mu, penalty = self.forward_process(X)
        self.mu = mu

        y_pred = self.leaf_nodes(self.mu)

        if is_training_data:
            return y_pred, penalty
        else:
            return y_pred

    # Data forwarding process implementation
    def forward_process(self, X):
        batch_size = X.size()[0]
        
        # Add input bias (ones) onto the front of each sample
        X = X.view(batch_size, -1)
        bias = torch.ones(batch_size, 1)
        X = torch.cat((bias, X), 1)

        # Compute path_prob
        path_prob = self.inner_nodes(X)
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)
        self.path_prob = path_prob

        # Compute the regularization term
        mu = X.data.new(batch_size, 1, 1).fill_(1.0)
        penalty = torch.tensor(0.0)
        begin_idx = 0
        end_idx = 1
        for layer_idx in range(0, self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]
            # Regularization term computation
            penalty = penalty + self.compute_penalty(layer_idx, mu, _path_prob)
            mu = mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            mu = mu * _path_prob
            # Update indices
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)
        mu = mu.view(batch_size, self.num_leaf_nodes)

        return mu, penalty

    def compute_penalty(self, layer_idx, mu, _path_prob):
        penalty = torch.tensor(0.0)
        batch_size = mu.size()[0]
        mu = mu.view(batch_size, 2 ** layer_idx)
        _path_prob = _path_prob.view(batch_size, 2 ** (layer_idx + 1))

        for node in range(0, 2 ** (layer_idx + 1)):
            alpha = torch.sum(_path_prob[:, node] * mu[:, node // 2], dim=0) / torch.sum(mu[:, node // 2], dim=0)
            penalty -= 0.5 * self.penalty_coeff[layer_idx] * (torch.log(alpha) + torch.log(1 - alpha))
        return penalty

# Convolutional Soft Decision Tree Model
class ConvSDT(nn.Module):
    def __init__(self, latent_dim, output_dim, depth):
        super(ConvSDT, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.tree = SDT(input_dim=latent_dim, output_dim=output_dim, depth=depth)

        self.loss_fn = nn.L1Loss()
        self.global_epoch = 0
    
    def forward(self, x, y:Optional[torch.Tensor]=None):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.tree(out)

        result = {'pred_pt': out[:,0], 'pred_eta': out[:,1]}
        # Compute loss
        if y is not None:
            loss = self.loss(out, y)
            result['loss'] = loss

        return result

    def loss(self, pred, y):
        return self.loss_fn(pred, y)
    

def explain_image(model, input_img, label_pt, label_eta, denoised_input_img:Optional[np.array]=None, mean=0, std=0, title=None):
    conv_layers = []

    for child in model.children():
        if type(child) == nn.Sequential:
            for layer in child:
                if type(layer) == nn.Conv2d:
                    conv_layers.append(layer)

    fig = plt.figure(figsize=(64, 70//3))
    gs = GridSpec(7+4, 32*2, height_ratios=[1, 1, 0.15, 1, 0.15, 1, 0.15, 1, 0.15, 0.5, 0.5])

    axes = []
    ax0 = fig.add_subplot(gs[30:34])
    pos = ax0.get_position()
    new_pos = [pos.x0, pos.y0 + 0.025, pos.width, pos.height]
    ax0.set_position(new_pos)
    axes.append(ax0)

    plot_image(ax0, input_img.squeeze(), title='Input image', fontsize=25)
    image = input_img.unsqueeze(0)

    outputs = []

    for layer in conv_layers:
        image = layer(image)
        outputs.append(image)

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map, 0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())

    for i in range(len(processed)):
        ax = fig.add_subplot(gs[10*(i+1):10*(i+1)+4])
        axes.append(ax)
        plt.axis('off')
        # plot_image(axes[i+1], processed[i], title='Conv '+str(i), grayscale=False, label=False, fontsize=25)
        # vmin = processed[i].min()
        # vmax = processed[i].max()
        # norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        # axes[i+1].imshow(processed[i], aspect='auto', extent=(0, 384, 0, 9), cmap=plt.get_cmap('seismic'), norm=norm, interpolation='none')
        # axes[i+1].set_title('Conv '+str(i), fontsize=25)
        # axes[i+1].set_xticks([])
        # axes[i+1].set_yticks([])

        # p_rows, p_cols = axes[i].get_images()[0].get_array().shape
        # c_rows, c_cols = axes[i+1].get_images()[0].get_array().shape
        # add_arrow(axes[i], axes[i+1], (c_cols//4, 5), (p_cols//2, 5), color='cyan', lw=5)

    # Collect model parameters for plotting
    kernels = dict()
    biases = dict()
    leaves = dict()

    for name, param in model.tree.named_parameters():
        n = 0
        # Inner nodes
        # Each node has shape (769) = 768 + 1 bias at position [0]
        if 'inner_nodes' in name:
            for i in range(len(param)):
                kernels[str(n)] = np.squeeze(param[i][1:])
                biases[str(n)] = np.squeeze(param[i][0])
                n += 1
        n = 0
        if 'leaf_nodes' in name:
            for i in range(len(param[0])):
                leaves[str(n)] = [np.squeeze(param[0][i]), np.squeeze(param[1][i])]
                n+=1

    # To get predicitions and initialize model.tree.mu and model.tree.path_prob
    out = model(input_img.unsqueeze(0))
    
    # Path to the most promising leaf
    path = ['0']
    path_probs = model.tree.path_prob[0][:, 0] # Probs to go to the left child from a node
    
    tree_images = []
    for key in kernels.keys():
        kernel_image = kernels[key].unsqueeze(0).detach().numpy()
        if key in path:
            if (path_probs[int(key)] < torch.tensor(0.5)) == False:
                # Go left
                child = 2 * int(key) + 1
            else: # Go right
                child = 2 * int(key) +  2
            path.append(str(child))
        # Upsampling kernel to make it match with the flatten input image
        kernel_image = np.resize(kernel_image, (1, 3456))
        img = input_img.unsqueeze(0).view(1, -1)
        kernel_image = img * kernel_image
        kernel_image = kernel_image.view(1, 9, 384).squeeze(0).detach().numpy()
        tree_images.append(kernel_image)

    tree_pos = [94,
                142+64, 174+64,
                198+64*2, 214+64*2, 230+64*2, 246+64*2,
                258+64*3, 266+64*3, 274+64*3, 282+64*3, 290+64*3, 298+64*3, 306+64*3, 314+64*3,
                321+64*4, 325+64*4, 329+64*4, 333+64*4, 337+64*4, 341+64*4, 345+64*4, 349+64*4, 353+64*4, 357+64*4, 361+64*4, 365+64*4, 369+64*4, 373+64*4, 377+64*4, 381+64*4]
    for i in range(len(tree_images)):
        j = 4 if i < 15 else 2
        ax = fig.add_subplot(gs[tree_pos[i]:tree_pos[i]+j])
        # if i == 0:
        #     p_rows, p_cols = axes[i].get_images()[0].get_array().shape
        #     c_rows, c_cols = axes[i+1].get_images()[0].get_array().shape
        #     add_arrow(axes[-1], ax, (c_cols//2, 7), (p_cols//2, 5), color='orange', lw=7)
        axes.append(ax)
        
        # im = plot_image(axes[-1], tree_images[i], title='node '+str(i), grayscale=False, label=False, fontsize=20)
        vmin = tree_images[i].min()
        vmax = tree_images[i].max()
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im = axes[-1].imshow(tree_images[i], aspect='auto', extent=(0, 384, 0, 9), cmap=plt.get_cmap('seismic'), norm=norm, interpolation='none')
        axes[-1].set_title('node '+str(i), fontsize=20)
        axes[-1].set_xticks([])
        axes[-1].set_yticks([])

        if i < 15:
            cbar = plt.colorbar(im, cax=fig.add_subplot(gs[tree_pos[i]+j]))
            cbar.set_ticks([])
        # axes[-1].text(tree_images[i].shape[1]+20, tree_images[i].shape[0]//2,
        #               f'prob:\n{path_probs[i]*100:.2f}%',
        #               weight='bold' if str(i) in path else None)

    preds_pt = np.array([l[0].detach().numpy() for l in leaves.values()]) * std + mean
    preds_eta = np.array([l[1].detach().numpy() for l in leaves.values()])
    mu_list = model.tree.mu[0]
    c = 384+64*4
    for i in range(len(preds_pt)):
        pt = preds_pt[i]
        eta = preds_eta[i]
        mu = mu_list[i] * 100.
        ax = fig.add_subplot(gs[c])
        ax.text(0.5, -0.15, f'{mu:.2f}%\n{pt:.1f}\n{eta:.2f}', fontsize=25)
        ax.axis('off')
        c += 2
        axes.append(ax)

    for i in range(4, len(axes[4:39])):
        left_child = 2 * (i - 4) + 1
        right_child = 2 * (i - 4) + 2
        if i >= 19:
            p_rows, p_cols = axes[i].get_images()[0].get_array().shape
            xA, yA = 1, 0.85
            color = 'green' if str(i-4) in path and str(left_child) in path else 'red'
            lw = 5 if color == 'green' else 2.5
            add_arrow(axes[i], axes[left_child+4], (xA, yA), (p_cols//2, 0), color=color, lw=lw)
            color = 'green' if str(i-4) in path and str(right_child) in path else 'red'
            lw = 5 if color == 'green' else 2.5
            add_arrow(axes[i], axes[right_child+4], (xA, yA), (p_cols//2, 0), color=color, lw=lw)
        else:
            p_rows, p_cols = axes[i].get_images()[0].get_array().shape
            c_rows, c_cols = axes[left_child+4].get_images()[0].get_array().shape
            color = 'green' if str(i-4) in path and str(left_child) in path else 'red'
            lw = 5 if color == 'green' else 2.5
            add_arrow(axes[i], axes[left_child+4], (c_cols, 9), (p_cols//2, 0), color=color, lw=lw)
            c_rows, c_cols = axes[right_child+4].get_images()[0].get_array().shape
            color = 'green' if str(i-4) in path and str(right_child) in path else 'red'
            lw = 5 if color == 'green' else 2.5
            add_arrow(axes[i], axes[right_child+4], (0, 9), (p_cols//2, 0), color=color, lw=lw)

    if denoised_input_img is not None:
        axden = fig.add_subplot(gs[36:40])
        plot_image(axden, denoised_input_img, title='Denoised image', fontsize=25)
        pos = axden.get_position()
        new_pos = [pos.x0, pos.y0 + 0.025, pos.width, pos.height]
        axden.set_position(new_pos)
        # p_rows, p_cols = axes[0].get_images()[0].get_array().shape
        # c_rows, c_cols = axden.get_images()[0].get_array().shape
        # add_arrow(axes[0], axden, (c_cols//2, 7), (p_cols//2, 5), color='purple')

    # Legend
    # ax32 = fig.add_subplot(gs[54:70])
    # ax32.legend(handles=[
    #                      # Patch(color='cyan', label='Convolution'),
    #                      # Patch(color='purple', label='Denoising'),
    #                      # Patch(color='orange', label='Convolutions'),
    #                      Patch(color='green', label='Maximum probability path'),
    #                      Patch(color='red', label='Secondary path')
    #                      ],
    #             prop={'size': 28})
    # ax32.axis('off')
    
    # Output also the real and predicted values
    label_pt = label_pt * std + mean
    pred_pt = out['pred_pt'].detach().numpy() * std + mean
    pred_eta = out['pred_eta'].detach().numpy()
    plt.suptitle(f'Real: [$p_T$={label_pt.numpy():.1f} GeV, $\\eta$={label_eta.numpy():.2f}] Predicted: [$p_T$={pred_pt[0]:.1f} GeV, $\\eta$={pred_eta[0]:.2f}]', fontsize=50)

    if title is not None:
        plt.savefig(title, dpi=300, bbox_inches='tight')
    plt.show()