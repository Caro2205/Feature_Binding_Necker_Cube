import os
import random
import numpy as np
import matplotlib.pyplot as plt
import VAE_model
from VAE_pipeline import CubeData, save_images
from VAE_pipeline import draw_cube
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pygame

'''
 Experiment: leaving out coordinates of a cube and 

'''

def main():
    # path to model that we want to do the experiment with
    use_whole_rec = False # should the whole reconstruction be used as input or only the values set to 0 at beginning
    runs = 20

    path = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/model_runs/run_2023_06_12-14_44_21/'
    has_vis_marker = True
    input_size = 8 * 4 if has_vis_marker else 8 * 3

    # get trained model
    model_path = path + 'saved_model_parameters.pt'
    model = VAE_model.VariationalAutoencoder(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # experiment data
    datapath = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Data/experiment_data.txt'
    targetpath = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Data/experiment_target.txt'

    batch_size = 1 # only look at one cube
    dataset = CubeData(datapath, has_vis_marker, targetpath)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)

    # create folder to store experiment results and pictures
    exp_path = path + 'experiment'
    os.makedirs(exp_path, exist_ok=True)
    for i in range(5):
        cube_path = os.path.join(exp_path, f'cube{i}')
        os.makedirs(cube_path, exist_ok=True)
        for j in range(24):
            run_path = os.path.join(cube_path, f'missing_values_{j}')
            os.makedirs(run_path, exist_ok=True)

    # create mask
    not_vis_markers = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30]
    def create_mask(n):
        chosen_values = random.sample(not_vis_markers, n)
        return chosen_values

    # initialize values
    loss_matrix = np.zeros((24, runs))

    for n in range(24):
        running_loss = np.zeros(runs)
        x = 0
        print('Works fine till')
        print(n)

        for corners, coordinates in dataloader.dataset:
            corners_masked = corners.clone()
            mask = create_mask(n)
            corners_masked[mask] = 0

            for i in range(runs):
                reconstruction, mu, sigma = model(corners_masked)

                if x < 5:
                    cube, target = dataset.__getitem__(x)
                    cube_path = os.path.join(exp_path, f'cube{x}')
                    run_path = os.path.join(cube_path, f'missing_values_{n}')
                    img_path = os.path.join(run_path, f'run{i}.png')
                    save_images(cube, reconstruction, target, img_path, has_vis_marker=False, mode=None)

                criterion = nn.MSELoss(reduction='none')
                MSE = criterion(reconstruction, coordinates)
                root_MSE = torch.sqrt(MSE)
                root_MSE_sum = torch.sum(root_MSE)
                running_loss[i] += root_MSE_sum

                # add new values to corners_masked
                reconstructed_vis = torch.empty(32)
                k = 0
                for j in range(32):
                    if (j + 1) % 4 == 0:
                        reconstructed_vis[j] = 1
                    else:
                        reconstructed_vis[j] = reconstruction[k]
                        k += 1

                if use_whole_rec:
                    corners_masked = reconstructed_vis
                else:
                    updated_corners_masked = corners_masked.clone()
                    updated_corners_masked[mask] = reconstructed_vis[mask]
                    corners_masked = updated_corners_masked
                    #mask_tensor = torch.tensor(mask)
                    #corners_masked.scatter_(dim=0, index=mask_tensor, src=reconstructed_vis[mask])
                    # #corners_masked[mask] = reconstructed_vis[mask]

            x += 1

            #loss_matrix[n, :] = running_loss / len(dataloader.dataset)
        loss = running_loss / (len(dataloader.dataset) * 24)

        loss_matrix[n, :] = loss

    np.savetxt(os.path.join(exp_path, 'loss_values.txt'), loss_matrix)
    print(loss_matrix)


    # create losses plot
    n_runs = loss_matrix.shape[1]
    run_labels = ['Run {}'.format(i) for i in range(1, n_runs + 1)]
    x = np.arange(1, n_runs+1)
    y = loss_matrix.T

    plt.plot(x, y)
    plt.xlabel('Run')
    plt.ylabel('Loss')
    plt.title('Loss per Run')
    plt.legend(run_labels, bbox_to_anchor=(1.05, 1))
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(exp_path, 'losses_plot.png'), bbox_inches='tight')


if __name__ == "__main__":
    main()