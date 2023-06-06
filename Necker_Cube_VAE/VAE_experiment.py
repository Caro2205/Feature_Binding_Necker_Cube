
import numpy as np
import VAE_model
from VAE_pipeline import CubeData
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
    has_vis_marker = True
    input_size = 8 * 4 if has_vis_marker else 8 * 3

    # get trained model
    model_path = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/model_runs/run_2023_06_06-13_18_12/saved_model_parameters.pt'
    model = VAE_model.VariationalAutoencoder(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    datapath = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Data/experiment_data.txt'
    targetpath = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Data/experiment_target.txt'

    train_batch_size = 1 # only look at one cube
    dataset = CubeData(datapath, has_vis_marker, targetpath)

    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=train_batch_size)

    # get corners of cube and target cube
    corners, coordinates = train_loader.dataset[0]

    mask = [1, 4, 5, 6, 10, 14, 16, 17, 20, 24, 25, 28] #3, 7, 11, 15, 19, 23, 27, 31 are visibility markers
    corners_masked = corners.clone()
    corners_masked[mask] = 0

    # how often to send missing coordinates through VAE
    runs = 10

    for i in range(runs):
        reconstruction, mu, sigma = model(corners_masked)

        criterion = nn.MSELoss(reduction='none')
        MSE = criterion(reconstruction, coordinates)
        root_MSE = torch.sqrt(MSE)
        root_MSE_sum = torch.sum(root_MSE)
        loss = root_MSE_sum / 24
        print('loss in run ' + str(i+1))
        print(loss)

        # add new values to corners_masked
        reconstructed_vis = torch.empty(32)
        k = 0
        for j in range(32):
            if (j+1) % 4 == 0:
                reconstructed_vis[j] = 1
            else:
                reconstructed_vis[j] = reconstruction[k]
                k += 1

        corners_masked[mask] = reconstructed_vis[mask]


if __name__ == "__main__":
    main()