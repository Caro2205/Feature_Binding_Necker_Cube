
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

    model_path = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/model_runs/run_2023_06_06-13_18_12/saved_model_parameters.pt'

    model = VAE_model.VariationalAutoencoder(input_size)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    datapath = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Data/experiment_data.txt'
    targetpath = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Data/experiment_target.txt'

    train_batch_size = 1
    dataset = CubeData(datapath, has_vis_marker, targetpath)

    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=train_batch_size)

    # get corners of one cube
    corners, coordinates = train_loader.dataset[0]

    reconstruction, mu, sigma = model(corners)

    criterion = nn.MSELoss(reduction='none')
    MSE = criterion(reconstruction, coordinates)
    root_MSE = torch.sqrt(MSE)
    root_MSE_sum = torch.sum(root_MSE)
    loss = root_MSE_sum / 24
    print('loss at beginning')
    print(loss)

    #for corners, coordinates in train_loader:
    #    print(corners)
    #    reconstruction, mu, sigma = model(corners)
    #    print(reconstruction)




if __name__ == "__main__":
    main()