import os
import scipy.stats as stats
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import VAE_model
import torch
from VAE_pipeline import CubeData, test_model, save_images
from VAE_pipeline import save_loss_plot


def save_test_info(folderdir, testdata_size, test_losses, xy_test_losses, z_test_losses):
    # t-test between z coords and xy coords
    ttest_test_losses_stat, ttest_test_losses_p = stats.ttest_rel(xy_test_losses, z_test_losses, alternative='less')

    os.makedirs(folderdir, exist_ok=True)

    test_name = 'test00.txt'
    with open(os.path.join(folderdir, test_name), "w") as f:
        print('Number of cubes in test-set: ' +str(testdata_size), file=f)
        print('Average test loss overall: ' + str(np.round(np.mean(test_losses), decimals=4)), file=f)
        print('Average test loss xy-coordinates: ' + str(np.round(np.mean(xy_test_losses), decimals=4)), file=f)
        print('Average test loss z-coordinates: ' + str(np.round(np.mean(z_test_losses), decimals=4)), file=f)
        print('t-test results test loss higher in z-coords than xy-coords:', file=f)
        print('test statistic: ' + str(np.round(ttest_test_losses_stat, decimals=4)), file=f)
        print('p-value: ' + str(np.round(ttest_test_losses_p, decimals=4)), file=f)
    f.close()

def main():
    has_vis_marker = True
    input_size = 8 * 4 if has_vis_marker else 8 * 3

    # set used model here
    datetime = '06_06-13_18_12'
    folderdir = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/model_runs/run_2023_' + datetime
    model_path = folderdir + '/saved_model_parameters.pt'
    model = VAE_model.VariationalAutoencoder(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_datapath = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Data/test_data.txt'  # data used to see how the model performs on that data after certain training epochs
    test_targetpath = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Data/test_target.txt'

    test_criterion = nn.MSELoss(reduction='none')
    test_dataset = CubeData(test_datapath, has_vis_marker, test_targetpath)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset)
    testimages_idx = list(range(0, test_dataset.__len__()))

    # save_loss_plot(losses, title, x_label, y_label, path, n_epochs, add_losses_z=None, add_losses_xy=None)
    test_losses, test_coords, z_test_losses, xy_test_losses = test_model(model, test_loader, test_criterion)

    folderdir = os.path.join(folderdir, 'test_results')
    os.makedirs(folderdir, exist_ok=True)

    testdata_size = test_dataset.__len__()
    save_test_info(folderdir, testdata_size, test_losses, xy_test_losses, z_test_losses)


if __name__ == "__main__":
    main()