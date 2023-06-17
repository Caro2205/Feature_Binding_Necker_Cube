import os
import scipy.stats as stats
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import VAE_model
import torch
from VAE_pipeline import CubeData, test_model, save_images


def save_test_info(folderdir, testdata_size, test_losses, xy_test_losses, z_test_losses, filename):
    # t-test between z coords and xy coords
    ttest_test_losses_stat, ttest_test_losses_p = stats.ttest_rel(xy_test_losses, z_test_losses, alternative='less')

    os.makedirs(folderdir, exist_ok=True)

    with open(os.path.join(folderdir, filename), "w") as f:
        print('Number of cubes in test-set: ' +str(testdata_size), file=f)
        print('Average test loss overall: ' + str(np.round(np.mean(test_losses), decimals=4)), file=f)
        print('Average test loss xy-coordinates: ' + str(np.round(np.mean(xy_test_losses), decimals=4)), file=f)
        print('Average test loss z-coordinates: ' + str(np.round(np.mean(z_test_losses), decimals=4)), file=f)
        print('t-test results test loss higher in z-coords than xy-coords:', file=f)
        print('test statistic: ' + str(np.round(ttest_test_losses_stat, decimals=4)), file=f)
        print('p-value: ' + str(np.round(ttest_test_losses_p, decimals=4)), file=f)
    f.close()

def main():
    has_vis_marker = False
    input_size = 8 * 4 if has_vis_marker else 8 * 3
    datapath = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Data/model_testing/'

    # set used model here
    datetime = '06_15-23_38_32'  #'06_06-13_18_12'
    folderdir = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/model_runs/run_2023_' + datetime + '/'
    model_path = folderdir + '/saved_model_parameters.pt'
    model = VAE_model.VariationalAutoencoder(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # create folder for storing overall test results
    folderdir = os.path.join(folderdir, 'test_results/')
    os.makedirs(folderdir, exist_ok=True)

    all_test_losses = []
    all_test_losses_z = []
    all_test_losses_xy = []

    for n_test in range(16):
        # set path to data
        data_filename = f"{n_test:02d}_data.txt"
        target_filename = f"{n_test:02d}_target.txt"
        test_datapath = os.path.join(datapath, data_filename)  # data used to see how the model performs on that data after certain training epochs
        test_targetpath = os.path.join(datapath, target_filename)

        # read in data
        test_criterion = nn.MSELoss(reduction='none')
        test_dataset = CubeData(test_datapath, has_vis_marker, test_targetpath)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset)

        #run model, create reconstructions, save losses
        test_losses, test_coords, z_test_losses, xy_test_losses = test_model(model, test_loader, test_criterion)

        # save average test losses overall, for z coords and xy coords
        all_test_losses.append(np.round(np.mean(test_losses), decimals=4))
        all_test_losses_z.append(np.round(np.mean(z_test_losses), decimals=4))
        all_test_losses_xy.append(np.round(np.mean(xy_test_losses), decimals=4))

        filename = f"test{n_test:02d}.txt"
        testdata_size = test_dataset.__len__()
        save_test_info(folderdir, testdata_size, test_losses, xy_test_losses, z_test_losses, filename)

        # create folder for saving reconstruction images and save them
        imagedir = os.path.join(folderdir, 'reconstruction_images/')
        os.makedirs(imagedir, exist_ok=True)
        test_imagedir = os.path.join(imagedir, filename + '/')
        os.makedirs(test_imagedir, exist_ok=True)

        indices = list(range(testdata_size))

        for i in indices:
            cube, target = test_dataset.__getitem__(i)
            output, _, _ = model(cube, mode="testing")      # create reconstruction

            filename = "cube_" + str(i + 1)
            path = test_imagedir + filename + ".png"

            save_images(cube, output, target, path, has_vis_marker, mode=None)

    # create barchart to compare losses of different tests
    losses = np.vstack((all_test_losses, all_test_losses_z, all_test_losses_xy))

    fig, ax = plt.subplots()

    bar_width = 0.3  # Width of each bar
    x = np.arange(16)  # x-axis positions
    offset = bar_width / 2  # Offset for aligning bars
    labels = ['Test {}'.format(i + 1) for i in range(16)]  # Entry labels
    colors = ['steelblue', 'darkorange', 'limegreen']

    for i in range(16):
        entry_data = losses[:, i]
        bar_positions = x[i] + np.array([-bar_width, 0, bar_width])
        ax.bar(bar_positions, entry_data, width=bar_width, color=colors)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel('average RMSE')
    #ax.set_title('Bar Chart of Three Vectors')
    legend_labels = ['all coords', 'z coords', 'xy coords']
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    ax.legend(legend_handles, legend_labels)

    # add dashed lines for better visualization
    dashed_lines_heights = [0.01, 0.02, 0.03, 0.04]
    for height in dashed_lines_heights:
        ax.axhline(height, linestyle='dashed', color='gray', linewidth = 0.5)

    plt.savefig(os.path.join(folderdir, 'comparison.png'), dpi=300, bbox_inches='tight')
    #plt.show()

if __name__ == "__main__":
    main()