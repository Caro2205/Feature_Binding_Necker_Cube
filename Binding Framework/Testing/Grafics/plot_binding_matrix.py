import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


########################################################################################################################
# plot multiple binding matrices in one plot
########################################################################################################################


def main():

    datetime = '2023_Sep_08-22_02_22'
    dir = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Python Projects/Code/Binding Framework/Testing/Grafics/'
    folder_path = dir + 'binding_test_results_necker_cube_static/' + datetime + '/b_dimensions_3/necker_cube_static_0/'
    bm_directory = folder_path + 'binding_matrices'

    selected_indices = [0, 1, 2, 5, 9, 10, 11, 12, 15, 30]
    titles = ['0', '100', '200', '500', '900', '1000', '1100', '1200', '1500', '3000']
    selected_matrices = []
    for index in selected_indices:
        filename = bm_directory + f"/bm_{index}.csv"
        matrix = np.loadtxt(filename, delimiter=',')
        selected_matrices.append(matrix)

        # Convert the NumPy array back to a PyTorch tensor
        #loaded_tensor = torch.from_numpy(loaded_numpy_array)

        # Append the loaded tensor to the list
        #loaded_tensors.append(loaded_tensor)

    #print(name)

    # Set up the plotting environment
    matrix_rc = {
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'text.color': 'black'
    }
    sns.set(rc=matrix_rc)
    palette = "viridis"
    sns.set_palette(palette)

    # Create the subplots
    num_selected = len(selected_indices)
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))

    for i, matrix in enumerate(selected_matrices):
        row, col = divmod(i, 5)

        cax = sns.heatmap(
            matrix,
            ax=axes[row, col],
            cmap=palette,
            vmin=0.0, vmax=1.0,
            cbar=False,
            xticklabels=False,
            yticklabels=False
        )

        axes[row, col].set_title(f"{titles[i]}", fontsize=18)
        axes[row, col].set_aspect('equal')

    # Hide empty subplots if necessary
    for i in range(num_selected, 10):
        row, col = divmod(i, 5)
        fig.delaxes(axes[row, col])

    # Create a color scale on the side
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    sm.set_array([])  # Dummy array for the colorbar
    cbar = fig.colorbar(sm, cax=fig.add_axes([0.92, 0.15, 0.02, 0.7]))
    cbar.ax.tick_params(color='black')
    cbar.ax.tick_params(labelcolor='black')

    #plt.tight_layout()
    #plt.savefig('path to folder', dpi=400)
    plt.show()

if __name__ == "__main__":
    main()