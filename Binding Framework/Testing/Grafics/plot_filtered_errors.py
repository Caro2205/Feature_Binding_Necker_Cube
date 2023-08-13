import matplotlib.pyplot as plt
import numpy as np
import os

def main():

    all_ORE = []

    fig, ax = plt.subplots()

    #C:\Users\49157\OneDrive\Dokumente\UNI\8. Semester\Bachelorarbeit\Python Projects\Code\Binding Framework\Testing\Grafics\binding_test_results_necker_cube_static\test_with_turn_up_inconsistent
    folder_name = 'with attractor lambda 0.005'

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

    for i in range(10):

        dir = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Python Projects/Code/Binding Framework/Testing/Grafics/'
        folder_path = dir + 'binding_test_results_necker_cube_static/' + folder_name + '/0' + str(i) + '/b_dimensions_3/necker_cube_static_0/'

        filtered_rec_losses = np.loadtxt(folder_path + 'filtered_reconstruction_loss.txt')
        ORE = np.loadtxt(folder_path + 'ORE_value.txt')
        all_ORE.append(ORE)

        ax.plot(filtered_rec_losses, color=colors[i], label=str(i))

    ore_mean = np.mean(all_ORE)
    ax.axhline(ore_mean, color='blue', label='mean ORE') # optimal reconstruction error
    ax.set_xlabel('Tuning Cycle')
    ax.set_ylabel('Filtered Reconstruction Loss')
    ax.set_yscale('log')
    ax.set_xlim(0, 4100)

    plt.legend(loc='upper right')

    path = dir + 'binding_test_results_necker_cube_static/' + folder_name + '/'
    plt.savefig(path + 'filtered_losses_log.png')
    plt.close()


if __name__ == "__main__":
    main()