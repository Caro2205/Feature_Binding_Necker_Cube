import matplotlib.pyplot as plt
import numpy as np
import os

########################################################################################################################
# plot filtered error of multiple inputs/runs
########################################################################################################################

def main():

    all_ORE = []

    fig, ax = plt.subplots()

    #C:\Users\49157\OneDrive\Dokumente\UNI\8. Semester\Bachelorarbeit\Python Projects\Code\Binding Framework\Testing\Grafics\binding_test_results_necker_cube_static\test_with_turn_up_inconsistent
    #C:\Users\49157\OneDrive\Dokumente\UNI\8. Semester\BACHEL~1\PYTHON~1\Code\BINDIN~1\Testing\Grafics\BINDIN~1\EXPERI~1\00\B_DIME~1\NECKER~1
    folder_name = 'experiment01_framework_like_tim'
    # "C:\Users\49157\OneDrive\Dokumente\UNI\8. Semester\Bachelorarbeit\Python Projects\Code\Binding Framework\Testing\Grafics\binding_test_results_necker_cube_static\experiment01_framework_like_tim\00\b_dimensions_3\NECKER~1\FILTER~1.TXT"

    #colors = ['r', 'g', 'b', 'c', 'm', 'y', '#cd6600', 'orange', 'purple', 'brown']
    colors = [
        '#E377C2',  # Pink
        '#1F77B4',  # Steel Blue
        '#FF7F0E',  # Dark Orange
        '#2CA02C',  # Green
        '#9467BD',  # Lavender
        '#D62728',  # Red
        '#FFD700',  # Gold
        '#7F7F7F',  # Gray
        '#BCBD22',  # Olive
        '#17BECF',  # Light Blue
        #
    ]

    for i in range(10):

        dir = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Python Projects/Code/Binding Framework/Testing/Grafics/'
        folder_path = dir + 'binding_test_results_necker_cube_static/' + folder_name + '/0' + str(i) + '/b_dimensions_3/necker_cube_static_0/'

        filtered_rec_losses = np.loadtxt(folder_path + 'filtered_reconstruction_loss.txt')
        ORE = np.loadtxt(folder_path + 'ORE_value.txt')
        all_ORE.append(ORE)

        ax.plot(filtered_rec_losses, color=colors[i], label=str(i))

    ore_mean = np.mean(all_ORE)
    ax.axhline(ore_mean, color='k', label='mean ORE') # optimal reconstruction error
    ax.set_xlabel('Tuning Cycle')
    ax.set_ylabel('Filtered Reconstruction Loss')
    ax.set_yscale('log')
    ax.set_xlim(0, 4100)

    plt.legend(loc='upper right', title='Cube/Run')

    path = dir + 'binding_test_results_necker_cube_static/' + folder_name + '/'
    plt.savefig(path + 'filtered_losses_log.png', dpi=400)
    plt.close()


    # plot losses (not filtered)
    all_ORE = []

    fig, ax = plt.subplots()

    for i in range(10):

        dir = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Python Projects/Code/Binding Framework/Testing/Grafics/'
        folder_path = dir + 'binding_test_results_necker_cube_static/' + folder_name + '/0' + str(i) + '/b_dimensions_3/necker_cube_static_0/'

        rec_losses = np.loadtxt(folder_path + 'reconstruction_losses.txt')
        rec_losses = rec_losses[:,1]
        ORE = np.loadtxt(folder_path + 'ORE_value.txt')
        all_ORE.append(ORE)

        ax.plot(rec_losses, color=colors[i], label=str(i), linewidth=0.5)

    ore_mean = np.mean(all_ORE)
    print(ore_mean)
    print(np.std(all_ORE))
    ax.axhline(ore_mean, color='k', label='mean ORE') # optimal reconstruction error
    ax.set_xlabel('Tuning Cycle')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_yscale('log')
    ax.set_xlim(0, 4100)

    plt.legend(loc='upper right', title='Cube/Run')

    path = dir + 'binding_test_results_necker_cube_static/' + folder_name + '/'
    plt.savefig(path + 'losses_log.png', dpi=400)
    plt.close()



if __name__ == "__main__":
    main()