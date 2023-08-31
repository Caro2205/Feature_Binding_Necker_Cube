import matplotlib.pyplot as plt
import numpy as np
import os

########################################################################################################################
# plot feature binding error of multiple runs
########################################################################################################################

def main():

    fig, ax = plt.subplots()

    #C:\Users\49157\OneDrive\Dokumente\UNI\8. Semester\Bachelorarbeit\Python Projects\Code\Binding Framework\Testing\Grafics\binding_test_results_necker_cube_static\test_with_turn_up_inconsistent
    folder_name = 't'

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

    for i in range(10):

        dir = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Python Projects/Code/Binding Framework/Testing/Grafics/'
        folder_path = dir + 'binding_test_results_necker_cube_static/' + folder_name + '/0' + str(i) + '/b_dimensions_3/necker_cube_static_0/'

        fbe = np.loadtxt(folder_path + 'feature_binding_losses.txt')
        fbe = fbe[:, 1]

        ax.plot(fbe, color=colors[i], label=str(i))

    ax.set_xlabel('Tuning Cycle')
    ax.set_ylabel('Feature Binding error')
    #ax.set_yscale('log')
    ax.set_xlim(0, 4100)

    plt.legend(loc='upper right')

    path = dir + 'binding_test_results_necker_cube_static/' + folder_name + '/'
    plt.savefig(path + 'fbe.png')
    plt.close()


if __name__ == "__main__":
    main()