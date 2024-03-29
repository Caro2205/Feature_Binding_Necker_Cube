import matplotlib.pyplot as plt
import numpy as np
import os

########################################################################################################################
# plot feature binding error of multiple runs
########################################################################################################################

def main():

    #"C:\Users\49157\OneDrive\Dokumente\UNI\8. Semester\Bachelorarbeit\Python Projects\Code\Binding Framework\Testing\Grafics\binding_test_results_necker_cube_static\experiment02_temp_dep_on_loss\00\b_dimensions_3\NECKER~1\FILTER~1.TXT"

    fig, ax = plt.subplots()

    #C:\Users\49157\OneDrive\Dokumente\UNI\8. Semester\Bachelorarbeit\Python Projects\Code\Binding Framework\Testing\Grafics\binding_test_results_necker_cube_static\test_with_turn_up_inconsistent
    #folder_name = 'experiment 2 - temp abhängig von loss'
    folder_name = 'experiment02_temp_dep_on_loss'

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

        fbe = np.loadtxt(folder_path + 'feature_binding_losses.txt')
        fbe = fbe[:, 1]

        ax.plot(fbe, color=colors[i], label=str(i))

    ax.set_xlabel('Tuning Cycle')
    ax.set_ylabel('Feature Binding Error')
    #ax.set_yscale('log')
    ax.set_xlim(0, 4100)

    plt.legend(loc='upper right', title='Cube/Run')

    path = dir + 'binding_test_results_necker_cube_static/' + folder_name + '/'
    plt.savefig(path + 'fbe.png', dpi=400)
    plt.close()


if __name__ == "__main__":
    main()