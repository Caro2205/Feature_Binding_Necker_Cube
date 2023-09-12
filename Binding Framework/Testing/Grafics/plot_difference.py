import matplotlib.pyplot as plt
import numpy as np
import os

########################################################################################################################
# plot difference between reconstruction loss and ore of every cube
########################################################################################################################

def main():

    all_ORE = []

    fig, ax = plt.subplots()
    #"C:\Users\49157\OneDrive\Dokumente\UNI\8. Semester\Bachelorarbeit\Python Projects\Code\Binding Framework\Testing\Grafics\binding_test_results_necker_cube_static\experiment02_temp_dep_on_loss\00\b_dimensions_3\NECKER~1\FILTER~1.TXT"

    #C:\Users\49157\OneDrive\Dokumente\UNI\8. Semester\Bachelorarbeit\Python Projects\Code\Binding Framework\Testing\Grafics\binding_test_results_necker_cube_static\test_with_turn_up_inconsistent
    folder_name = 'experiment02_temp_dep_on_loss'

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

    #"C:\Users\49157\OneDrive\Dokumente\UNI\8. Semester\Bachelorarbeit\Python Projects\Code\Binding Framework\Testing\Grafics\binding_test_results_necker_cube_static\experiment02_temp_dep_on_loss\00\b_dimensions_3\NECKER~1\FILTER~1.TXT"

    for i in range(10):

        dir = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Python Projects/Code/Binding Framework/Testing/Grafics/'
        folder_path = dir + 'binding_test_results_necker_cube_static/' + folder_name + '/0' + str(i) + '/b_dimensions_3/necker_cube_static_0/'

        rec_losses = np.loadtxt(folder_path + 'filtered_reconstruction_loss.txt')
        #rec_losses = rec_losses[:, 1]
        ORE = np.loadtxt(folder_path + 'ORE_value.txt')
        all_ORE.append(ORE)

        difference = rec_losses - ORE
        print('cube ' +str(i) )
        print(np.min(difference))
        ax.plot(difference, color=colors[i], label=str(i))

    #ore_mean = np.mean(all_ORE)
   # ax.axhline(ore_mean, color='k', label='mean ORE') # optimal reconstruction error
    ax.set_xlabel('Tuning Cycle')
    ax.set_ylabel('Reconstruction Loss - ORE')
    #ax.set_yscale('log')
    #ax.set_ylim(-1, 5)
    ax.set_xlim(0, 4100)

    plt.legend(loc='upper right', title='Cube/Run')

    path = dir + 'binding_test_results_necker_cube_static/' + folder_name + '/'
    plt.savefig(path + 'difference.png', dpi=400)
    plt.close()


if __name__ == "__main__":
    main()