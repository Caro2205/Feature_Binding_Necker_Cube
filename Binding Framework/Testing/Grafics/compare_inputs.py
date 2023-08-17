import matplotlib.pyplot as plt
import numpy as np

########################################################################################################################
# plot filtered error of consistent / inconsistent inputs against each other as well as the loss mean of both
########################################################################################################################

def main():
    all_ORE_con = []
    all_ORE_incon = []

    fig, ax = plt.subplots()

    folder_name_con = 'with attractor lambda 0.005' # folder name that contains all runs with consistent inputs
    folder_name_incon = 'w' # runs with inconsistent inputs

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

    for i in range(10):
        dir = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Python Projects/Code/Binding Framework/Testing/Grafics/'
        folder_path_con = dir + 'binding_test_results_necker_cube_static/' + folder_name_con + '/0' + str(
            i) + '/b_dimensions_3/necker_cube_static_0/'

        folder_path_incon = dir + 'binding_test_results_necker_cube_static/' + folder_name_incon + '/0' + str(
            i) + '/b_dimensions_3/necker_cube_static_0/'

        filtered_rec_losses_con = np.loadtxt(folder_path_con + 'filtered_reconstruction_loss.txt')
        ORE_con = np.loadtxt(folder_path_con + 'ORE_value.txt')
        all_ORE_con.append(ORE_con)

        ax.plot(filtered_rec_losses_con, color=colors[i], label=str(i))

        filtered_rec_losses_incon = np.loadtxt(folder_path_incon + 'filtered_reconstruction_loss.txt')
        ORE_incon = np.loadtxt(folder_path_incon + 'ORE_value.txt')
        all_ORE_incon.append(ORE_incon)

        ax.plot(filtered_rec_losses_incon, color='r', linestyle='--')

    ore_mean_con = np.mean(all_ORE_con)
    ax.axhline(ore_mean_con, color='blue', label='mean ORE')  # optimal reconstruction error

    ore_mean_incon = np.mean(all_ORE_incon)
    ax.axhline(ore_mean_incon, color='blue', linestyle='--')  # optimal reconstruction error

    ax.set_xlabel('Tuning Cycle')
    ax.set_ylabel('Filtered Reconstruction Loss')
    ax.set_yscale('log')
    ax.set_xlim(0, 4100)

    plt.legend(loc='upper right')

    path = dir + 'binding_test_results_necker_cube_static/' + folder_name_con + '/'
    plt.savefig(path + 'filtered_losses_log.png')
    plt.close()


if __name__ == "__main__":
    main()