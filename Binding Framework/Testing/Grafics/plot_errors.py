import matplotlib.pyplot as plt
import numpy as np
import os


########################################################################################################################
# plot loss and logarithmic loss against cycle as well as the temperature
########################################################################################################################


def main():

    datetime = '2023_Aug_10-15_07_38'
    #datetime = 'test'
    #datetime = 'erster run'
    ore = 0.0008661210886202753
    ore_without_z = 0.0008006133139133453
    dir = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Python Projects/Code/Binding Framework/Testing/Grafics/'
    folder_path = dir + 'binding_test_results_necker_cube_static/' + datetime + '/b_dimensions_3/necker_cube_static_0/'

    reconstruction_losses = np.loadtxt(folder_path + 'reconstruction_losses.txt')
    rec = reconstruction_losses[:, 1]

    feature_binding_losses = np.loadtxt(folder_path + 'feature_binding_losses.txt')
    bind = feature_binding_losses[:, 1]

    filtered_rec_losses = np.loadtxt(folder_path + 'filtered_reconstruction_loss.txt')

    temp_col = np.loadtxt(folder_path + 'temperature_values_column.txt')
    temp_row = np.loadtxt(folder_path + 'temperature_values_row.txt')

    ############## plot filtered reconstruction losses ################################################
    fig, ax1 = plt.subplots()
    ax1.plot(filtered_rec_losses, color='r', label='filtered reconstruction loss')
    ax1.axhline(ore, color='blue', label='ORE') # optimal reconstruction error
    #ax1.axhline(ore_without_z, color='green', label='ORE without z') # ore calculated without the z values
    ax1.set_xlabel('Tuning Cycle')
    ax1.set_ylabel('Filtered Reconstruction Loss')

    ax2 = ax1.twinx()
    ax2.plot(temp_col, color='yellow', label='column temperature')
    ax2.set_ylabel('Temperature Value')
    ax2.set_ylim([0, 5])

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    plt.legend(all_handles, all_labels)
    plt.savefig(folder_path + 'filtered_reconstruction_losses.png')
    plt.close()

    ############## plot logarithmic filtered reconstruction losses ################################################
    fig, ax1 = plt.subplots()
    ax1.plot(filtered_rec_losses, color='r', label='filtered reconstruction loss')
    ax1.axhline(ore, color='blue', label='ORE')  # optimal reconstruction error
    # ax1.axhline(ore_without_z, color='green', label='ORE without z') # ore calculated without the z values
    ax1.set_xlabel('Tuning Cycle')
    ax1.set_ylabel('Filtered Reconstruction Loss')
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ax2.plot(temp_col, color='yellow', label='column temperature')
    ax2.set_ylabel('Temperature Value')
    ax2.set_ylim([0, 5])

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    plt.legend(all_handles, all_labels)
    plt.savefig(folder_path + 'filtered_reconstruction_losses_log.png')
    plt.close()


    ############## plot reconstruction losses #########################################################
    fig, ax1 = plt.subplots()

    ax1.plot(rec, color='r', label='reconstruction error')
    ax1.axhline(ore, color='blue', label='ORE') # optimal reconstruction error
    ax1.axhline(ore_without_z, color='green', label='ORE without z') # ore calculated without the z values
    ax1.set_xlabel('Tuning Cycle')
    ax1.set_ylabel('Reconstruction Loss')

    ax2 = ax1.twinx()
    ax2.plot(temp_col, color='yellow', label='column temperature')
    ax2.set_ylabel('Temperature Value')
    ax2.set_ylim([0, 5])
    #plt.plot(temp_row, color='black', label='row temperature')

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    plt.legend(all_handles, all_labels)
    plt.savefig(folder_path + 'reconstruction_losses.png')
    plt.close()

    ############## plot logarithmic reconstruction losses ################################################
    fig, ax1 = plt.subplots()

    ax1.plot(rec, color='r', label='reconstruction error')
    ax1.set_yscale('log')
    ax1.axhline(ore, color='blue', label='ORE') # optimal reconstruction error
    ax1.axhline(ore_without_z, color='green', label='ORE without z') # ore calculated without the z values
    ax1.set_xlabel('Tuning Cycle')
    ax1.set_ylabel('Reconstruction Loss')

    ax2 = ax1.twinx()
    ax2.plot(temp_col, color='yellow', label='column temperature')
    ax2.set_ylabel('Temperature Value')
    ax2.set_ylim([0, 5])
    #plt.plot(temp_row, color='black', label='row temperature')
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    # Plot the combined legend
    plt.legend(all_handles, all_labels)
    plt.savefig(folder_path + 'reconstruction_losses_log.png')
    plt.close()
    plt.plot(bind, color='r')
    plt.xlabel('Tuning Cycle')
    plt.ylabel('Feature Binding Loss')
    plt.savefig(folder_path + 'feature_binding_losses.png')

if __name__ == "__main__":
    main()