import matplotlib.pyplot as plt
import numpy as np
import os

########################################################################################################################
# plot filtered error of multiple inputs/runs
########################################################################################################################

def main():
    all_losses = []
    fig, ax = plt.subplots()

    #C:\Users\49157\OneDrive\Dokumente\UNI\8. Semester\Bachelorarbeit\Python Projects\Code\Binding Framework\Testing\Grafics\binding_test_results_necker_cube_static\test_with_turn_up_inconsistent
    folder_name = '10 runs medium model'
    dir =  'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/model_runs/' + folder_name

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

    for i in range(10):

        folder_path = dir + '/0' + str(i) + '/'

        rec_losses = np.loadtxt(folder_path + 'reconstruction_losses.txt')
        rec_losses = rec_losses[:,1]

        all_losses.append(rec_losses)

        ax.plot(rec_losses, color=colors[i], label=str(i))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_yscale('log')
    #ax.set_xlim(0, 2000)

    plt.legend(loc='upper right')

    plt.savefig(dir + '/loss_plot.png')
    plt.close()

    mean_loss = np.mean(all_losses, axis=0)
    std = np.std(all_losses, axis=0)

    fig, ax = plt.subplots()
    ax.plot(mean_loss, color='r')
    ax.plot(mean_loss + std, color='b')
    ax.plot(mean_loss - std, color='b')
    ax.set_yscale('log')
    plt.savefig(dir + '/mean_plot.png')
    plt.close()




if __name__ == "__main__":
    main()