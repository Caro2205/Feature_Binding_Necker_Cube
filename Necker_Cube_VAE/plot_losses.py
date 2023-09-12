import matplotlib.pyplot as plt
import numpy as np
import os

########################################################################################################################
# plot filtered error of multiple inputs/runs
########################################################################################################################

def main():
    all_losses = []
    last_val_losses = []
    fig, ax = plt.subplots()

    #C:\Users\49157\OneDrive\Dokumente\UNI\8. Semester\Bachelorarbeit\Python Projects\Code\Binding Framework\Testing\Grafics\binding_test_results_necker_cube_static\test_with_turn_up_inconsistent
    folder_name = '10 runs large model'
    dir =  'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/model_runs/' + folder_name

    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

    for i in range(10):

        folder_path = dir + '/0' + str(i) + '/'

        rec_losses = np.loadtxt(folder_path + 'reconstruction_losses.txt')
        rec_losses = rec_losses[:,1]

        all_losses.append(rec_losses)

        val_losses = np.loadtxt(folder_path + 'validation_losses.txt')
        val_losses = val_losses[:,1]
        last_val_loss = val_losses[-1]
        last_val_losses.append(last_val_loss)

        ax.plot(rec_losses, color=colors[i], label=str(i))

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_yscale('log')
    #ax.set_xlim(0, 2000)

    plt.legend(loc='upper right')

    plt.savefig(dir + '/lossss_plot.png')
    plt.close()

    mean_loss = np.mean(all_losses, axis=0)
    std = np.std(all_losses, axis=0)
    print('standard deviation:')
    print(std)


    #mean_loss = mean_loss[-1700:]
    #std = std[-1700:]

    fig, ax = plt.subplots()
    ax.plot(mean_loss, color='#006f61', label='mean RMSE')
    #ax.plot(mean_loss + std, color='b')
    #ax.plot(mean_loss - std, color='b')
    #ax.fill_between(range(len(mean_loss)), mean_loss - std, mean_loss + std, color='b', alpha=0.3, label='standard deviation')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    ax.set_ylabel('Reconstruction Error')
    ax.legend()
    plt.savefig(dir + '/large.png')
    plt.close()


    mean_loss = mean_loss[-1050:]
    std = std[-1050:]
    fig, ax = plt.subplots()
    ax.plot(mean_loss, color='#006f61', label='mean RMSE')
    ax.fill_between(range(len(mean_loss)), mean_loss - std, mean_loss + std, color='#35ac9d', alpha=0.3, label='standard deviation')
    #ax.set_yscale('log')
    #ax.set_xticks([0, 250, 500, 750, 1000, 1250, 1500, 1750]) # for small model
    ax.set_xticks([0,  150,  300,  450 , 600,  750 , 900, 1050])
    #ax.set_xticklabels(['250', '500', '750', '1000', '1250', '1500', '1750', '2000']) # for small model
    ax.set_xticklabels(['950', '1100', '1250', '1400', '1550', '1700', '1850', '2000'])
    ax.set_xlabel('Epoch')
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    ax.set_ylabel('Reconstruction Error')
    ax.legend()
    plt.savefig(dir + '/large_last.png')
    plt.close()

    test = np.round(np.mean(last_val_losses), decimals=4)

    filename = dir + '/mean_last_validation_loss.txt'
    with open(filename, "w") as f:
        print(test, file=f)
    f.close()

if __name__ == "__main__":
    main()