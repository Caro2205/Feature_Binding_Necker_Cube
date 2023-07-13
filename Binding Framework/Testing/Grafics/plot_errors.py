import matplotlib.pyplot as plt
import numpy as np
import os

def main():

    #datetime = '2023_Jul_03-21_56_56'
    #datetime = 'ganz gutes ergebnis'
    datetime = 'erster run'
    ore = 0.00107966
    ore_without_z = 0.00092452
    dir = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Python Projects/Code/Binding Framework/Testing/Grafics/'
    folder_path = dir + 'binding_test_results_necker_cube_static/' + datetime + '/b_dimensions_3/necker_cube_static_0/'

    reconstruction_losses = np.loadtxt(folder_path + 'reconstruction_losses.txt')
    rec = reconstruction_losses[:, 1]

    feature_binding_losses = np.loadtxt(folder_path + 'feature_binding_losses.txt')
    bind = feature_binding_losses[:, 1]

    plt.plot(rec, color='r')
    plt.axhline(ore, color='blue', label='ORE')
    plt.axhline(ore_without_z, color='green', label='ORE without z')
    plt.xlabel('Tuning Cycle')
    plt.ylabel('Reconstruction Loss')
    #plt.yscale('log')
    plt.legend()
    plt.savefig(folder_path + 'reconstruction_losses.png')
    plt.close()

    plt.plot(rec, color='r')
    plt.axhline(ore, color='blue', label='ORE')
    plt.axhline(ore_without_z, color='green', label='ORE without z')
    plt.xlabel('Tuning Cycle')
    plt.ylabel('Reconstruction Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(folder_path + 'reconstruction_losses_log.png')
    plt.close()

    plt.plot(bind, color='r')
    plt.xlabel('Tuning Cycle')
    plt.ylabel('Feature Binding Loss')
    plt.savefig(folder_path + 'feature_binding_losses.png')

if __name__ == "__main__":
    main()