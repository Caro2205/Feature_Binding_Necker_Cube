

import os
import time
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pygame
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary


import VAE_model as VAE
import VAE_model_small
import VAE_model_small as VAE_small
import VAE_model_large as VAE_large

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
COL_CLOSE = RED
COL_FAR = (161, 181, 107)
WIDTH = 500
HEIGHT = 500
CORNER_POS = [WIDTH / 2, HEIGHT / 2]

HID_COORD = 0

class CubeData(Dataset):
    def __init__(self, datapath, has_vis_marker, targetpath=None):
        self.has_vis_marker = has_vis_marker

        xy = np.loadtxt(datapath, delimiter=',', dtype=np.float32)

        if targetpath is None:
            xy = np.loadtxt(datapath, delimiter=',', dtype=np.float32)
            x = np.copy(xy)  # input for network
            y = np.copy(xy)  # desired network output
        else:
            x_data = np.loadtxt(datapath, delimiter=',', dtype=np.float32)
            y_data = np.loadtxt(targetpath, delimiter=',', dtype=np.float32)
            x = np.copy(x_data)  # input for network
            y = np.copy(y_data)  # desired network output

        if targetpath is None:
            for row in x:
                for i in range(3, 32, 4):  # replaces x,y,z for corners when vis = 0
                    if row[i] == 0:
                        row[i - 1] = HID_COORD
                        row[i - 2] = HID_COORD
                        row[i - 3] = HID_COORD

        if not self.has_vis_marker:  # deletes vis value for every corner
            x = np.delete(x, list(range(3, x.shape[1], 4)), axis=1)

        self.x = torch.from_numpy(x)  # contains vis information or not depending on has_vis_marker
        y = np.delete(y, list(range(3, y.shape[1], 4)), axis=1)  # deletes vis value for every corner
        self.y = torch.from_numpy(y)  # contains only coordinates without visibility information, for all corners

        self.n_samples = xy.shape[0]  # number of cubes in dataset

    def __getitem__(self, index):
        return self.x[index], self.y[index]  # network input, desired network output

    def __len__(self):
        return self.n_samples


def train_model(model, dataloader, n_epochs, criterion, optimizer, outputs, epoch_losses, reconstruction_losses,
                epoch_idx):
    for epoch in range(n_epochs):
        epoch_idx += 1
        running_loss = 0
        running_rec_loss = 0
        for corners, coordinates in dataloader:
            reconstruction, mu, sigma = model(corners)

            kl_divergence = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            # https://sannaperzon.medium.com/paper-summary-variational-autoencoders-with-pytorch-implementation-1b4b23b1763a
            reconstruction_loss = criterion(reconstruction, coordinates) * coordinates.shape[0] * coordinates.shape[1]  # batch_size * n_values for one cube -> = reduction='sum'
            #reconstruction_loss = criterion(reconstruction, coordinates) * coordinates.shape[0] # *batch_size
            loss = reconstruction_loss + 0.5 * kl_divergence
            running_loss += loss.item()
            running_rec_loss += reconstruction_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #n_datapoints = len(dataloader) * dataloader.batch_size
        n_cubes = len(dataloader.sampler)
        epoch_loss = running_loss / n_cubes
        epoch_rec_loss = running_rec_loss / n_cubes

        print('Epoch:', str(epoch_idx))
        print('Epoch Training Loss:', str(epoch_loss))
        print('Reconstruction Loss:', str(epoch_rec_loss))

        reconstruction_losses.append(epoch_rec_loss)
        epoch_losses.append(epoch_loss)

        outputs.append(
            (epoch, corners, reconstruction))  # speichert den letzten batch an cube/reconstructions ab

    return outputs, epoch_losses, reconstruction_losses


def validate_model(model, dataloader, n_epochs, criterion, validation_outputs, validation_losses):
    running_loss = 0
    with torch.no_grad():
        for corners, coordinates in dataloader:
            reconstruction, mu, sigma = model(corners, "testing")
            validation_outputs.append(reconstruction)

            MSE = criterion(reconstruction, coordinates)
            root_MSE = torch.sqrt(MSE)
            root_MSE_sum = torch.sum(root_MSE)
            running_loss += root_MSE_sum #* coordinates.shape[0] # * batch_size

        loss = running_loss / (len(dataloader.sampler) * 24)     # / (größe validation datensatz * anzahl Koordinaten)
        validation_losses.append(loss.item())

        return validation_outputs, validation_losses



def save_loss_plot(losses, title, x_label, y_label, path, n_epochs):
    plt.style.use('fivethirtyeight')
    plt.xticks(range(0, n_epochs, 200))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.plot(losses)
    plt.savefig(path+".pdf", bbox_inches='tight')
    # plt.show()
    plt.close()


def coordinate_reformer(cube):
    reform_cube = np.zeros((0, 3))
    for i in range(0, 22, 3):
        coord = cube[i:i + 3]  # 3 consecutive values (x, y, z of one coordinate)
        reform_cube = np.vstack((reform_cube, coord))

    return reform_cube


def full_corner_reformer(cube):
    reform_cube = np.zeros((0, 4))
    for i in range(0, 30, 4):
        coord = cube[i:i + 4]
        reform_cube = np.vstack((reform_cube, coord))

    return reform_cube


def draw_cube(corner_list, scale, corner_size, screen, has_vis_marker):
    close_corners, far_corners = check_corners_close(corner_list)

    # parallel to x-axis
    for i in (0, 2, 4, 6):
        adj_corner = 1
        # if not has_vis_marker or (corner_list[i][3] == 1 and corner_list[i + 4][3] == 1):
        if not (corner_list[i][0] == corner_list[i][1] == corner_list[i][2] == HID_COORD) and \
                not (corner_list[i + adj_corner][0] == corner_list[i + adj_corner][1] == corner_list[i + adj_corner][
                    2] == HID_COORD):
            col = COL_CLOSE if np.any(np.all(corner_list[i] == close_corners, axis=1)) else COL_FAR
            pygame.draw.line(screen, col, (corner_list[i][0] * scale + CORNER_POS[0],
                                           corner_list[i][1] * scale + CORNER_POS[1]),
                             (corner_list[i + adj_corner][0] * scale + CORNER_POS[0],
                              corner_list[i + adj_corner][1] * scale + CORNER_POS[1]))

    # parallel to y-axis
    for i in (1, 5):
        adj_corner = 1
        # if not has_vis_marker or (corner_list[i][3] == 1 and corner_list[i + 2][3] == 1):
        if not (corner_list[i][0] == corner_list[i][1] == corner_list[i][2] == HID_COORD) and \
                not (corner_list[i + adj_corner][0] == corner_list[i + adj_corner][1] == corner_list[i + adj_corner][
                    2] == HID_COORD):
            col = COL_CLOSE if np.any(np.all(corner_list[i] == close_corners, axis=1)) else COL_FAR
            pygame.draw.line(screen, col, (corner_list[i][0] * scale + CORNER_POS[0],
                                           corner_list[i][1] * scale + CORNER_POS[1]),
                             (corner_list[i + adj_corner][0] * scale + CORNER_POS[0],
                              corner_list[i + adj_corner][1] * scale + CORNER_POS[1]))
    for i in (0, 4):
        adj_corner = 3
        # if not has_vis_marker or (corner_list[i][3] == 1 and corner_list[i + 2][3] == 1):
        if not (corner_list[i][0] == corner_list[i][1] == corner_list[i][2] == HID_COORD) and \
                not (corner_list[i + adj_corner][0] == corner_list[i + adj_corner][1] == corner_list[i + adj_corner][
                    2] == HID_COORD):
            col = COL_CLOSE if np.any(np.all(corner_list[i] == close_corners, axis=1)) else COL_FAR
            pygame.draw.line(screen, col, (corner_list[i][0] * scale + CORNER_POS[0],
                                           corner_list[i][1] * scale + CORNER_POS[1]),
                             (corner_list[i + adj_corner][0] * scale + CORNER_POS[0],
                              corner_list[i + adj_corner][1] * scale + CORNER_POS[1]))

    # parallel to z-axis
    for i in (0, 1, 2, 3):
        adj_corner = 7
        # if not has_vis_marker or (corner_list[i][3] == 1 and corner_list[i + 1][3] == 1):
        if not (corner_list[i][0] == corner_list[i][1] == corner_list[i][2] == HID_COORD) and \
                not (corner_list[adj_corner - i][0] == corner_list[adj_corner - i][1] == corner_list[adj_corner - i][
                    2] == HID_COORD):
            pygame.draw.line(screen, BLACK, (corner_list[i][0] * scale + CORNER_POS[0],
                                             corner_list[i][1] * scale + CORNER_POS[1]),
                             (corner_list[adj_corner - i][0] * scale + CORNER_POS[0],
                              corner_list[adj_corner - i][1] * scale + CORNER_POS[1]))

    for corner in corner_list:
        # if not has_vis_marker or corner[3] == 1:
        if not (corner[0] == corner[1] == corner[2] == HID_COORD):
            # col = COL_CLOSE if corner[2] > 0 else COL_FAR
            # col = COL_CLOSE if corner in close_corners else COL_FAR
            col = COL_CLOSE if np.any(np.all(corner == close_corners, axis=1)) else COL_FAR
            pygame.draw.circle(screen, col, (corner[0] * scale + CORNER_POS[0],
                                             corner[1] * scale + CORNER_POS[1]), corner_size, 0)


# displays a cube from the dataset or a reconstructed cube. Reconstruction parameter has to be set (False by default)
def display_cube(corner_list, scale=100, title="cube", has_vis_marker=True):
    corner_size = 4
    pygame.init()

    screen = pygame.display.set_mode([WIDTH, HEIGHT])
    pygame.display.set_caption(title)

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        screen.fill(WHITE)

        draw_cube(corner_list, scale, corner_size, screen, has_vis_marker)

        pygame.display.update()

    pygame.quit()


def save_images(cube, reconstruction, path, has_vis_marker, mode=None):
    cube = cube.detach().numpy()
    reconstruction = reconstruction.detach().numpy()

    cube = full_corner_reformer(cube) if has_vis_marker else coordinate_reformer(cube)
    reconstruction = coordinate_reformer(reconstruction)

    s1 = pygame.Surface((WIDTH, HEIGHT))
    s2 = pygame.Surface((WIDTH, HEIGHT))
    s3 = pygame.Surface((2 * WIDTH, HEIGHT))

    s1.fill(WHITE)
    s2.fill(WHITE)
    s3.fill(WHITE)

    draw_cube(cube, 100, 4, s1, has_vis_marker)
    draw_cube(reconstruction, 100, 4, s2, has_vis_marker=False)

    s3.blit(s1, (0, 0))
    s3.blit(s2, (WIDTH, 0))
    pygame.draw.line(s3, BLACK, (WIDTH, 0), (WIDTH, HEIGHT))

    if mode == "rec_only":
        pygame.image.save(s2, path)
    elif mode == "original_only":
        pygame.image.save(s1, path)
    else:
        pygame.image.save(s3, path)


def create_folders(curr_time):
    foldername = "run_" + curr_time  # folder for this the model run
    folderdir = "C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/model_runs/" + foldername + "/"  # this run in general run folder
    validation_imagedir = folderdir + "validation_images/"  # folder for this run's images
    validation_coorddir = folderdir + "validation_coords/"
    testimagedir = folderdir + "test_images/"
    os.makedirs(validation_imagedir)
    os.mkdir(validation_coorddir)
    os.mkdir(testimagedir)

    return folderdir, validation_imagedir, validation_coorddir, testimagedir


def save_model_info(has_vis_marker, filename, n_epochs, training_batch_size, learning_rate, weight_decay, datapath,
                    trainingdata_size,
                    epoch_losses, rec_losses_train, testdata_size, rec_loss_test):
    with open(filename, "w") as file:
        print("Marker for visibility of corner was used:", str(has_vis_marker), file=file)
        print("Number Epochs:", str(n_epochs), file=file)
        print("Training Batch Size:", str(training_batch_size), file=file)
        print("Learning Rate:", str(learning_rate), file=file)
        print("Weight Decay:", str(weight_decay), file=file)
        print("Training Data:", datapath, file=file)
        print("\n", file=file)
        print("Size Trainingdata:", str(trainingdata_size), file=file)
        print("Loss last epoch:", str(epoch_losses[len(epoch_losses) - 1]), file=file)
        print("Reconstruction loss last epoch:", str(rec_losses_train[len(rec_losses_train) - 1]), file=file)
        print("\n", file=file)
        print("Size Testingdata:", (str(testdata_size)), file=file)
        print("Test Reconstruction Loss:", str(rec_loss_test), file=file)
    file.close()


def test_model(model, dataloader, criterion):
    test_losses = []
    test_coords = []
    #image_path = path + "/test_data_image_reconstructions"

    for corners, coordinates in dataloader:
        reconstruction, mu, sigma = model(corners, mode="testing")

        MSE = criterion(reconstruction, coordinates)
        root_MSE = torch.sqrt(MSE)
        #################################
        root_MSE_sum = torch.sum(root_MSE)
        root_MSE_av = root_MSE_sum/24
        test_losses.append(root_MSE_av.item())

        # # reconstruct images and coordinates
        # save_images(corners, reconstruction, image_path,  has_vis_marker=has_vis_marker)
        # #todo: save_coordinates()
        test_coords.append(reconstruction)

    return test_losses, test_coords


def check_corners_close(corner_list):
    sorted_corners = np.copy(corner_list)
    # print(sorted_corners)
    x = sorted_corners[sorted_corners[:, 2].argsort()]
    close_corners = x[:4]
    far_corners = x[4:]

    return close_corners, far_corners


# saves reconstructions: images and coordinates (if save_coords=True)
def save_outputs(model, indices, dataset, imagedir, coorddir, folderdir, epochs, has_vis_marker,
                 save_coords):
    coords = []
    os.mkdir(imagedir + "epoch_" + str(epochs))
    indices_sorted = []
    indices_sorted = indices.copy()
    indices_sorted.sort()

    for i in indices_sorted:
        cube, _ = dataset.__getitem__(i)
        output, _, _ = model(cube, mode="testing")

        filename = "cube_" + str(i + 1)
        path = imagedir + "epoch_" + str(epochs) + "/" + filename + ".png"

        save_images(cube, output, path, has_vis_marker, mode=None)
        coords.append(coordinate_reformer(output.detach().numpy()))

    if save_coords:
        save_coordinates(coorddir, epochs, indices_sorted, coords)


def save_coordinates(coorddir, epochs, indices_sorted, coords):
    filename = coorddir + "/output_coord_epoch_" + str(epochs) + ".txt"
    for i, test_index in enumerate(indices_sorted):
        with open(filename, 'a') as f:
            print("Cube_", str(test_index + 1), file=f)
            print(coords[i], file=f)
            print("\n", file=f)
        f.close()


def seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# stores seed, and test_loss to make comparisons possible
def store_seed(path, test_losses, curr_time):
    with open(path, 'a') as f:
        print("Seed: " + str(seed), "Running time: " + curr_time, file=f)
        print(test_losses, file=f)
        print(file=f)
    f.close()


def main():
################################### parameters have to be set here #####################################################
    # Hyperparameters
    learning_rate = 1e-4  # 1e-3
    weight_decay = 1e-4  # 1e-4
    n_epochs = 2000 #1500
    train_batch_size = 60 # 40
    validation_batch_size = 60
    n_save_outputs = 500 # at every xth epoch, the outputs are saved

    # Paths for datasets
    datapath = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Data/data.txt'  # data used for training (model input)
    targetpath = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Data/target.txt'  # can be None     # data used for training (desired model output)

    test_datapath = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Data/data.txt'  # data used to see how the model performs on that data after certain training epochs
    test_targetpath = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Data/target.txt'  #'./training_data/testing_target.txt'

    # has_vis_marker indicates if the used data has a marker (1 or 0) to indicate whether a corner's
    # coordinate is visible or not visible (-> x, y, z = 0)
    has_vis_marker = False  # False
    input_size = 8 * 4 if has_vis_marker else 8 * 3

    # define model
    # model variable defines which VAE is used

    #model = VAE_model_small.VariationalAutoencoder(input_size=input_size)
    model = VAE.VariationalAutoencoder(input_size=input_size)         # normal
    #model = VAE_large.VariationalAutoencoder(input_size=input_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
 #######################################################################################################################

    # load data and create dataloader for training and testing

    # dataset = CubeData(datapath, has_vis_marker)
    dataset = CubeData(datapath, has_vis_marker, targetpath)

    # indices decide which cubes are used for training and validation set -> are shuffeled
    dataset_size = dataset.__len__()
    print('Dataset size: ' + str(dataset_size))
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor(0.2 * dataset_size))
    train_indices, validation_indices = indices[split:], indices[:split]

    # sampler give indices used for training and sampling
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(validation_indices)
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=train_batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=validation_batch_size, sampler=valid_sampler)

    # initialization of test dataset -> is only used after training is complete
    if test_datapath is not None:
        test_dataset = CubeData(test_datapath, has_vis_marker, test_targetpath)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset)
        testimages_idx = list(range(0, test_dataset.__len__()))
        # for i in range(test_dataset.__len__()):
        #     testimages_idx.append(i)


    # create folders for model
    curr_time = time.strftime("%Y_%m_%d-%H_%M_%S")
    folderdir, imagedir, coorddir, testimagedir = create_folders(curr_time)

    # run model and saves image reconstructions of validation images in every epoch in print_epoch
    outputs = []
    epoch_losses = []
    reconstruction_losses = []

    validation_losses = []
    validation_outputs = []

    validation_crit = nn.MSELoss(reduction='none')

    # epoch_steps = []
    # for i in range(10):
    #     epoch_steps.append(i)
    #
    # for i in range(10, n_epochs, 10):
    #     epoch_steps.append(i)

    # training method is called and validation is conducted
    for i in range(n_epochs):
        if i == 1000:
            learning_rate = 1e-5
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        n_training_epochs = 1
        #n_training_epochs = 1 if i < 10 else 10
        if i % n_save_outputs == 0 & i != 0:  #at which points to save images
            save_outputs(model, validation_indices, dataset, imagedir, coorddir, folderdir, i, has_vis_marker, True)        # validation dataset
            # sollte erst am Ende passieren, da test dataset
            save_outputs(model, testimages_idx, test_dataset, testimagedir, coorddir, folderdir, i, has_vis_marker, False)  # test dataset

        # model is trained for n_training_epochs epochs -> in total for n_epochs epochs
        outputs, epoch_losses, reconstruction_losses = train_model(model, train_loader, n_training_epochs, criterion, optimizer,
                                                                   outputs, epoch_losses, reconstruction_losses,
                                                                   i)

        # use validation dataset to calculate root mse
        validation_outputs, validation_losses = validate_model(model, validation_loader, n_epochs, validation_crit, validation_outputs, validation_losses)




    # print losses
    save_loss_plot(epoch_losses, 'Average loss in each epoch', 'Iterations', 'Loss', folderdir + "epoch_losses", n_epochs)
    save_loss_plot(reconstruction_losses, 'Average reconstruction loss in each epoch', 'Iterations', 'Loss',
                   folderdir + "rec_losses", n_epochs)
    save_loss_plot(validation_losses, "Reconstruction RMSE of validation data", 'Iterations', 'Loss', folderdir + "validation_losses", n_epochs)

    # saving model parameters
    model_name = folderdir + "saved_model_parameters.pt"
    torch.save(model.state_dict(), model_name)

############## training and validation are completed ##############


    # reloading the model parameters
    #model = VAE.VariationalAutoencoder(input_size=input_size)
    model.load_state_dict(torch.load(model_name))
   # summary(model, input_size = (3, 64, 64), batch_size = -1)
    print("Model:")
    print(model)

    # test model
    # test_losses = test_model(model, validation_loader, nn.MSELoss())
    # print("\nTest MSE Reconstruction Losses:", str(test_losses))

    # save test reconstruction images and coordinates
    if test_datapath is not None:
        save_outputs(model, testimages_idx, test_dataset, testimagedir, coorddir, folderdir, "final", has_vis_marker, False)  # test dataset
        test_criterion = nn.MSELoss(reduction='none')
        test_losses, test_coords = test_model(model, test_loader, test_criterion)

        filename = folderdir + "/test_losses.txt"
        with open(filename, "w") as f:
            for i, loss in enumerate(test_losses):
                print(i + 1, loss, file=f)
        f.close()
        # save_loss_plot(test_losses, "Reconstruction RMSE of test data", 'Iterations', 'Loss',
        #                folderdir + "test_losses", len(test_dataset))

        filename = folderdir + "/test_coords_final.txt"
        with open(filename, "w") as f:
            for i, loss in enumerate(test_coords):
                print("Test Cube:", str(i + 1), loss, file=f)
        f.close()

    else:
        test_losses = None

    save_outputs(model, validation_indices, dataset, imagedir, coorddir, folderdir, "final", has_vis_marker,
                 True)  # validation dataset

    # save model information
    filename = folderdir + "/Hyperparameter.txt"
    n_traindata = dataset_size - split
    n_testdata = split
    save_model_info(has_vis_marker, filename, n_epochs, train_batch_size, learning_rate, weight_decay,
                    datapath, n_traindata, epoch_losses, reconstruction_losses, n_testdata, test_losses)

    filename = folderdir + "/epoch_losses.txt"
    with open(filename, "w") as f:
        for i, loss in enumerate(epoch_losses):
            print(i + 1, loss, file=f)
    f.close()

    filename = folderdir + "/reconstruction_losses.txt"
    with open(filename, "w") as f:
        for i, loss in enumerate(reconstruction_losses):
            print(i + 1, loss, file=f)
    f.close()


    filename = folderdir + "/validation_losses.txt"
    with open(filename, "w") as f:
        for epoch, loss in enumerate(validation_losses):
            print(epoch+1, loss, file=f)
    f.close()
    # save seed and test MSE to ensure responsibility
    # seed and test MSE are added to existing file in main directory
    path = "./check_reproducibility_with_vis_marker.txt" if has_vis_marker \
        else "./check_reproducibility_without_vis_marker.txt "
    store_seed(path, test_losses, curr_time)


if __name__ == "__main__":
    seed = 0
    seeding(seed)
    main()

    #for i in range(10):
    #    seed = i
    #    seeding(seed)
    #    main()

    # seed = 1
    # seeding(seed)
    # main()
