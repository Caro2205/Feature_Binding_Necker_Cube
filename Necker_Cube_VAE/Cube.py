'''
Author: Carolin Wengert
carolinwengert@gmail.com
based on code by Tim Gerne (gernetim@gmail.com)
'''

import numpy as np
import pygame
import time
import random
import os
import torch

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
LIGHT_GREEN = (161, 181, 107)
COL_CLOSE = RED
COL_FAR = LIGHT_GREEN
WIDTH = 500
HEIGHT = 500
CORNER_POS = [WIDTH / 2, HEIGHT / 2]  # used to have cube drawn in the middle of the screen
FAC_RAD_TO_DEG = np.pi / 180

################# Definiton of helping vectors that are used to define the corneres of a cube object ###################
base_vectors = np.zeros((0, 3))

base_vectors_lst = [(-1, -1, -1),
                    ( 1, -1, -1),
                    ( 1,  1, -1),
                    (-1,  1, -1),
                    (-1,  1,  1),
                    ( 1,  1,  1),
                    ( 1, -1,  1),
                    (-1, -1,  1)]

for vec in base_vectors_lst:
    base_vectors = np.vstack((base_vectors, vec))



# create cube class
class Cube:
    # defines Cube object by its 8 corners
    # based on https://math.stackexchange.com/questions/107778/simplest-equation-for-drawing-a-cube-based-on-its-center-and-or-other-vertices
    def __init__(self, center, side_length, visibility=[1, 1, 1, 1, 1, 1, 1, 1]):
        self.sidelength = side_length
        self.corners = np.zeros((0, 4))
        self.coords = np.zeros((0, 3))
        self.vis = np.zeros((0, 1))

        for i in range(0, 8):
            vis_array = np.array([visibility[i]])
            self.vis = np.vstack((self.vis, vis_array))

        for vector in base_vectors:
            new_coord = center + (side_length / 2) * vector
            self.coords = np.vstack((self.coords, new_coord))

        self.corners = np.hstack((self.coords, self.vis))

    def delete_corners(self):
        for row in self.corners:
            if row[3] == 0:
                row[0: 3] = 0

    def print_corners(self):
        for corner in self.corners:
            print(corner)

    def print_coords(self):
        for coord in self.coords:
            print(coord)

    def rotate_x(self, theta):
        rotation_matrix_x = np.array([[1, 0, 0],
                                      [0, np.cos(theta * FAC_RAD_TO_DEG), np.sin(theta * FAC_RAD_TO_DEG)],
                                      [0, -np.sin(theta * FAC_RAD_TO_DEG), np.cos(theta * FAC_RAD_TO_DEG)]])
        new_coords = np.zeros((0, 3))
        for coord in self.coords:
            coord_transp = np.array([[x] for x in coord])
            new_coords_transp = np.matmul(rotation_matrix_x, coord_transp)
            new_coord = new_coords_transp.T
            new_coords = np.vstack((new_coords, new_coord))
        self.coords = new_coords
        self.corners = np.hstack((new_coords, self.vis))

    def rotate_y(self, theta):
        rotation_matrix_y = np.array([[np.cos(theta * FAC_RAD_TO_DEG), 0, -np.sin(theta * FAC_RAD_TO_DEG)],
                                      [0, 1, 0],
                                      [np.sin(theta * FAC_RAD_TO_DEG), 0, np.cos(theta * FAC_RAD_TO_DEG)]])
        new_coords = np.zeros((0, 3))
        for coord in self.coords:
            coord_transp = np.array([[x] for x in coord])
            new_coord_transp = np.matmul(rotation_matrix_y, coord_transp)
            new_coord = new_coord_transp.T
            new_coords = np.vstack((new_coords, new_coord))
        self.coords = new_coords
        self.corners = np.hstack((new_coords, self.vis))

    def rotate_z(self, theta):
        rotation_matrix_z = np.array([[np.cos(theta * FAC_RAD_TO_DEG), -np.sin(theta * FAC_RAD_TO_DEG), 0],
                                      [np.sin(theta * FAC_RAD_TO_DEG), np.cos(theta * FAC_RAD_TO_DEG), 0],
                                      [0, 0, 1]])
        new_coords = np.zeros((0, 3))
        for coord in self.coords:
            coord_transp = np.array([[x] for x in coord])
            new_coord_transp = np.matmul(rotation_matrix_z, coord_transp)
            new_coord = new_coord_transp.T
            new_coords = np.vstack((new_coords, new_coord))
        self.coords = new_coords
        self.corners = np.hstack((new_coords, self.vis))


    def print_cube(self, scale=1, title="cube"):
        corner_size = 4
        pygame.init()

        screen = pygame.display.set_mode([WIDTH, HEIGHT])
        pygame.display.set_caption(title)
        screen.fill(WHITE)

        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            screen.fill(WHITE)

            for i in (0, 2, 4, 6):
                adj_corner = 1
                col = COL_CLOSE if self.corners[i][2] > 0 else COL_FAR
                if self.corners[i][3] == 1 and self.corners[i + adj_corner][3] == 1:
                    pygame.draw.line(screen, col, (self.corners[i][0] * scale + CORNER_POS[0],
                                                   self.corners[i][1] * scale + CORNER_POS[1]),
                                     (self.corners[i + adj_corner][0] * scale + CORNER_POS[0],
                                      self.corners[i + adj_corner][1] * scale + CORNER_POS[1]))

            # parallel to y-axis
            for i in (1, 5):
                adj_corner = 1
                col = COL_CLOSE if self.corners[i][2] > 0 else COL_FAR
                if self.corners[i][3] == 1 and self.corners[i + adj_corner][3] == 1:
                    pygame.draw.line(screen, col, (self.corners[i][0] * scale + CORNER_POS[0],
                                                   self.corners[i][1] * scale + CORNER_POS[1]),
                                     (self.corners[i + adj_corner][0] * scale + CORNER_POS[0],
                                      self.corners[i + adj_corner][1] * scale + CORNER_POS[1]))
            for i in (0, 4):
                adj_corner = 3
                col = COL_CLOSE if self.corners[i][2] > 0 else COL_FAR
                if self.corners[i][3] == 1 and self.corners[i + adj_corner][3] == 1:
                    pygame.draw.line(screen, col, (self.corners[i][0] * scale + CORNER_POS[0],
                                                   self.corners[i][1] * scale + CORNER_POS[1]),
                                     (self.corners[i + adj_corner][0] * scale + CORNER_POS[0],
                                      self.corners[i + adj_corner][1] * scale + CORNER_POS[1]))

            # parallel to z-axis
            for i in (0, 1, 2, 3):
                adj_corner = 7
                #col = COL_CLOSE if self.corners[i][2] > 0 else COL_FAR
                if self.corners[i][3] == 1 and self.corners[adj_corner - i][3] == 1:
                    pygame.draw.line(screen, BLACK, (self.corners[i][0] * scale + CORNER_POS[0],
                                                     self.corners[i][1] * scale + CORNER_POS[1]),
                                     (self.corners[adj_corner - i][0] * scale + CORNER_POS[0],
                                      self.corners[adj_corner - i][1] * scale + CORNER_POS[1]))

            for corner in self.corners:
                col = COL_CLOSE if corner[2] > 0 else COL_FAR
                if corner[3] == 1:
                    pygame.draw.circle(screen, col, (corner[0] * scale + CORNER_POS[0],
                                                     corner[1] * scale + CORNER_POS[1]), corner_size, 0)

            pygame.display.update()

        pygame.image.save(screen, 'cube_version_1.png')
        pygame.quit()

    def add_noise(self, intensity):
        new_coords = np.zeros((0, 3))
        for [x, y, z, vis] in self.corners:
            noise_x = np.random.normal(0, 1) * (intensity * self.sidelength)
            noise_y = np.random.normal(0, 1) * (intensity * self.sidelength)
            noise_z = np.random.normal(0, 1) * (intensity * self.sidelength)

            x += noise_x
            y += noise_y
            z += noise_z

            new_coord = np.hstack((x, y, z))
            new_coords = np.vstack((new_coords, new_coord))

        self.coords = new_coords
        self.corners = np.hstack((new_coords, self.vis))

    def delete_all_z(self):
        new_coords = np.zeros((0, 3))
        for [x, y, z, vis] in self.corners:
            new_coord = np.stack((x, y, 0))
            new_coords = np.vstack((new_coords, new_coord))

        self.coords = new_coords
        self.corners = np.hstack((new_coords, self.vis))


def main(data_filename = None, target_filename = None, generate_training_dataset=True):
    if generate_training_dataset:
        target_cubes = []
        data_cubes = []
        center_0 = (0, 0, 0)
        def generate_cubes(n_del_corners = 0, n_cubes = 200, noise = 0, del_z = False, side_len_lower = 0.5, side_len_upper = 3, rand_cor = False, rand_cor_range = 0, noise_freq = 10, input_noise=0):
            # (Tim's BA:) noise >= 0.4 bad for model performance
            rotations = []
            lengths = []

            for j in np.linspace(side_len_lower, side_len_upper, n_cubes):
                lengths.append(j)

            # change range of rotations here if needed
            for k in np.linspace(-180, 180, n_cubes):
                rotations.append(k)

            rotations_x = rotations
            random.shuffle(rotations_x)
            rotations_y = rotations
            random.shuffle(rotations_y)
            rotations_z = rotations
            random.shuffle(rotations_z)
            random.shuffle(lengths)

            for i in range(0, n_cubes, 1):
                visibility = [1, 1, 1, 1, 1, 1, 1, 1]
                cube = Cube(center_0, lengths[i], visibility)
                cube.rotate_x(rotations_x[i])
                cube.rotate_y(rotations_y[i])
                cube.rotate_z(rotations_z[i])
                target_cubes.append(cube)

                if rand_cor: # choose a random amount of corners to delete
                    rand_del_cor = random.sample(range(rand_cor_range), 1)
                    r = rand_del_cor[0]
                    del_corners = random.sample(range(8), r)
                    for j in range(r):
                        visibility[del_corners[j]] = 0
                else: # use pre-defined number of corners to delete
                    del_corners = random.sample(range(8), n_del_corners)
                    for j in range(n_del_corners):
                        visibility[del_corners[j]] = 0



                cube = Cube(center_0, lengths[i], visibility)
                cube.rotate_x(rotations_x[i])
                cube.rotate_y(rotations_y[i])
                cube.rotate_z(rotations_z[i])
                if i % noise_freq == 0 and input_noise != 0:
                    cube.add_noise(input_noise) # intensity=noise_seq[i]
                cube.delete_corners()
                if del_z: cube.delete_all_z()
                data_cubes.append(cube)


        ###### select what cubes to add to the dataset #################################################################

        data_mode = 'training' #'training'
        sl_l = 0.5
        sl_u = 3
        n = 3000
        n_cor = [0, 0]
        g_noise = [0] * len(n_cor)
        z_mis = [False, True]
        r_cor = [False] * len(n_cor)
        corner_range = [0] * len(n_cor)  # random number of corners to delete from 0 to n-1
        inp_noise = 0.1
        noise_f = 3

        for i in range(len(n_cor)):
            generate_cubes(n_cubes=n, n_del_corners=n_cor[i], side_len_lower=sl_l, side_len_upper=sl_u, noise=g_noise[i], del_z=z_mis[i],
                           rand_cor=r_cor[i], rand_cor_range=corner_range[i], noise_freq=noise_f, input_noise=inp_noise)


        ### create training data file
        curr_time = time.strftime("%Y_%m_%d-%H_%M_%S")

        #filename = "data_" + curr_time + ".txt"
        if data_filename is not None:
            filename = data_filename
        else:
            filename = "data.txt"

        path = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Data/' + filename

        try:
            os.remove(path)
        except OSError:
            pass

        with open(path, 'a') as file:
            for cube in data_cubes:
                corner = []
                for [x, y, z, vis] in cube.corners:
                    corner.append(round(x, 5))
                    corner.append(round(y, 5))
                    corner.append(round(z, 5))
                    corner.append(round(vis, 5))
                print(*corner, sep=',', file=file)
        file.close()

        #filename = "target_" + curr_time + ".txt"
        if target_filename is not None:
            filename = target_filename
        else:
            filename = "target.txt"

        path = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Data/' + filename

        try:
            os.remove(path)
        except OSError:
            pass

        with open(path, 'a') as file:
            for cube in target_cubes:
                corner = []
                for [x, y, z, vis] in cube.corners:
                    corner.append(round(x, 5))
                    corner.append(round(y, 5))
                    corner.append(round(z, 5))
                    corner.append(round(vis, 5))
                print(*corner, sep=',', file=file)
        file.close()

        if data_mode == 'training':
            filename = 'training_data_info.txt'
        else:
            filename = 'test_data_info.txt'
        path = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/model_runs/' + filename

        with open(path, "w") as file:
            print('Dataset for: ' + data_mode, file=file)
            print('Total number of cubes in dataset: ', str(len(data_cubes)), file=file)
            print("Number of cubes in one category: ", str(n), file=file)
            print('Range of sidelengths: ', str(sl_l) + '-' + str(sl_u), file=file)
            print('Are number of missing corners chosen randomly', str(r_cor), file=file)
            print('Range of random numbers of deleted corners: ', str(corner_range), file=file)
            print('Number of corners missing in one category: ', str(n_cor), file=file)
            #print('Noise used in one category: ', str(g_noise), file=file)
            print('Noise of: ' + str(inp_noise) + ' is added to every ' + str(noise_f) + 'th cube.', file=file)
            print('z-coordinates missing: ', str(z_mis), file=file)
        file.close()

        #target_cubes[0].print_cube(scale=300)
        #cube_1 = target_cubes[0]
        #cube_1.print_cube(scale=100)
        #cube_2 = target_cubes[1]
        #cube_2.print_cube(scale=100)
        print('total number of cubes in the dataset: ' + str(len(data_cubes)))
        print(str(len(target_cubes)))
    else:
        ########### generate input cubes for framework
        # creates one cube based on given rotation angles and the opposite observation
        input_cubes = []

        def generate_cubes_input_framework(rot_x, rot_y, rot_z):
            visibility = [1, 1, 1, 1, 1, 1, 1, 1]

            cube1 = Cube((0, 0, 0), 1, visibility)
            cube1.rotate_x(rot_x)
            cube1.rotate_y(rot_y)
            cube1.rotate_z(rot_z)
            input_cubes.append(cube1)

            cube2 = Cube((0, 0, 0), 1, visibility)
            cube2.rotate_x(-rot_x)
            cube2.rotate_y(-rot_y)
            cube2.rotate_z(rot_z)
            input_cubes.append(cube2)


        generate_cubes_input_framework(92, 56, 5)
        filename = 'input_cubes_framework'

        path = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Data/' + filename + '.txt'
        with open(path, 'a') as file:
            for cube in input_cubes:
                corner = []
                for [x, y, z, vis] in cube.corners:
                    corner.append(round(x, 5))
                    corner.append(round(y, 5))
                    corner.append(round(z, 5))
                    corner.append(round(vis, 5))
                print(*corner, sep=',', file=file)
        file.close()

        corners_tensor = torch.tensor([])  # Create an empty tensor

        with open(path, 'a') as file:
            for cube in input_cubes:
                cube_tensor = torch.tensor([])
                for [x, y, z, vis] in cube.corners:
                    corner = torch.tensor([round(x, 5), round(y, 5), round(z, 5)]) # , round(vis, 5)
                    cube_tensor = torch.cat((cube_tensor, corner.unsqueeze(0)), dim=0)
                corners_tensor = torch.cat((corners_tensor, cube_tensor.unsqueeze(0)), dim=0)
            print(corners_tensor.shape)
            print(corners_tensor)
            torch.save(corners_tensor, 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Data/' + filename + '.pt')


if __name__ == "__main__":
    #generate_training_dataset: generate training and target dataset or testing and target dataset
    #if false: generate only input cubes for the framework
    #data filenames are only for generate_training_dataset
    #for input cubs rotation angles and filename have to be set in the code
    main(data_filename='framework_input.txt', target_filename='training_target.txt', generate_training_dataset=False)


    # cube_1 = Cube((0, 0, 0), 1, [1, 1, 1, 1, 1, 1, 1, 1])
    # cube_1.rotate_x(20)
    # cube_1.rotate_y(30)
    # cube_1.print_cube(scale=300)

    #cube_2 = Cube((0, 0, 0), 1, [1, 1, 1, 1, 1, 1, 1, 1])
    #cube_2.rotate_x(-20)
    #cube_2.rotate_y(-30)
    #cube_2.print_cube(scale=300)
