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


def main(data_filename = None, target_filename = None):
    target_cubes = []
    data_cubes = []
    center_0 = (0, 0, 0)
    def generate_cubes(n_del_corners = 0, n_cubes = 200, noise = 0, del_z = False, side_len_lower = 0.5, side_len_upper = 3):
        #noise_intens = []  # (Tim's BA:) noise >= 0.4 bad for model performance
        rotations = []
        lengths = []

        # for i in np.linspace(0, 0.4, n_cubes):
        #    noise_intens.append(i)

        noise_seq = [noise] * n_cubes

        for i in np.linspace(side_len_lower, side_len_upper, n_cubes):
            lengths.append(i)

        # change range of rotations here if needed
        for i in np.linspace(-180, 180, n_cubes):
            rotations.append(i)

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

            del_corners = random.sample(range(8), n_del_corners)
            for j in range(n_del_corners):
                visibility[del_corners[j]] = 0
            cube = Cube(center_0, lengths[i], visibility)
            cube.rotate_x(rotations_x[i])
            cube.rotate_y(rotations_y[i])
            cube.rotate_z(rotations_z[i])
            cube.add_noise(intensity=noise_seq[i])
            cube.delete_corners()
            if del_z: cube.delete_all_z()
            data_cubes.append(cube)

    ###### select what cubes to add to the dataset #################################################################
    # n_cubes standardmäßig 200
    data_mode = 'testing' #'training'
    sl_l = 1
    sl_u = 1
    n = 20
    n_cor = [0, 1, 2]
    g_noise = [0]
    z_mis = False

    generate_cubes(n_del_corners=n_cor[0], side_len_lower=sl_l, side_len_upper=sl_u, n_cubes=n)
    generate_cubes(n_del_corners=n_cor[1], side_len_lower=sl_l, side_len_upper=sl_u, n_cubes=n)
    generate_cubes(n_del_corners=n_cor[2], side_len_lower=sl_l, side_len_upper=sl_u, n_cubes=n)


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

    filename = 'dataset_info.txt'
    path = 'C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/model_runs/' + filename

    with open(path, "w") as file:
        print('Dataset for: ' + data_mode, file=file)
        print('Total number of cubes in dataset: ', str(len(data_cubes)), file=file)
        print("Number of cubes in one category: ", str(n), file=file)
        print('Range of sidelengths: ', str(sl_l) + '-' + str(sl_u), file=file)
        print('Number of corners missing in one category: ', str(n_cor), file=file)
        print('Noise used in one category: ', str(g_noise), file=file)
        print('z-coordinates missing: ', str(z_mis), file=file)
    file.close()

    #target_cubes[0].print_cube(scale=300)
    cube_1 = target_cubes[0]
    cube_1.print_cube(scale=100)
    cube_2 = target_cubes[1]
    cube_2.print_cube(scale=100)
    print('total number of cubes in the dataset: ' + str(len(data_cubes)))

if __name__ == "__main__":
    main(data_filename='test_data.txt', target_filename='test_target.txt')


    # cube_1 = Cube((0, 0, 0), 1, [1, 1, 1, 1, 1, 1, 1, 1])
    # cube_1.rotate_x(20)
    # cube_1.rotate_y(30)
    # cube_1.print_cube(scale=300)

    #cube_2 = Cube((0, 0, 0), 1, [1, 1, 1, 1, 1, 1, 1, 1])
    #cube_2.rotate_x(-20)
    #cube_2.rotate_y(-30)
    #cube_2.print_cube(scale=300)
