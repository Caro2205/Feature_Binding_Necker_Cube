"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""

import pygame
import pygame.camera
import os 
import torch
import sys

import sys
sys.path.append('C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/BindingAndPerspectiveTaking')
# Before run: replace ... with current directory path


class SKEL_RENDERER():

    """
        Class to visualize skeleton motion by method render().

    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        os.environ["SDL_VIDEO_CENTERED"]='1'
        self.black, self.white, self.red, self.blue, self.green = (0,0,0), (250,250,250), (255,0,0), (0,0,255), (0,255,0)
        self.width, self.height = 1500, 1500

        self.hard_add_joint = False


    def connection(self, j1, j2, skeleton): 
        p1 = skeleton[j1]
        p2 = skeleton[j2]
        pygame.draw.line(
            self.screen, 
            self.black, 
            (p1[0], p1[1]), 
            (p2[0], p2[1]), 
            2)


    def direction_arrow(self, joint, dir_pont, mag): 
        pygame.draw.line(
            self.screen, 
            self.green, 
            (joint[0], joint[1]), 
            (dir_pont[0], dir_pont[1]),
            mag
            )


    def render(self, position, direction, magnitude, gestalt):
        """
            Visualize motion of skeleton described by given parameters: 
                - position: tensor of shape (time steps, num_features, 3)
                    position of skeleton features in every time step
                - direction: tensor of shape (time steps, num_features, 3)
                    direction or velocity of features in every time step
                - magnitude: tensor of shape (time steps, num_features, 1)
                    magnitude of motion of features in every time step
                - gestalt: bool 
                    indicating whether the data has values for direction and magnitude. 
                    If data does not have more than positional data, set gestalt = False.

        """

        pygame.init()

        shot_cnt = 0
        shot_freq = 5


        pygame.display.set_caption("3D skeleton")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.fps = 30

        run = True
        frame_cnt = 0
        num_frames = position.size()[0]
        num_features = position.size()[1]
        up_scale = 500

        while run: 
            self.clock.tick(self.fps)

            c_frame_pos = position[frame_cnt] *up_scale
            if gestalt:
                c_frame_mag = magnitude[frame_cnt]
                c_frame_dir = direction[frame_cnt] *up_scale

            self.screen.fill(self.white)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    run = False
            center = (self.width/2,self.height/2)
            proj_skeleton = torch.tensor([])
            for i in range(num_features):
                joint = c_frame_pos[i].to(self.device)
                if gestalt:
                    mag_arrow = c_frame_mag[i].to(self.device)
                    dir_arrow = c_frame_dir[i].to(self.device) *mag_arrow *10

                z = 1
                projection_matrix = torch.Tensor([
                    [z,0,0], 
                    [0,-z,0]]).to(self.device)
                
                projected_joint = torch.matmul(projection_matrix, joint)
                x_j = int(projected_joint[0] + center[0])
                y_j = int(projected_joint[1] + center[1])
                projected_joint = torch.tensor([x_j, y_j])

                if gestalt:
                    dir_point = torch.matmul(projection_matrix, dir_arrow)
                    x_d = int(projected_joint[0] + dir_point[0])
                    y_d = int(projected_joint[1] + dir_point[1])
                    dir_point = torch.tensor([x_d, y_d])

                    self.direction_arrow(projected_joint, dir_point, 5)

                pygame.draw.circle(self.screen, self.blue, (x_j, y_j), 10)
                proj_skeleton = torch.cat([proj_skeleton, projected_joint.view(1,2)])
            
            proj_skeleton = proj_skeleton.int()

            if num_features == 15: 
                self.connection(0,1, proj_skeleton)
                self.connection(1,2, proj_skeleton)
                self.connection(3,4, proj_skeleton)
                self.connection(4,5, proj_skeleton)

                self.connection(0,6, proj_skeleton)
                self.connection(3,6, proj_skeleton)

                self.connection(6,7, proj_skeleton)
                self.connection(7,8, proj_skeleton)

                self.connection(7,9, proj_skeleton)
                self.connection(9,10, proj_skeleton)
                self.connection(10,11, proj_skeleton)
                self.connection(7,12, proj_skeleton)
                self.connection(12,13, proj_skeleton)
                self.connection(13,14, proj_skeleton)

            elif num_features >= 16 and self.hard_add_joint==True: 
                self.connection(0,1, proj_skeleton)
                self.connection(1,2, proj_skeleton)
                self.connection(3,4, proj_skeleton)
                self.connection(4,5, proj_skeleton)

                self.connection(0,6, proj_skeleton)
                self.connection(3,6, proj_skeleton)

                self.connection(6,7, proj_skeleton)
                self.connection(7,9, proj_skeleton)

                self.connection(7,10, proj_skeleton)
                self.connection(10,11, proj_skeleton)
                self.connection(11,12, proj_skeleton)
                self.connection(7,13, proj_skeleton)
                self.connection(13,14, proj_skeleton)
                self.connection(14,15, proj_skeleton)

            

            if frame_cnt<num_frames-1: frame_cnt += 1
            else: run = False

            pygame.display.update()

            ## Generating screenshots and saving them in respective folder. 
            # if shot_cnt % shot_freq == 0:
            #     filename = "Screenshots/"+ f'{shot_cnt}' +".png"
            #     pygame.image.save(self.screen, filename)
            #     filename = "Screenshots/"+ f'{shot_cnt}' +".pdf"
            #     pygame.image.save(self.screen, filename)
            
            shot_cnt += 1


        pygame.quit()


    


def main(): 
    skelrenderer = SKEL_RENDERER()

    # directory of experiment
    experiment_path = ...
    part = ...
    exp = ...

    dim = 'Pos/'
    # dim = 'PosVel/'
    # dim = 'PosDirMag/'

    sample = ...

    file = 'final_inputs.pt'
    # file = 'final_predictions.pt'


    data_path = part+exp+dim+sample+file
    data = torch.load(experiment_path+data_path)

    shape = data.size()

    num_frames = shape[0]
    num_features = 15
    num_dimensions = int(shape[1] / num_features)
        
    if num_dimensions == 3:
        gestalt = False
    else:
        gestalt = True

    if gestalt:
        data = data.view(num_frames, num_features, num_dimensions)
        pos = data[:,:,:3]
        dir = data[:,:,3:6]
        mag = data[:,:,-1]

        skelrenderer.render(pos, dir, mag, gestalt)
    else:
        data = data.view(num_frames, num_features, num_dimensions)
        skelrenderer.render(data, None, None, gestalt)


if __name__ == "__main__":
    main()
