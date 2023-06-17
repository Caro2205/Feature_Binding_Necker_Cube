"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""

from numpy.core.numeric import Inf
from torch.functional import Tensor, norm
import torch 
from torch.utils.data import Dataset
import numpy as np

import sys
sys.path.append('C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/BindingAndPerspectiveTaking')
# Before run: replace ... with current directory path

from Data_Compiler.amc_parser import test_all

class skeleton_data_set(Dataset):
    def __init__(self, data):
        self.ins = data[:-1]
        self.tars = data[1:]
  
    def __len__(self):
        return len(self.ins)
  
    def __getitem__(self, index):

        return (self.ins[index], self.tars[index])


class Preprocessor():

    """
        Class to prepare data before or during training and testing of LSTM and BPAT inference.
    """
  
    def __init__(
        self, 
        num_observations=None, 
        num_features=15, 
        num_dimensions=3
    ):
        
        self.num_features = num_features
        if num_observations is None: 
            self.num_observations = num_features
        else:
            self.num_observations = num_observations
        
        self.mirror = False
        self.nxm = False
        self._num_dimensions = num_dimensions
        self.num_spatial_dimensions = 3

        # paths for distractors
        self.thaichi_asf_path = 'Data_Compiler/samples/S12T04_thaichi.asf'
        self.thaichi_amc_path = 'Data_Compiler/samples/S12T04_thaichi.amc'
        self.modern_asf_path = 'Data_Compiler/samples/S05T02_modern.asf'
        self.modern_amc_path = 'Data_Compiler/samples/S05T02_modern.amc'


    #############################################################################################
    #### Basic setters and data handlers.
    #############################################################################################

    def set_parameters(self, num_feat, num_obs, num_dim):
        self.num_features = num_feat
        self.num_observations = num_obs
        self.num_dimensions = num_dim
        
    
    def set_distractors(self, distractor):
        self.nxm = True
        self.distractor = distractor


    def set_num_observations(self, num_obs):
        self.num_observations = num_obs


    def set_mirror(self, mirror):
        self.mirror = mirror


    def reset_dimensions(self, dim):
        self._num_dimensions = dim
        print(f'Reset dimensions to {self._num_dimensions}')

        
    def compile_data(self, asf_path, amc_path, frame_samples):
        visual_input, selected_joint_names = test_all(asf_path, amc_path, frame_samples, 30, self.num_observations) 
        visual_input = torch.from_numpy(visual_input).type(torch.float)
        
        visual_input = visual_input.permute(1,0,2)
        
        return visual_input, selected_joint_names
    

    def std_scale_data(self, input_data, scale_factor):
        normed = torch.norm(input_data, dim=2)
        scale_factor = 1/(np.sqrt(scale_factor) * normed.std())
        scale_mat = torch.Tensor([[scale_factor, 0, 0], 
                                  [0, scale_factor, 0], 
                                  [0, 0, scale_factor]])
        scaled = torch.matmul(input_data, scale_mat)
        print(f'Scaled data by factor {scale_factor}')
        print(f'New minimum: {torch.min(scaled)}')
        print(f'New maximum: {torch.max(scaled)}')
        return scaled


    def add_noise(self, input_data, noise_factor):
        noise = noise_factor * (torch.rand(input_data.shape) - 0.5)
        return input_data + noise


    def create_inout_sequences(self, input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L-tw):
            train_seq = input_data[i:i+tw]
            train_label = input_data[i+tw:i+tw+1]
            inout_seq.append([train_seq ,train_label])
        return inout_seq


    #############################################################################################
    #### Functions to create different Gestalt variants for data.
    #############################################################################################

    def get_motion_data(self, abs_data, num_frames):
        motion_dt = torch.Tensor(abs_data.shape)[1:]

        for i in range(num_frames-1):
            # motion_dt[i] = abs_data[i+1] - abs_data[i]
            motion_dt[i] = abs_data[i] - abs_data[i+1]
            # NOTE: try with causal time (t-1) - (t)
        # print('Constructed motion data.')

        return motion_dt

    
    def get_direction_data(self, abs_data, num_frames): 
        velocity = self.get_motion_data(abs_data, num_frames)        
        direction = torch.nn.functional.normalize(velocity, dim=2)

        return direction


    def get_magnitude_data(self, abs_data, num_frames): 
        velocity = self.get_motion_data(abs_data, num_frames)
        magnitude = torch.norm(velocity, dim=2) 

        return magnitude

    
    def get_gestalt_dir_mag(self, input, frame_samples):
        direction = self.get_direction_data(input, frame_samples)
        magnitude = self.get_magnitude_data(input, frame_samples).unsqueeze(dim=2)

        return torch.cat([input[1:], direction, magnitude], dim=2)


    def get_gestalt_vel(self, input, frame_samples):
        velocity = self.get_motion_data(input, frame_samples)

        return torch.cat([input[1:], velocity], dim=2)

    
    #############################################################################################
    #### Functions to load and prepare data for LSTM training.
    #############################################################################################

    """
        Get LSTM data for walker. 
    """
    def get_LSTM_data_gestalten(self, 
        asf_path, 
        amc_path, 
        frame_samples, 
        num_test_data, 
        train_window, 
        noise
    ):

        visual_input, selected_joint_names = self.compile_data(
            asf_path=asf_path, 
            amc_path=amc_path, 
            frame_samples=frame_samples
        )
        visual_input = self.std_scale_data(visual_input, 15)    

        if noise is not None: 
            visual_input = self.add_noise(visual_input, noise) 

        if self._num_dimensions == 6:
            visual_input = self.get_gestalt_vel(visual_input, frame_samples)
            frame_samples -= 1
        elif self._num_dimensions == 7:
            visual_input = self.get_gestalt_dir_mag(visual_input, frame_samples)
            frame_samples -= 1
            
        visual_input = visual_input.reshape(
            1, 
            frame_samples, 
            (self._num_dimensions) *self.num_features
        )
        
        train_data = visual_input[:,:-num_test_data,:]
        test_data = visual_input[:,-num_test_data:,:]

        train_inout_seq = self.create_inout_sequences(train_data[0], train_window)

        return train_inout_seq, train_data, test_data


    """
        Get LSTM data for optical illusions. 
    """
    def get_LSTM_OI_data_gestalten(self, 
        path, 
        frame_samples, 
        num_test_data, 
        train_window, 
        noise
    ):

        visual_input = torch.load(path)  

        if noise is not None: 
            visual_input = self.add_noise(visual_input, noise) 

        if self._num_dimensions == 6:
            visual_input = self.get_gestalt_vel(visual_input, frame_samples)
            frame_samples -= 1
        elif self._num_dimensions == 7:
            visual_input = self.get_gestalt_dir_mag(visual_input, frame_samples)
            frame_samples -= 1
            
        visual_input = visual_input.reshape(
            1, 
            frame_samples, 
            (self._num_dimensions) *self.num_features
        )
        
        train_data = visual_input[:,:-num_test_data,:]
        test_data = visual_input[:,-num_test_data:,:]

        train_inout_seq = self.create_inout_sequences(train_data[0], train_window)

        return train_inout_seq, train_data, test_data

    
    #############################################################################################
    #### Functions to load and prepare data for Active Tuning inference.
    #############################################################################################


    """
        Get inference data for walker. 
    """
    def get_AT_data_gestalten(self, asf_path, amc_path, frame_samples):
        visual_input, selected_joint_names = self.compile_data(asf_path=asf_path, amc_path=amc_path, frame_samples=frame_samples)
        visual_input = self.std_scale_data(visual_input, 15)        

        if self.nxm:
            visual_input, selected_joint_names = self.append_distractor_motion(visual_input, selected_joint_names, frame_samples)
        
        if self._num_dimensions == 6:
            visual_input = self.get_gestalt_vel(visual_input, frame_samples)
        elif self._num_dimensions == 7:
            visual_input = self.get_gestalt_dir_mag(visual_input, frame_samples)

        return visual_input, selected_joint_names


    """
        Get inference data for optical illusions. 
    """
    def get_AT_data_oi_gestalten(self, path, frame_samples):
        visual_input = torch.load(path)[:frame_samples] 
        selected_joint_names = [*range(self.num_features)] 

        if self.nxm:
            visual_input, selected_joint_names = self.append_distractor_motion(visual_input, selected_joint_names, frame_samples)
        
        if self._num_dimensions == 6:
            visual_input = self.get_gestalt_vel(visual_input, frame_samples)
        elif self._num_dimensions == 7:
            visual_input = self.get_gestalt_dir_mag(visual_input, frame_samples)

        return visual_input, selected_joint_names
    
    #############################################################################################

    def convert_data_AT_to_VAE(self, data):
        return data.reshape(1, self._num_dimensions*self.num_features)


    def convert_data_VAE_to_AT(self, data):
        return data.reshape(self.num_features, self._num_dimensions)


    #############################################################################################
    #### Functions to include distractors into data.
    #############################################################################################

    def append_distractor_motion(self, visual_input, selected_joint_names, frame_samples):
        for (dis, feats) in self.distractor:

            if dis == 'thaichi': 
                dis_asf = self.thaichi_asf_path
                dis_amc = self.thaichi_amc_path
            elif dis == 'modern': 
                dis_asf = self.modern_asf_path
                dis_amc = self.modern_amc_path

            disfeat_position, disfeat_names = self.compile_data(asf_path=dis_asf, amc_path=dis_amc, frame_samples=frame_samples)
            disfeat_position = self.std_scale_data(disfeat_position, 15)

            for i in feats: 
                visual_input = torch.cat([visual_input, disfeat_position[:,i].unsqueeze(1)], dim=1)
                selected_joint_names.append(dis+'_'+disfeat_names[i])

        return visual_input, selected_joint_names


    def get_distractor_position(self, num_frames):
        x_turn = 1
        x_speed = 0.01
        x_radius = -0.3

        y_turn = 1
        y_speed = 0.01
        y_radius = -0.2

        z_turn = 1
        z_speed = 0.001
        z_radius = 0.2

        pos = torch.zeros(num_frames, 3)
        
        x_i = np.arange(-1*x_turn, 1*x_turn, x_speed)
        x = 0
        y_i = np.arange(-1*y_turn, 1*y_turn, y_speed)
        y = 0
        z_i = np.arange(-1*z_turn, 1*z_turn, z_speed)
        z = 0

        for frame in range(num_frames):
            
            pos[frame, 0] = x_i[x]
            if x == len(x_i)-1:
                x_turn *= -1
                x_i = np.arange(-1, 1, x_speed) *x_turn
                x = 0
            else: 
                x += 1
            
            pos[frame, 1] = y_i[y]
            if y == len(y_i)-1:
                y_turn *= -1
                y_i = np.arange(-1, 1, y_speed) *y_turn
                y = 0
            else: 
                y += 1

            pos[frame, 2] = z_i[z]
            if z == len(z_i)-1:
                z_turn *= -1
                z_i = np.arange(-1, 1, z_speed) * z_turn
                z = 0
            else: 
                z += 1

        pos = torch.mul(torch.acos(pos), torch.Tensor([x_radius, y_radius, z_radius]))
        pos = pos.reshape(num_frames, 1, 3)

        
        print('Created distractor.')

        return pos
            
                


        





