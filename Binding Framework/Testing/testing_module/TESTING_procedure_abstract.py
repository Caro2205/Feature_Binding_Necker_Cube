"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""

import pandas as pd   
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

import torch 
from datetime import datetime
import os

import sys
sys.path.append('/Users/MartinButz2/Documents/CODE/Python/BindingDancersAndCubes/TimGerneProject')
#os.chdir('C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/BindingAndPerspectiveTaking')

# Before run: replace ... with current directory path

from Data_Compiler.data_preparation import Preprocessor
from Data_Compiler.skeleton_renderer import SKEL_RENDERER
from BindingAndPerspTaking.perspective_taking import Perspective_Taker

"""

    This class mainly exists due to organizational reasons. 
    It holds all smaller functions used in the general tester. 
    Most importantly, the paths of the data used in the experiments is defined here. 
        -> select data by comment in or out the path. 

"""

class TEST_PROCEDURE(ABC): 

    def __init__(self, num_features, num_observations, num_dimensions): 
        self.num_features = num_features
        self.num_observations = num_observations
        self.num_dimensions = num_dimensions
        self.gestalten = False
        self.dir_mag_gest = False

        self.illusion = None

        self.mirror_data = False

        self.both_directions = True

        self.ill_data_cnt = 0

        self.preprocessor = Preprocessor(
            self.num_observations, 
            self.num_features, 
            self.num_dimensions
            )
            
        # fixed. could be changed to flexible. 
        if self.num_dimensions > 3:
            self.gestalten = True
            if self.num_dimensions > 6:
                self.dir_mag_gest = True


        self.PERSP_TAKER = Perspective_Taker(num_observations, num_dimensions)
        self.skelrenderer = SKEL_RENDERER()

        self.set_modification = False

        print('Initialized test procedure.')


    def set_illusion(self, illusion):
        self.illusion = illusion


    def set_edge_points(self, num_points):
        if self.illusion == 'necker_cube':
            self.edge_points = num_points 
            self.num_features = 8 + self.edge_points * 12
            self.num_observations = self.num_features
            self.PERSP_TAKER.set_parameters(self.num_observations, self.num_dimensions)
        else:
            self.edge_points = None


    def set_dimensions(self, new_dimensions):
        self.num_dimensions = new_dimensions
        self.preprocessor.reset_dimensions(new_dimensions)
        # fixed. could be changed to flexible. 
        if self.num_dimensions > 3:
            self.gestalten = True
            if self.num_dimensions > 6:
                self.dir_mag_gest = True
            else:
                self.dir_mag_gest = False
        
        self.PERSP_TAKER.set_parameters(self.num_observations, self.num_dimensions)


    def load_data(self):
        pass

        
    def create_trial_directory(self, path): 
        if not os.path.isdir(path):
            os.mkdir(path)
        
        dateTimeObj = datetime.now()
        timestamp = dateTimeObj.strftime("%Y_%b_%d-%H_%M_%S")
        self.result_path = path+timestamp+'/'
        os.mkdir(self.result_path)

    
    def get_data_paths(self):
        #paths = []
        #paths.append("Data_Compiler/optical_illusions/tensor_dataset_tim.pt")
        #return paths

        if self.illusion == 'dancer':
            return self.get_dancer_data_path()
        elif self.illusion == 'necker_cube':
            return self.get_necker_cube_data_path()
        elif self.illusion == 'necker_cube_static':
            return ["Data_Compiler/optical_illusions/tensor_dataset_tim.pt"]
        else:
            return self.get_walker_data_paths()


    def get_dancer_data_path(self):
        data_paths = []
        file = f"Data_Compiler/optical_illusions/{self.illusion}_data/"

        data_paths += [f"{file}frames10000_rot['0.9990', '0.0000', '0.0350', '0.0000']_2022Mar23104115_flatback_asarmr180.pt"]
        
        if self.both_directions:
            data_paths += [f"{file}frames10000_rot['0.9990', '0.0000', '-0.0350', '0.0000']_2022Mar23104121_flatback_asarmr180.pt"]
            
        if self.mirror_data:
            data_paths += [f"{file}frames10000_rot['0.9990', '0.0000', '0.0350', '0.0000']_2022Mar23104107_mirrored_flatback_asarmr180.pt"]

            if self.both_directions:
                data_paths += [f"{file}frames10000_rot['0.9990', '0.0000', '-0.0350', '0.0000']_2022Mar23104100_mirrored_flatback_asarmr180.pt"]
            
        return data_paths


    def get_necker_cube_data_path(self):
        data_paths = []
        file = f"Data_Compiler/optical_illusions/{self.illusion}_data/"
        if self.edge_points == 0: 
            data_paths += [f"{file}frames10000_edgeP0_rot['0.9999', '-0.0004', '-0.0006', '-0.0010']_2021Nov20173528" + ".pt"]
            if self.both_directions:
                data_paths += [f"{file}frames10000_edgeP0_rot['0.9999', '0.0004', '0.0006', '0.0010']_2021Nov20173304" + ".pt"]
                
        elif self.edge_points == 1: 
            data_paths += [f"{file}frames10000_edgeP1_rot['0.9999', '-0.0004', '-0.0006', '-0.0010']_2021Nov21164706" + ".pt"]
            if self.both_directions:
                data_paths += [f"{file}frames10000_edgeP1_rot['0.9999', '0.0004', '0.0006', '0.0010']_2021Nov21165259" + ".pt"]

        elif self.edge_points == 3: 
            data_paths += [f"{file}frames10000_edgeP3_rot['0.9999', '-0.0004', '-0.0006', '-0.0010']_2021Nov21165025" + ".pt"]
            if self.both_directions:
                data_paths += [f"{file}frames10000_edgeP3_rot['0.9999', '0.0004', '0.0006', '0.0010']_2021Nov21165235" + ".pt"]
        else: 
            print('ERROR: Invalid number of edge points.')
            exit()
        
        return data_paths


    def get_walker_data_paths(self):
        data_amc_path = []
        data_asf_path = []
        # data from LSTM training
        data_amc_path += ['Data_Compiler/samples/S35T07.amc']
        data_asf_path += ['Data_Compiler/samples/S35T07.asf']

        # inference test data
        data_amc_path += ['Data_Compiler/samples/S05T01.amc']
        data_asf_path += ['Data_Compiler/samples/S05T01.asf']

        data_amc_path += ['Data_Compiler/samples/S06T01.amc']
        data_asf_path += ['Data_Compiler/samples/S06T01.asf']

        data_amc_path += ['Data_Compiler/samples/S08T02.amc']
        data_asf_path += ['Data_Compiler/samples/S08T02.asf']

        data_amc_path += ['Data_Compiler/samples/S07T02.amc']
        data_asf_path += ['Data_Compiler/samples/S07T02.asf']

        return (data_amc_path, data_asf_path)

    
    def load_data_all(self, data_paths, sample_nums, modification):      

        if self.illusion == None: 
            (amc_paths, asf_paths) = data_paths
            max_frame_cnt = 1000
        else: 
            max_frame_cnt = 10000

        data = []
        
        maxLen = max(sample_nums)
        if maxLen > max_frame_cnt:
            sample_nums = sample_nums[1:]

        for i in range(1):   #range(len(sample_nums)):
            if not self.set_modification:
                if self.illusion == None:
                    optimal_data= self.load_data_original(
                        asf_paths[i], 
                        amc_paths[i], 
                        sample_nums[i])
                else:
                    optimal_data= self.load_data_oi_original(
                        data_paths[i], 
                        sample_nums[i])
                data += [optimal_data]
            
            if modification is not None:
                if self.illusion == None:
                    modified_data = self.load_data_modified(
                        asf_paths[i], 
                        amc_paths[i], 
                        sample_nums[i], 
                        modification)
                else:
                    modified_data = self.load_data_oi_modified(
                        data_paths[i], 
                        sample_nums[i], 
                        modification)
                data += [modified_data]

        # if len(set(sample_nums)) != 1:
        if maxLen > 1000 or len(sample_nums) != 0:
            for i in range(len(data)):
                (n, d, f) = data[i]
                if d.shape[0] < maxLen:
                    mult_factor = np.ceil(np.array([maxLen/d.shape[0]]))
                    mult_factor = mult_factor.astype(int)
                    for k in range(mult_factor[0]):
                        d = torch.cat([d,d])
                    d_new = d[:maxLen]
                    data[i] = (n, d_new, f)

        self.ill_data_cnt = 0

        self.BPAT.set_feature_names(data[0][2])
                    
        return data


    def load_data_original(self, asf_path, amc_path, num_samples): 
        
        observations, joint_names = self.preprocessor.get_AT_data_gestalten(asf_path, amc_path, num_samples)

        name, _ = asf_path.split('.')
        _ , _ , name = name.split('/')
        return (name, observations, joint_names)


    def load_data_oi_original(self, path, num_samples): 
        
        observations, joint_names = self.preprocessor.get_AT_data_oi_gestalten(path, num_samples)
        if self.illusion == 'dancer':
            # joint_names = [
            #     'Lhipjoint', 'Lfemur', 'Lfoot',
            #     'Rhipjoint', 'Rfemur', 'Rfoot',
            #     'Lowerback', 'Throax', 'Head',
            #     'Lclavicle', 'Lradius', 'Lhand',
            #     'Rclavicle', 'Rradius', 'Rhand']
            joint_names = ['LefDowFar', 'RigDowFar', 'RigUpFar', 'LefUpFar', 'LefUpClo', 'RigUpClo', 'RigDowClo', 'LefDowClo']
        elif self.illusion == 'necker_cube_static':
            joint_names = ['LefDowFar', 'RigDowFar', 'RigUpFar', 'LefUpFar', 'LefUpClo', 'RigUpClo', 'RigDowClo',
                             'LefDowClo']

        name = f'{self.illusion}_{self.ill_data_cnt}'
        self.ill_data_cnt += 1
        # TODO: get more adaptive 

        return (name, observations, joint_names)        


    def load_data_modified(self, asf_path, amc_path, num_samples, modification): 
        (name, data, joint_names) = self.load_data_original(asf_path, amc_path, num_samples)
        name, data = self.modify_data(name, data, num_samples, modification)

        return (name, data, joint_names)


    def load_data_oi_modified(self, path, num_samples, modification): 
        (name, data, joint_names) = self.load_data_oi_original(path, num_samples)
        name, data = self.modify_data(name, data, num_samples, modification)

        return (name, data, joint_names)


    def modify_data(self, name, data, num_samples, modification):
        original_shape = data.shape
        # TODO modify
        for mode, modify in modification: 
            if mode == 'qrotate':
                if self.gestalten:
                    if self.dir_mag_gest:
                        mag = data[:,:, -1].view(num_samples-1, self.num_observations, 1)
                        data = data[:,:, :-1]
                    data = torch.cat([
                        data[:,:, :self.num_dimensions], 
                        data[:,:, self.num_dimensions:]], dim=2)
                    data = data.view((num_samples-1)*self.num_observations*2, 3)
                    data = self.PERSP_TAKER.qrotate(data, modify)   
                    data = data.reshape(num_samples-1, self.num_observations, 6)
                    if self.dir_mag_gest:
                        data = torch.cat([data, mag], dim=2)
                else:
                    data = data.view(num_samples*self.num_observations, self.num_dimensions)
                    data = self.PERSP_TAKER.qrotate(data, modify)   
                    data = data.view(original_shape) 

                print("Q-Rotated", name)
                name += "_qrotated"

            elif mode == 'eulrotate':
                rotmat = self.PERSP_TAKER.compute_rotation_matrix_(modify[0], modify[1], modify[2])

                if self.gestalten:
                    if self.dir_mag_gest:
                        mag = data[:,:, -1].view(num_samples-1, self.num_observations, 1)
                        data = data[:,:, :-1]
                    data = torch.cat([
                        data[:,:, :self.num_dimensions], 
                        data[:,:, self.num_dimensions:]], dim=2)
                    data = data.view((num_samples-1)*self.num_observations*2, 3)
                    data = self.PERSP_TAKER.rotate(data, rotmat)   
                    data = data.reshape(num_samples-1, self.num_observations, 6)
                    if self.dir_mag_gest:
                        data = torch.cat([data, mag], dim=2)
                else:
                    data = data.view(num_samples*self.num_observations, self.num_dimensions)
                    data = self.PERSP_TAKER.rotate(data, rotmat)   
                    data = data.view(original_shape)

                
                print("Euler-Rotated", name)
                # name += "_eulrotated"
                name += "_qrotated" # wrong name needed for automatical evaluation

            elif mode == 'translate':
                if self.gestalten:
                    non_pos = data[:,:,3:]
                    data = data[:,:,:3]
                    original_shape = data.size()
                    data = data.view((num_samples-1)*self.num_observations, 3)
                else:
                    original_shape = data.size()
                    data = data.view(num_samples*self.num_observations, self.num_dimensions)
                   
                data = self.PERSP_TAKER.translate(data, modify)
                data = data.view(original_shape)

                if self.gestalten:
                    data = torch.cat([data, non_pos], dim=2)

                print("Translated", name)
                name += "_translated"
            else: 
                print('Unknown modification ', mode, ' for ', name, ' was skipped.')  


        return name, data


    def prepare_inference(self, 
        BPAT, 
        num_frames, 
        model_path, 
        layer_norm,
        tuning_length, 
        num_tuning_cycles, 
        at_loss_function, 
        at_learning_rate, 
        at_learning_rate_state, 
        at_momentum, 
        at_signdamps):

        BPAT.init_model_(model_path, layer_norm)
        BPAT.set_tuning_parameters_(
            tuning_length, 
            num_tuning_cycles, 
            at_loss_function, 
            at_learning_rate, 
            at_learning_rate_state, 
            at_momentum, 
            at_signdamps
        )
        BPAT.init_inference_tools()

    
    def init_modification_params(self):
        self.new_rotation = None
        self.rerotate = None
        self.new_translation = None
        self.retranslate = None


    def construct_info_string(self, info_string, loss_parameters):
        info_string += f' - structure of tuning:  \t{self.structure}\n'
        info_string += f' - number of observations: \t{self.BPAT.num_input_features}\n'
        info_string += f' - number of features: \t\t{self.BPAT.num_input_features}\n'
        info_string += f' - number of dimensions: \t{self.BPAT.num_input_dimensions}\n'
        info_string += f' - mirror data: \t\t{self.mirror_data}\n\n'

        info_string += f' - model: \t\t\t{self.BPAT.core_model}\n\n'
        info_string += f' - model path: \t\t{self.BPAT.core_model_path}\n\n'
        info_string += f' - number of tuning cycles: \t{self.BPAT.tuning_cycles}\n'
        info_string += f' - size of tuning horizon: \t{self.BPAT.tuning_length}\n'
        info_string += f' - loss function: \t\t{self.BPAT.at_loss}\n'
        for name, value in loss_parameters:
            info_string += f'\t> {name}: \t{value}\n'

        return info_string



    def evaluate(self, BPAT, optimal_inputs, final_predictions):
        self.save_results_to_pt([final_predictions], ['final_predictions'])
        results = BPAT.get_result_history(optimal_inputs, final_predictions)

        figures = []
        figures += [BPAT.evaluator.plot_prediction_errors(results[0])]
        figures += [BPAT.evaluator.plot_at_losses(results[1], 'History of overall losses during active tuning')]

        fig_names = ['prediction_errors', 'at_loss_history']

        if len(results) == 8:
            pred_name = ['predictions_errors_dim']

            z_pred = results[2]
            self.save_results_to_pt([z_pred], pred_name)

            if z_pred.shape[1] == 30:
                pred_name = ['predictions_errors_dim_pos']
                pred_fig = [BPAT.evaluator.plot_zpred_loss(z_pred[:, :15].cpu(), 'Dim prediction errors (Pos) after active tuning')]
                self.save_figures(pred_fig, pred_name) 

                pred_name = ['predictions_errors_dim_vel']
                pred_fig = [BPAT.evaluator.plot_zpred_loss(z_pred[:, 15:].cpu(), 'Dim prediction errors (Vel) after active tuning')]
                self.save_figures(pred_fig, pred_name) 
            else: 
                pred_fig = [BPAT.evaluator.plot_zpred_loss(z_pred.cpu(), 'Dim prediction errors after active tuning')]
                self.save_figures(pred_fig, pred_name)    
            # NOTE: Make nicer. combine with result history. Check dependencies. 

            results = results[:2] + results[3:]
        
        return results, figures, fig_names


    def save_tesor_to_csv(self, tensor, path): 
        torch.save(tensor, path)


    def save_figures(self, figures, names):
        for i in range(len(figures)): 
            fig = figures[i]
            fig.savefig(self.result_path + names[i] + '.png')
            # fig.savefig(self.result_path + names[i] + '.pdf')

    
    def write_to_file(self, string, path):
        out = open(path, "wt")
        out.write(string)
        out.close()


    def save_results_to_csv(self, results, names): 
        for i in range(len(results)):
            df = pd.DataFrame(results[i])
            if not df.empty:
                df.to_csv(self.result_path + names[i] + '.csv')


    def save_results_to_pt(self, results, names): 
        for i in range(len(results)):
            torch.save(results[i], self.result_path + names[i] + '.pt')

    
    def render(self, data):
        self.skelrenderer.render(data, None, None, False)
    
    
    def render_gestalt(self, data):
        pos = data[:,:,:3]
        dir = data[:,:,3:6]
        if self.dir_mag_gest:
            mag = data[:,:,-1]
        else:
            mag = torch.ones(data[:,:,1].shape)

        self.skelrenderer.render(pos, dir, mag, True)
  
    
    def set_num_observations(self, num_obs):
        self.num_observations = num_obs


    def set_mirroring(self, mirror):
        self.mirror_data = mirror
        self.preprocessor.set_mirror(mirror)


    def set_both_directions(self, both_dir):
        self.both_directions = both_dir


    def run(self):
        pass


    def terminate(self):
        plt.close('all')
