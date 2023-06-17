"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""
import os

import torch
from torch import nn

# selber hinzugefuegt
import sys
# sys.path.append('C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/BindingAndPerspectiveTaking')

from Testing.testing_module.general_tester import TESTER
from Testing.testing_module.TESTING_statistical_evaluation_abstract import TEST_STATISTICS

# from general_tester import TESTER
# from TESTING_statistical_evaluation_abstract import TEST_STATISTICS

"""
    Class for defining parameters of experiment for optical illusion data.
    Variable self.illusion defines which illusion is used (dancer or necker cube), and thus 
    differentiates between different parameter values, models, etc.  

    In here, all basic parameters are set to their default values. 
    If a parameter is not defined as 'changed_parameter', this default value is used in all 
    experiment trials. 
    If a parameter is defined as 'changed_parameter', its value is changed for all 
    experiment trials, i.e. in every trial its value is set to the next value in tested_values. 
    Parameters for which a testing is implemented can be set as 'changed_parameter').  
    

    Method 'perform_experiment' is to call from separate experiment classes. 
        -> defines default parameters
        -> performes all trials for different values for 'changed_parameter'
        -> evaluates experiment by creating comparing plots over all trials 

"""


class EXPERIMENT_INTERFACE_OPT_ILLUSIONS(TESTER):

    def __init__(self, num_features, num_observations, num_dimensions, experiment_name, illusion, edge_points):
        super().__init__(num_features, num_observations, num_dimensions, experiment_name)
        super().set_illusion(illusion)
        super().set_edge_points(edge_points)

        self.attr_path = None

        self.stats = TEST_STATISTICS(self.num_features, self.num_observations, self.num_dimensions)
        print('Initialized experiment.')


    def set_attractor_path(self, attr_path):
        self.attr_path = attr_path

    def load_attractor(self, sample_nums):
        sample_num = [sample_nums[0]]
        with torch.no_grad():
            self.attr_dt = super().load_data_all(self.attr_path, sample_num, None)   # TODO: perhaps add modification if more than binding is inferred!!!


    def perform_experiment(self,
            sample_nums, 
            modification,
            structure,
            changed_parameter, 
            tested_values, 
            rotation_type,
            distractors, 
            temperature): 


        #########################################################################################
        #### Experiment parameters ####
        #########################################################################################

        # NOTE: set manually, for all parameters that are not tested

        ### Tuning parameters       #############################################################
        # length of tuning horizon:

        tuning_length = 10  # 10
        if self.illusion == "necker_cube_static":
            tuning_length = 0

        # number of tuning cycles performed for every frame
        num_tuning_cycles = 3000#2000#1000

        ### Tuning structure        #############################################################
        structure = structure       # either: 'parallel', 'sequential', 
                                    #  'parallel_temp_turnup', or 
                                    #  'parallel_temp_turnup_inferdim'


        ### Initial Binding Matrix  #############################################################
        if structure == 'parallel_temp_turnup_BMforce' or structure == "necker_cube_static_bind":
            set_BM = True
        else: 
            set_BM = False
        init_BM_version = 'original'    # either: 'tie, 'contra', or 'original'


        ### Tuning loss             #############################################################
        # loss function for active tuning:
        at_loss_function = nn.MSELoss(reduction="sum")  # nn.SmoothL1Loss(reduction='sum', beta=0.00001)
        loss_parameters = [('reduction', 'sum')]        #[('beta', 0.00001), ('reduction', 'sum')]


        ### Binding parameters      #############################################################
        # scaler: 
        scaler = 'rcwSM'        # either: 'unscaled', 'sigmoid' or 'rcwSM'

        # prescaling of Softmax:
        pres = 'clamp'     # either: None, 'clamp', or 'tanh_fixed'
        # sigma = 10

        ## dim3
        sigma = 2

        ## dim6
        #sigma = 5
        

        ### Temperature parameters  #############################################################
        temperature_params = []
        ## If temperature is NOT turned up:
        if temperature == 'fixed':
            # temperature_params = (0.05,0.05)             # format: (temp_row, temp_col)
            # temperature_params = (0.1,0.1)             # format: (temp_row, temp_col)
            # temperature_params = (1,1)             # format: (temp_row, temp_col)
            temperature_params = (3,5)             # format: (temp_row, temp_col)

        elif temperature == "smooth_turn_down":
            temp_max = ((80,3.0),(80, 3.0))  #((130,1.0),(130, 1.0))
            # ((80,3.0),(80, 3.0)) gibt gute Binding Matrix

            temperature_params = [
                temp_max
                ]

            # number of frames after which the temperature is turned up:
            temp_range_col = 1      # columns
            temp_range_row = 1      # rows

            # number of frames after which the temperature is set back to 0: 
            temp_reset_frame = 200 + tuning_length

        ## If temperature is turned up: 
        elif temperature == 'turn_up': 

            ## Different values for different illusions
            if self.illusion == 'dancer':
                # maximum temperature:
                # temp_max = (0.6,0.6)
                # temp_max = (0.5,0.5)
                # temp_max = (0.3,0.3)
                # temp_max = (1,1)
                # temp_max = (2,2)

                temp_max = (3,5)
                # temp_max = (3,3)
                # temp_max = (5,5)
                # temp_max = (10,10)

                # value by which the temperature is turned up in every step:
                temp_step_col = 0.05    # column
                temp_step_row = 0.05    # rows

                # temp_step_col = 0.1    # column
                # temp_step_row = 0.1    # rows

                # temp_step_col = 0.2    # column
                # temp_step_row = 0.2    # rows

                # temp_step_col = 0.3    # column
                # temp_step_row = 0.3    # rows

                # temp_step_col = 0.5    # column
                # temp_step_row = 0.5    # rows

            elif self.illusion == 'necker_cube':
                # maximum temperature:
                temp_max = (5,5) 
            
                # value by which the temperature is turned up in every step:
                temp_step_col = 0.1    # column
                temp_step_row = 0.1    # rows

            elif self.illusion == 'necker_cube_static':
                temp_max = (3, 5)
                temp_step_col = 0.05  # column
                temp_step_row = 0.05  # rows
            else:
                print('ERROR: Invalid illusion! ')
                exit()

            # number of frames of which the gradients are consideres for temperature turn-up:
            temp_grad_range_col = 100
            temp_grad_range_row = 100

            # number of frames after which the temperature is turned up:
            # temp_range_col = 1      # columns
            # temp_range_row = 1      # rows

            temp_range_col = 5      # columns
            temp_range_row = 5      # rows

            # temp_range_col = 10      # columns
            # temp_range_row = 10      # rows
            
            # function to determine values of temperature:
            temp_fct = 'linear'                     # either: 'sigmoid' or 'linear'
            temp_fct_relative = 'lpfilter_deriv'    # either: 'lpfilter_deriv', 'mean' or 'sum_of_abs_diff'

            # number of frames after which the temperature is set back to 0: 
            temp_reset_frame = 210         
            # temp_reset_frame = 250         
            # temp_reset_frame = 1000000         

            # collect important parameters for setup
            temperature_params = [
                temp_max, 
                temp_step_col, 
                temp_step_row, 
                temp_fct, 
                temp_fct_relative
                ]

        else: 
            print("ERROR: Invalid temperature variant.")
            exit()

        ### NxM Binding parameters  #############################################################
        # more observations than the LSTM has input:
        nxm_bool = False

        # index of additional observation features:
        index_additional_features = []

        # how to initialize outcast line:
        initial_value_outcast_line = 0.01

        # pre-binding enhancement for outcast line:
        nxm_enhance = 'square'  # either: 'square', 'squareroot', 'log10'

        # pre-binding scaling for outcast line:
        nxm_outcast_line_scaler = 0.1

        # check for distractors and in case of existence increase number of observations
        if distractors is not None: 
            nxm_bool = True
            idx = self.num_features-1
            for (_, num_add_feat) in distractors: 
                i_num_add_feat = len(num_add_feat)
                self.num_observations += i_num_add_feat
                for i in range(i_num_add_feat):
                    idx += 1
                    index_additional_features.append(idx)


        ### Rotation parameters     #############################################################
        rot_type = rotation_type    # either: 'eulrotate' or 'qrotate'

        both_directions = True


        ### Mirroring               #############################################################
        mirror = False #True


        ### Learning rates          #############################################################
        #at_learning_rate_binding = 0.1
        at_learning_rate_binding = 1
        # at_learning_rate_binding = 0.05
        at_learning_rate_rotation =  0.005
        at_learning_rate_translation = 0.05
        at_learning_rate_state = 0.0


        ### Momenta #############################################################
        if self.illusion == 'dancer':
            at_momentum_binding = 0.95
            # at_momentum_binding = 0.9
            # at_momentum_binding = 0.999
            # at_momentum_binding = 0.99
            at_momentum_rotation = 0.9
            at_momentum_translation = 0.0

            # at_momentum_binding = 0.99
            # # at_momentum_binding = 0.99
            # at_momentum_rotation = 0.8
            # at_momentum_translation = 0.8
        elif self.illusion == 'necker_cube':
            # at_momentum_binding = 0.95
            at_momentum_binding = 0.99
            at_momentum_rotation = 0.95
            at_momentum_translation = 0.99

        elif self.illusion == 'necker_cube_static':
            at_momentum_binding = 0.95
            at_momentum_rotation = 0.9
            at_momentum_translation = 0.0

        else:
            print('ERROR: Invalid illusion! ')
            exit()
        


        ### Signdamping values      #############################################################
        # at_signdamp_binding = 0.5
        # # at_signdamp_binding = 0.8
        # # at_signdamp_binding = 0.2
        # at_signdamp_rotation = 0.2
        # # at_signdamp_rotation = 0.6
        # # at_signdamp_translation = 0.7
        # at_signdamp_translation = 0.4

        at_signdamp_binding = 0.0
        at_signdamp_rotation = 0.0
        at_signdamp_translation = 0.0

        # at_signdamp_binding = 0.8
        # at_signdamp_binding = 0.1
        # at_signdamp_binding = 0.01
        # at_signdamp_binding = 0.9
        # at_signdamp_rotation = 0.6
        # at_signdamp_translation = 0.7

        if self.illusion == 'dancer':
            at_signdamp_binding = 0.5
            # at_signdamp_binding = 0.99
            at_signdamp_rotation = 0.0
            at_signdamp_translation = 0.4

            # # at_signdamp_binding = 0.0
            # at_signdamp_binding = 0.0
            # at_signdamp_rotation = 0.5
            # at_signdamp_translation = 0.9

        elif self.illusion == 'necker_cube':
            # at_signdamp_binding = 0.0
            at_signdamp_binding = 0.99
            at_signdamp_rotation = 0.0
            at_signdamp_translation = 0.0

        elif self.illusion == 'necker_cube_static':
            at_signdamp_binding = 0.5
            at_signdamp_rotation = 0.0
            at_signdamp_translation = 0.4

        else:
            print('ERROR: Invalid illusion! ')
            exit()
            


        ### Gradient calculation    #############################################################
        # how to calculate the gradient over the tuning horizon
        grad_calc_binding = 'meanOfTunHor' #'weightedInTunHor'  # 'lastOfTunHor'
        grad_calc_rotation = 'meanOfTunHor'
        grad_calc_translation = 'meanOfTunHor'
        grad_calculations = [grad_calc_binding, grad_calc_rotation, grad_calc_translation]


        ### Gradient weighting      #############################################################
        # bias for gradient weighting (not important for 'meanOfTunHor')
        grad_bias_binding = 1.4
        grad_bias_rotation = 1.1
        grad_bias_translation = 1.1 
        grad_biases = [grad_bias_binding, grad_bias_rotation, grad_bias_translation]


    
        experiment_results = []

        ## experiment performed for all values of the tested parameter
        for val in tested_values: 

            #####################################################################################
            #### Set-up experiment ####
            #####################################################################################

            ### Tested parameter    #############################################################

            if changed_parameter == 'dimensions':
                self.set_dimensions(val)
                print(f'CHANGED gestalt!\nNew value for dimension: {val}')

            elif changed_parameter == 'number_of_edges':
                self.set_edge_points(val)
                print(f'CHANGED number of edgepoints!\nNew number of edges: {val}')

            elif changed_parameter == 'dimensions_and_edge_numbers':
                (dim, edg) = val
                self.set_dimensions(dim)
                print(f'CHANGED gestalt!\nNew value for dimension: {dim}')

                self.set_edge_points(edg)
                print(f'CHANGED number of edgepoints!\nNew number of edges: {edg}')

            elif changed_parameter == 'prescale':
                pres = val
                print(f'CHANGED prescaling!\nNew prescale: {val}')

            elif changed_parameter == 'sigma':
                sigma = val
                print(f'CHANGED prescaling!\nNew sigma: {val}')

            elif changed_parameter == 'structure':
                structure = val
                print(f'CHANGED structure type!\nNew structure: {val}')

            elif changed_parameter == 'temperature_fixed':
                temperature_params = val
                print(f'CHANGED temperature!\nNew value for temperature: {val}')

            elif changed_parameter == 'temperature_turn_up':
                (temp_max, 
                    temp_step_col, 
                    temp_step_row, 
                    temp_range_col, 
                    temp_range_row, 
                    temp_fct, 
                    temp_reset_frame)  = val

                temperature_params = [
                    temp_max, 
                    temp_step_col, 
                    temp_step_row, 
                    temp_fct, 
                    temp_fct_relative
                ]

                print(f'CHANGED temperature!\nNew value for temperature: {val}')

            else: 
                print(f'ERROR: Invalid value for changed_parameter: {changed_parameter}')
                exit()


            self.set_BPAT_structure(structure)
            self.preprocessor.set_parameters(
                self.num_features, 
                self.num_observations, 
                self.num_dimensions)


            ### Core model processing #############################################################
            layerNorm = False
            model_path = f'CoreLSTM/models/optical_illusions/{self.illusion}/'
            if self.illusion == 'dancer':
                # self.BPAT.set_hidden_num_oi(200)
                self.BPAT.set_hidden_num_oi(300)
                # self.BPAT.set_hidden_num_oi(120)
                # self.BPAT.set_hidden_num_oi(100)

                if self.num_dimensions == 3:
                    # model_path += 'mod_3_15_100_MSELoss()_0.01_0.9_0.0_10_100_nse0.0005'
                    # model_path += 'mod_3_15_100_MSELoss()_0.001_0.99_0.0_15_400_nseNone_mirrored'
                    # model_path += 'mod_3_15_110_MSELoss()_0.001_0.99_0.0_15_500_nse0.0001_mirrored'

                    # model_path += 'mod_3_15_120_MSELoss()_0.001_(0.3, 0.4)_0.18_10_100_lNORM_mirrturn20'
                    # model_path += 'mod_3_15_120_MSELoss()_0.001_(0.3, 0.4)_0.18_5_100_lNORM_mirrturn35'

                    # model_path += 'mod_testing3_15_200_MSELoss()_0.01_(0.9, 0.999)_0.0_10_300_lNORM_mirrturn35'
                    # model_path += 'mod_testing3_15_300_MSELoss()_0.01_(0.9, 0.999)_0.0_20_400_mirrturn35'
                    # model_path += 'mod_testing_flatback3_15_300_MSELoss()_0.01_(0.9, 0.999)_0.0_20_400_mirrturn35'
                    # model_path += 'mod_testing3_15_300_MSELoss()_0.01_(0.9, 0.999)_0.0_20_500_mirrturn35'

                    model_path += 'mod_3_15_300_MSELoss()_0.01_(0.9, 0.999)_0.0_20_500_mirrturn35asarattr_sep'


                elif self.num_dimensions == 6:
                    # model_path += 'mod_6_15_300_MSELoss()_0.01_(0.9, 0.999)_0.0_20_300_mirrturn35asar180testing'
                    # model_path += 'mod_6_15_300_MSELoss()_0.01_(0.9, 0.999)_0.0_20_500_mirrturn35asar180attr_septesting'
                    # model_path += 'mod_6_15_300_MSELoss()_0.01_(0.9, 0.999)_0.0_20_500_nse0.0001_mirrturn35asarattr_septesting'
                    # model_path += 'mod_6_15_300_MSELoss()_0.01_(0.9, 0.999)_0.0_20_500_nse0.0001_mirrturn35_reat2asarattr_sep'

                    # model_path += 'mod_6_15_300_MSELoss()_0.01_(0.9, 0.999)_0.0_20_400_nse0.0001_mirrturn35asar'
                    # model_path += 'mod_6_15_300_MSELoss()_0.01_(0.9, 0.999)_0.0_20_500_nse0.0001_mirrturn35asar_2'
                    # model_path += 'mod_6_15_200_MSELoss()_0.01_(0.9, 0.999)_0.0_20_400_nse0.0001_mirrturn35asar'
                    # model_path += 'mod_6_15_200_MSELoss()_0.01_(0.9, 0.999)_0.0_20_500_nse0.0001_mirrturn35asar_2'
                    # model_path += 'mod_6_15_200_MSELoss()_0.01_(0.9, 0.999)_0.0_20_500_nse0.0001_mirrturn35asar'

                    ## velocity: t - t-1
                    model_path += 'mod_6_15_300_MSELoss()_0.01_(0.9, 0.999)_0.0_20_400_nse0.0001_mirrturn35asardifvel'

                elif self.num_dimensions == 7:
                    model_path += 'mod_7_15_100_MSELoss()_0.01_0.9_0.0_10_100_nse0.0005'
                else:
                    print('ERROR: Unvalid number of dimensions!\nPlease use 3, 6, or 7.')
                    exit()

            elif self.illusion == 'necker_cube':
                self.BPAT.set_hidden_num_oi(150)
                if self.num_dimensions == 7:
                    if self.edge_points == 0:
                        model_path += ...
                    elif self.edge_points == 1:
                        model_path += 'mod_7_20_100_MSELoss()_0.01_0.9_0.0_10_100_nse0.0005_ep1'
                    elif self.edge_points == 3:
                        model_path += ...
                    else:
                        print('ERROR: No model for given number of edge points.')
                elif self.num_dimensions == 6:
                    if self.edge_points == 0:
                        model_path += 'mod_6_8_150_MSELoss()_0.001_0.95_0.0_10_150_nse0.0005_ep0_bothdir'
                        # model_path += 'mod_6_8_200_MSELoss()_0.01_0.9_0.0_10_100_nse0.0005_ep0_bothdir'
                    elif self.edge_points == 1:
                        # model_path += 'mod_6_20_100_MSELoss()_0.01_0.9_0.0_10_100_nse0.0005_ep1'
                        model_path += 'mod_6_20_150_MSELoss()_0.001_0.95_0.0_10_150_nse0.0005_ep1_bothdir'
                        # model_path += 'mod_6_20_200_MSELoss()_0.01_0.9_0.0_10_100_nse0.0005_ep1_bothdir'
                    elif self.edge_points == 3  :
                        model_path += 'mod_6_44_150_MSELoss()_0.001_0.95_0.0_10_150_nse0.0005_ep3_bothdir'
                        # model_path += 'mod_6_44_200_MSELoss()_0.01_0.9_0.0_10_100_nse0.0005_ep3_bothdir'
                    else:
                        print('ERROR: No model for given number of edge points.')
                elif self.num_dimensions == 3:
                    if self.edge_points == 0:
                        model_path += 'mod_3_8_150_MSELoss()_0.001_0.95_0.0_10_150_nse0.0005_ep0_bothdir'
                        # model_path += 'mod_3_8_200_MSELoss()_0.01_0.9_0.0_10_100_nse0.0005_ep0_bothdir'
                    elif self.edge_points == 1:
                        # model_path += 'mod_3_20_100_MSELoss()_0.01_0.9_0.0_10_100_nse0.0005_ep1'
                        model_path += 'mod_3_20_150_MSELoss()_0.001_0.95_0.0_10_150_nse0.0005_ep1_bothdir'
                        # model_path += 'mod_3_20_200_MSELoss()_0.01_0.9_0.0_10_100_nse0.0005_ep1_bothdir'
                    elif self.edge_points == 3:
                        model_path += 'mod_3_44_150_MSELoss()_0.001_0.95_0.0_10_150_nse0.0005_ep3_bothdir'
                        # model_path += 'mod_3_44_200_MSELoss()_0.01_0.9_0.0_10_100_nse0.0005_ep3_bothdir'
                    else:
                        print('ERROR: No model for given number of edge points.')
                else:
                    print('ERROR: Unvalid number of dimensions!\nPlease use 3, 6, or 7.')
                    exit()

            elif self.illusion == "necker_cube_static":
                #model_path = 'C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/TimGerneProjectCodeOnly/VAE_models/saved_model_parameters'

                #model_path = 'C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/TimGerneProjectCodeOnly/VAE_models/saved_model_parameters_VAE_2_dataset_1'
                #model_path = 'C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/TimGerneProjectCodeOnly/VAE_models/saved_model_parameters_VAE_2_dataset_1_0.2'
                model_path = 'C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/TimGerneProjectCodeOnly/VAE_models/saved_model_parameters_VAE_2_dataset_1_0.4'
                #model_path = 'C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/TimGerneProjectCodeOnly/VAE_models/saved_model_parameters_VAE_2_dataset_1_0.8'
                #model_path = 'C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/TimGerneProjectCodeOnly/VAE_models/saved_model_parameters_VAE_2_dataset_0'


               # model_path = 'C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/TimGerneProjectCodeOnly/VAE_models/saved_model_parameters_VAE_3_dataset_1'
                #model_path = 'C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/TimGerneProjectCodeOnly/VAE_models/saved_model_parameters_VAE_3_dataset_3'




            else:
                print('ERROR: Invalid illusion! ')
                exit()

            model_path += '.pt'
         

            ### BPAT parameters   #############################################################

            self.BPAT.set_optical_illusion(True)
            self.BPAT.set_binding_prescale(pres)
            self.BPAT.set_binding_sigma(sigma)
            self.BPAT.set_gradient_calculation(grad_calculations)
            self.BPAT.set_weighted_gradient_biases(grad_biases)  
            self.BPAT.set_rotation_type(rot_type)    
            self.BPAT.set_scale_mode(scaler)
            self.BPAT.set_init_axis_angle(0)
            self.BPAT.set_binding_prescale(pres)
            self.set_mirroring(mirror)
            self.set_both_directions(both_directions)

            ## temperature ##
            self.BPAT.set_temperature_parameters(temperature_params)
            if structure == 'parallel_temp_turnup_relative':
                self.BPAT.set_relative_temperature_grad_range(temp_grad_range_col, temp_grad_range_row)
            if temperature == 'turn_up' or temperature == 'smooth_turn_down':
                self.BPAT.set_range_temperature_turnup(temp_range_col, temp_range_row)
                self.BPAT.set_temp_reset(temp_reset_frame)
            
            if nxm_bool:
                self.BPAT.set_distractors(distractors)
                self.preprocessor.set_distractors(distractors)
                self.BPAT.set_additional_features(index_additional_features)
                self.BPAT.set_outcast_init_value(initial_value_outcast_line)
                self.BPAT.set_nxm_enhancement(nxm_enhance)
                self.BPAT.set_nxm_last_line_scale(nxm_outcast_line_scaler) 


            ## attractors ##
            if self.attr_path is not None: 
                self.load_attractor(sample_nums)
                self.BPAT.set_other_attractor(self.attr_dt)

            ## Set initial binding matrix
            self.BPAT.set_binding_matrix_init(set_BM, init_BM_version)
            
            
            ###################################################################################
            #### Run experiment ####
            ###################################################################################
            sample_names, result_names, results = super().run(
                changed_parameter+"_"+str(val)+"/",
                modification,
                sample_nums, 
                model_path, 
                layerNorm,
                tuning_length, 
                num_tuning_cycles, 
                at_loss_function,
                loss_parameters,
                [at_learning_rate_binding, at_learning_rate_rotation, at_learning_rate_translation], 
                at_learning_rate_state, 
                [at_momentum_binding, at_momentum_rotation, at_momentum_translation],
                [at_signdamp_binding, at_signdamp_rotation, at_signdamp_translation]
            )

            experiment_results += [results]

        #########################################################################################
        #### Experiment evaluation ####
        #########################################################################################

        self.stats.set_parameters(self.num_features, self.num_observations, self.num_dimensions)

        dfs = self.stats.load_csvresults_to_dataframe(
            self.prefix_res_path, 
            changed_parameter, 
            tested_values, 
            sample_names, 
            result_names[1:]
            )

        self.stats.plot_histories(
            dfs, 
            self.prefix_res_path, 
            changed_parameter, 
            result_names[1:],
            result_names[1:]
        )

        # self.stats.plot_value_comparisons(
        #     dfs,
        #     self.prefix_res_path,
        #     changed_parameter,
        #     result_names,
        #     result_names
        # )


        #########################################################################################
        print("Terminated experiment.")

