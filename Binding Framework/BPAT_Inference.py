"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""

# packet imports 
from builtins import super
import numpy as np
import torch
import copy
from torch import nn, autograd
import matplotlib.pyplot as plt

import sys
import os

#sys.path.append(
#    'C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/BindingAndPerspectiveTaking')
# os.chdir('C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/BindingAndPerspectiveTaking')
# Before run: replace ... with current directory path

# class imports 
from BindingAndPerspTaking.binding_nxm import BINDER_NxM
from BindingAndPerspTaking.perspective_taking import Perspective_Taker
# from CoreLSTM.core_lstm import CORE_NET
from Data_Compiler.data_preparation import Preprocessor
from BPAT_evaluation import BPAT_evaluator

from VAE_models import VAE_model
from VAE_models import VAE_model_large


class BPAT_Inference():
    """
    	Class of BPAT inference. 

        Contains methods to initialize and modify BPAT parameters. 
        Parameters are all initialized by default values set in experiment_interface(_oi).


        run_inference() performs BPAT inference in respective subclasses. 

    """

    def __init__(self):
        ## General parameters 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        autograd.set_detect_anomaly(True)

        torch.set_printoptions(precision=8)

        ## Set default parameters
        ##   -> Can be changed during experiments 

        self.scale_mode = 'rcwSM'
        self.rotation_type = 'qrotate'
        self.grad_bias_binding = 1.5
        self.grad_bias_rotation = 1.5
        self.grad_bias_translation = 1.5
        self.nxm = False
        self.distractors = None
        self.additional_features = []  # None
        self.nxm_enhance_ocvals = 'square',
        self.nxm_enhance_ocgrads = True
        self.nxm_last_line_scale = 0.1
        self.outcast_init_value = 0.1
        self.gestalten = False
        self.initial_angle = 0

        self.num_spatial_dimensions = 3

        self.oi = False
        self.lstm_num_hidden = 200
        self.lstm_num_hidden_oi = 120
        # self.lstm_num_hidden_oi = 100

        self.temp_turnup = False
        self.relative_temp_turnup = False
        self.range_temp_turnup_col = None
        self.range_temp_turnup_row = None
        self.rel_temp_grad_range_col = None
        self.rel_temp_grad_range_row = None
        self.temp_reset_frame = None

    ############################################################################
    ##########  PARAMETERS  ####################################################

    def set_scale_mode(self, mode):
        self.scale_mode = mode
        print('Set scale mode: ' + self.scale_mode)

    def set_distractors(self, distractors):
        self.distractors = distractors
        print(f'Distractor motions with corresponding features: {self.distractors}')

    def set_temperature_parameters(self, temp_params):
        if len(temp_params) != 2:
            self.temp_turnup = True
        self.temp_params = temp_params
        print(f'Set temperature for binding: {self.temp_params}')

    def set_range_temperature_turnup(self, range_col, range_row):
        self.range_temp_turnup_col = range_col
        self.range_temp_turnup_row = range_row

    def set_relative_temperature_grad_range(self, grad_range_col, grad_range_row):
        self.rel_temp_grad_range_col = grad_range_col
        self.rel_temp_grad_range_row = grad_range_row

    def set_temp_reset(self, reset_frame):
        self.temp_reset_frame = reset_frame

    def set_binding_prescale(self, prescale):
        self.binding_prescale = prescale
        print(f'Set scale combination: {self.binding_prescale}')

    def set_binding_sigma(self, sigma):
        self.binding_sigma = sigma
        print(f'Set scale combination: {self.binding_sigma}')

    def set_additional_features(self, index_addition):
        self.additional_features = index_addition
        print(f'Additional features to the LSTM-input at indices {self.additional_features}')

    def set_nxm_enhancement(self, enhancement):
        self.nxm_enhance_ocvals = enhancement
        print(f'Enhancement for outcast line: {self.nxm_enhance_ocvals}')

    def set_nxm_last_line_scale(self, scale_factor):
        self.nxm_last_line_scale = scale_factor
        print(f'Scaler for outcast line: {self.nxm_last_line_scale}')

    def set_outcast_init_value(self, init_value):
        self.outcast_init_value = init_value
        print(f'Initial value for outcast line: {self.outcast_init_value}')

    def set_rotation_type(self, rotation):
        self.rotation_type = rotation
        print('Set type of rotation: ' + self.rotation_type)

    def set_init_axis_angle(self, angle):
        self.initial_angle = angle
        print(f'Set axis angle of initial rotation: {self.initial_angle}')

    def set_weighted_gradient_biases(self, biases):
        # bias > 1 => favor recent
        # bias < 1 => favor earlier
        print('Set biases for gradient weighting:')
        self.grad_bias_binding = biases[0]
        print(f'\t> binding: {self.grad_bias_binding}')
        self.grad_bias_rotation = biases[1]
        print(f'\t> rotation: {self.grad_bias_rotation}')
        self.grad_bias_translation = biases[2]
        print(f'\t> translation: {self.grad_bias_translation}')

    def set_gradient_calculation(self, calculation_types):
        print('Set types of gradient calculation:')
        self.grad_calc_binding = calculation_types[0]
        print(f'\t> binding: {self.grad_calc_binding}')
        self.grad_calc_rotation = calculation_types[1]
        print(f'\t> rotation: {self.grad_calc_rotation}')
        self.grad_calc_translation = calculation_types[2]
        print(f'\t> translation: {self.grad_calc_translation}')

    def get_distractors(self):
        return self.distractors

    def get_additional_features(self):
        return self.additional_features

    def get_oc_grads(self):
        return self.oc_grads

    def set_optical_illusion(self, oi_bool):
        self.oi = oi_bool
        print(f'Set bool for optical illusion: {self.oi}')

    def set_hidden_num_oi(self, num_h):
        self.lstm_num_hidden_oi = num_h
        print(f'Set number of hidden neurons of LSTM: {num_h}')

    def set_hidden_num(self, num_h):
        self.lstm_num_hidden = num_h
        print(f'Set number of hidden neurons of LSTM: {num_h}')

    def set_binding_matrix_init(self, set_BM, version_BM):
        self.set_binding_matrix = set_BM
        self.bm_version = version_BM

    def set_feature_names(self, feature_names):
        self.feature_names = feature_names

    ############################################################################
    ##########  HYPER-PARAMETERS FOR INFERENCE  ################################

    def set_dimensions(self, num_dimensions):
        self.num_input_dimensions = num_dimensions
        self.gestalten = False
        self.dir_mag_gest = False

        if self.num_input_dimensions > 3:
            self.gestalten = True

            if self.num_input_dimensions > 6:
                self.dir_mag_gest = True
                self.num_mag = 1
            else:
                self.dir_mag_gest = False

            self.num_pos = self.num_spatial_dimensions
            self.num_dir = self.num_spatial_dimensions

    def set_data_parameters_(self,
                             num_frames,
                             num_observations,
                             num_input_features,
                             num_input_dimensions):

        ## Define data parameters
        self.num_frames = num_frames
        self.num_observations = num_observations
        self.num_input_features = num_input_features
        self.nxm = (self.num_observations != self.num_input_features)
        self.set_dimensions(num_input_dimensions)
        self.input_per_frame = self.num_input_features * self.num_input_dimensions

        self.binder = BINDER_NxM(
            num_observations=self.num_observations,
            num_features=self.num_input_features)

        self.binder.set_temperature(self.temp_params)
        self.binder.set_prescale(self.binding_prescale)
        self.binder.set_sigma(self.binding_sigma)

        # finally - store the universal diagonal binding matrix.
        self.bm_diag = self.binder.set_binding_matrix('original').requires_grad_()

        self.perspective_taker = Perspective_Taker(
            self.num_input_features,
            self.num_spatial_dimensions)

        self.preprocessor = Preprocessor(
            self.num_observations,
            self.num_input_features,
            self.num_input_dimensions)

        self.evaluator = BPAT_evaluator(
            self.num_frames,
            self.num_observations,
            self.num_input_features,
            self.preprocessor)

    def set_tuning_parameters_(self,
                               tuning_length,
                               num_tuning_cycles,
                               loss_function,
                               at_learning_rates_BPAT,
                               at_learning_rate_state,
                               at_momenta_BPAT,
                               at_signdamps_BPAT):

        ## Define tuning parameters 
        self.tuning_length = tuning_length  # length of tuning horizon
        self.tuning_cycles = num_tuning_cycles  # number of tuning cycles in each iteration

        # possible loss functions
        self.at_loss = loss_function
        self.mse = nn.MSELoss()
        self.l1Loss = nn.L1Loss()
        self.smL1Loss = nn.SmoothL1Loss(reduction='sum')
        self.l2Loss = lambda x, y: self.mse(x, y) * (self.num_input_dimensions * self.num_input_features)

        # define learning parameters 
        self.at_learning_rate_binding = at_learning_rates_BPAT[0]
        self.at_learning_rate_rotation = at_learning_rates_BPAT[1]
        self.at_learning_rate_translation = at_learning_rates_BPAT[2]
        self.at_learning_rate_state = at_learning_rate_state
        self.bm_momentum = at_momenta_BPAT[0]
        self.r_momentum = at_momenta_BPAT[1]
        self.c_momentum = at_momenta_BPAT[2]
        self.b_signdamp = at_signdamps_BPAT[0]
        self.r_signdamp = at_signdamps_BPAT[1]
        self.c_signdamp = at_signdamps_BPAT[2]
        self.at_loss_function = self.mse

        print('Parameters set.')

    def init_model_(self, model_path, layer_norm):
        # # Load LSTM model
        # self.core_model_path = model_path
        # print(f'Load model from {model_path}')
        # if self.oi:
        #     self.core_model = CORE_NET(
        #         input_size=self.num_input_dimensions*self.num_input_features,
        #         hidden_layer_size=self.lstm_num_hidden_oi,
        #         layer_norm=layer_norm)
        # else:
        #     self.core_model = CORE_NET(
        #         input_size=self.num_input_dimensions*self.num_input_features,
        #         hidden_layer_size=self.lstm_num_hidden)
        #
        # self.core_model.load_state_dict(torch.load(model_path, map_location=self.device))
        # self.core_model.eval()
        # self.core_model.to(self.device)

        # eigene Modelle
        # erstes modell auf allen 4 varianten trainiert:
        self.core_model_path = model_path

        self.core_model = VAE_model.VariationalAutoencoder()
        #self.core_model = VAE_model_large.VariationalAutoencoder()

        self.core_model.load_state_dict(torch.load(model_path))
        self.core_model.eval()
        self.core_model.to(self.device)

        print('Model loaded.')

    def init_inference_tools(self):
        ## Define tuning variables
        # general
        self.obs_count = 0
        self.at_observations = torch.tensor([]).to(self.device)
        self.at_model_inputs = torch.tensor([]).to(self.device)
        self.at_predictions = torch.tensor([]).to(self.device)
        self.at_final_predictions = torch.tensor([]).to(self.device)
        self.at_losses = []

        # state 
        self.at_states = []

        # binding
        self.Bs = []
        self.B_grads = [None] * (self.tuning_length + 1)
        self.B_upd = [None] * (self.tuning_length + 1)
        self.bm_losses = []
        self.bm_dets = []
        self.oc_grads = []

        # rotation
        self.Rs = []
        self.R_grads = [None] * (self.tuning_length + 1)
        self.R_upd = [None] * (self.tuning_length + 1)
        self.rm_losses = []
        self.rv_losses = []

        # translation
        self.Cs = []
        self.C_grads = [None] * (self.tuning_length + 1)
        self.C_upd = [None] * (self.tuning_length + 1)
        self.c_losses = []

    def set_comparison_values(self, ideal_binding, ideal_rotation, ideal_translation):
        # binding
        if ideal_binding is not None:
            self.ideal_binding = ideal_binding.to(self.device)

        # rotation
        if ideal_rotation is not None:
            self.identity_matrix = torch.Tensor(np.identity(self.num_spatial_dimensions))
            (ideal_rotation_values, self.ideal_rotation) = ideal_rotation
            self.ideal_rotation = self.ideal_rotation.to(self.device)
            if self.rotation_type == 'qrotate':
                self.ideal_quat = ideal_rotation_values.to(self.device)
                self.ideal_angle = self.perspective_taker.qeuler(self.ideal_quat, 'xyz').to(self.device)
            elif self.rotation_type == 'eulrotate':
                self.ideal_angle = ideal_rotation_values.to(self.device)
            else:
                print(f'ERROR: Received unknown rotation type!\n\trotation type: {self.rotation_type}')
                exit()

        # translation
        if ideal_translation is not None:
            self.ideal_translation = ideal_translation.to(self.device)

    ############################################################################
    ##########  INFERENCE - HELPERFUNCTIONS  ###################################

    def init_binding(self):
        ## Binding matrices 
        # Init binding entries 
        if self.set_binding_matrix == True:
            bm = self.binder.set_binding_matrix(self.bm_version)
            if self.bm_version == "contra":
                mask = bm > 0
                self.ideal_binding = torch.zeros(bm.shape).to(self.device)
                self.ideal_binding[mask] = 1
        else:
            bm = self.binder.init_binding_matrix()
        outcast_line = torch.ones(1, self.num_observations).to(self.device) * self.outcast_init_value

        # Store binding entries
        for i in range(self.tuning_length + 1):
            matrix = bm.clone().to(self.device)
            if self.nxm:
                matrix = torch.cat([matrix, outcast_line])
            matrix.requires_grad_()
            self.Bs.append(matrix)


    def init_rotation(self):
        if self.rotation_type == 'qrotate':
            ## Rotation quaternion 
            rq = self.perspective_taker.init_quaternion(self.initial_angle)

            for i in range(self.tuning_length + 1):
                quat = rq.clone().to(self.device)
                quat.requires_grad_()
                self.Rs.append(quat)

        elif self.rotation_type == 'eulrotate':
            ## Rotation euler angles 
            ra = self.perspective_taker.init_angles_(self.initial_angle)

            for i in range(self.tuning_length + 1):
                angles = []
                for j in range(self.num_spatial_dimensions):
                    angle = ra[j].clone().to(self.device)
                    angle.requires_grad_()
                    angles.append(angle)
                self.Rs.append(angles)

        else:
            print('ERROR: Received unknown rotation type!')
            exit()

    def init_translation(self):
        tb = self.perspective_taker.init_translation_bias_()

        for i in range(self.tuning_length + 1):
            transba = tb.clone().to(self.device)
            transba.requires_grad = True
            self.Cs.append(transba)

    def init_dimension_inference(self):
        self.Ds = []
        self.Ds_opt = [None] * (self.tuning_length + 1)
        self.D_grads = [None] * (self.tuning_length + 1)
        self.D_upd = [None] * (self.tuning_length + 1)
        self.d_losses = []

        if self.num_input_dimensions == 3:
            mom_dim = torch.zeros(self.num_observations).to(self.device)
            d_i = torch.zeros(self.num_observations)
        else:
            mom_dim = torch.zeros(2, self.num_observations).to(self.device)
            d_i = torch.zeros(2, self.num_observations)

        self.bin_momentum_dimension = [mom_dim] * (self.tuning_length + 1)

        for i in range(self.tuning_length + 1):
            d_init = d_i.clone().to(self.device)
            d_init.requires_grad = True
            self.Ds.append(d_init)

    def perform_bpt_binding_only(self, do_bind, idx, obs, bm=None):
        if do_bind:
            if bm is None:
                bm = self.binder.scale_binding_matrix(
                    self.Bs[idx],
                    self.scale_mode)
                if self.nxm:
                    bm = bm[:-1]

            x_B = self.binder.bind(obs, bm)  # -> Matrix multiplication (torch.matmul)
        else:
            x_B = obs

        x_BRC = self.preprocessor.convert_data_AT_to_VAE(x_B)  # reshape data

        return x_BRC, bm

    def perform_bpt(self, do_bind, do_rot, do_trans, idx, obs, bm=None, rotmat=None):

        S = self.num_spatial_dimensions
        O = self.num_observations
        ###########################  TRANSLATION  #############################
        if self.gestalten:
            if self.dir_mag_gest:
                mag = obs[:, -1].view(O, 1)
                dir = obs[:, S:2 * S].view(O, S)
                pos = obs[:, :S].view(O, S)
            else:
                dir = obs[:, S:].view(O, S)
                pos = obs[:, :S].view(O, S)

        else:
            pos = obs

        if do_trans:
            x_C = self.perspective_taker.translate(pos, self.Cs[idx])
        else:
            x_C = pos
        ###########################  ROTATION  ################################
        if self.gestalten:
            x_C = torch.cat([x_C, dir])

        if do_rot:
            if self.rotation_type == 'qrotate':
                x_R = self.perspective_taker.qrotate(x_C, self.Rs[idx])
            else:
                rotmat = self.perspective_taker.compute_rotation_matrix_(
                    self.Rs[idx][0], self.Rs[idx][1], self.Rs[idx][2])
                x_R = self.perspective_taker.rotate(x_C, rotmat).view(-1, S)
        else:
            x_R = x_C

        if self.gestalten:
            pos = x_R[:-O, :]
            dir = x_R[-O:, :]
            x_R = torch.cat([pos, dir], dim=1)

        ###########################  BINDING  #################################
        if self.dir_mag_gest:
            x_R = torch.cat([x_R, mag], dim=1)

        if do_bind:
            if bm is None:
                bm = self.binder.scale_binding_matrix(
                    self.Bs[idx],  # letzter Listeneintrag
                    self.scale_mode)
                if self.nxm:
                    bm = bm[:-1]
            x_B = self.binder.bind(x_R, bm)  # hier ist x_R der Wüfel und BM die gleichverteilte BM
        else:
            x_B = x_R

        #######################################################################

        x_BRC = self.preprocessor.convert_data_AT_to_VAE(x_B)

        return x_BRC, bm, rotmat

    def get_grad_binding(self, grad_calc_bind):
        # Calculate gradients with respect to the entires 
        for i in range(self.tuning_length + 1):
            self.B_grads[i] = self.Bs[i].grad   # warum 2 gradienten

            # Calculate overall gradients
        if grad_calc_bind == 'lastOfTunHor':
            ### version 1
            grad_B = self.B_grads[0]
        elif grad_calc_bind == 'meanOfTunHor':
            ### version 2 / 3
            grad_B = torch.mean(torch.stack(self.B_grads), dim=0)
            

        elif grad_calc_bind == 'weightedInTunHor':
            ### version 4
            weighted_grads_B = [None] * (self.tuning_length + 1)
            for i in range(self.tuning_length + 1):
                weighted_grads_B[i] = np.power(self.grad_bias_binding, i) * self.B_grads[i]
            grad_B = torch.mean(torch.stack(weighted_grads_B), dim=0)

        if self.nxm:
            factor = 14
            if self.nxm_enhance_ocgrads:
                dummy_line_grad = 2 * (torch.nn.functional.sigmoid(factor * grad_B[-1]) - 0.5)
            else:
                dummy_line_grad = grad_B[-1]
            grad_B = torch.cat([grad_B[:-1], dummy_line_grad.view(1, self.num_observations)])

        # print(f'grad_B: {grad_B}')

        return grad_B

    def get_grad_rotation(self, grad_calc_rot):
        ## get gradients
        if self.rotation_type == 'qrotate':
            for i in range(self.tuning_length + 1):
                # save grads for all parameters in all time steps of tuning horizon
                self.R_grads[i] = self.Rs[i].grad / self.num_input_features
        else:
            for i in range(self.tuning_length + 1):
                # save grads for all parameters in all time steps of tuning horizon
                grad = []
                for j in range(self.num_spatial_dimensions):
                    grad.append(self.Rs[i][j].grad)
                self.R_grads[i] = torch.stack(grad) / self.num_input_features

                # Calculate overall gradients
        if grad_calc_rot == 'lastOfTunHor':
            ### version 1
            grad_R = self.R_grads[0]
        elif grad_calc_rot == 'meanOfTunHor':
            ### version 2 / 3
            grad_R = torch.mean(torch.stack(self.R_grads), dim=0)
        elif grad_calc_rot == 'weightedInTunHor':
            ### version 4
            weighted_grads_R = [None] * (self.tuning_length + 1)
            for i in range(self.tuning_length + 1):
                weighted_grads_R[i] = np.power(self.grad_bias_rotation, i) * self.R_grads[i]
            grad_R = torch.mean(torch.stack(weighted_grads_R), dim=0)

        return grad_R

    def get_grad_translation(self, grad_calc_trans):
        ## Get gradients 
        for i in range(self.tuning_length + 1):
            # save grads for all parameters in all time steps of tuning horizon 
            self.C_grads[i] = self.Cs[i].grad / self.num_input_features

        # Calculate overall gradients 
        if grad_calc_trans == 'lastOfTunHor':
            ### version 1
            grad_C = self.C_grads[0]
        elif grad_calc_trans == 'meanOfTunHor':
            ### version 2 / 3
            grad_C = torch.mean(torch.stack(self.C_grads), dim=0)
        elif grad_calc_trans == 'weightedInTunHor':
            ### version 4
            weighted_grads_C = [None] * (self.tuning_length + 1)
            for i in range(self.tuning_length + 1):
                weighted_grads_C[i] = np.power(self.grad_bias_translation, i) * self.C_grads[i]
            grad_C = torch.mean(torch.stack(weighted_grads_C), dim=0)

        return grad_C

    def get_grad_dimension_inference(self):
        ## Get gradients 
        for i in range(self.tuning_length + 1):
            # save grads for all parameters in all time steps of tuning horizon 
            self.D_grads[i] = self.Ds[i].grad

    def loss_binding(self, updated_B, fbe):
        c_bm = self.binder.scale_binding_matrix(updated_B, self.scale_mode)

        mat_loss = self.evaluator.FBE(c_bm, self.ideal_binding)

        if self.nxm:
            fbe_af = self.evaluator.FBE_nxm_additional_features(
                c_bm, self.ideal_binding, self.additional_features)
            c_bm = self.evaluator.clear_nxm_binding_matrix(c_bm, self.additional_features)
            i_bm = self.evaluator.clear_nxm_binding_matrix(self.ideal_binding, self.additional_features)

            clear_loss = self.evaluator.FBE(c_bm, i_bm)

            mat_loss = torch.stack([clear_loss, fbe_af, mat_loss])

        self.bm_losses.append(mat_loss.cpu())
        print(f'loss of binding matrix (FBE): {mat_loss}')
        fbe.append(mat_loss.item())

        # Compute determinante of binding matrix
        det = torch.det(c_bm)
        self.bm_dets.append(det.cpu())
        print(f'determinante of binding matrix: {det}')

    def loss_rotation(self, updated_R):
        if self.rotation_type == 'qrotate':
            # Compare quaternion values
            # quat_loss = torch.sum(self.perspective_taker.qmul(self.ideal_quat, upd_R))
            quat_loss = 2 * torch.arccos(torch.abs(torch.sum(torch.mul(self.ideal_quat, updated_R))))
            quat_loss = torch.rad2deg(quat_loss)
            print(f'loss of quaternion: {quat_loss}')
            self.rv_losses.append(quat_loss.cpu())
            # Compute rotation matrix
            rotmat = self.perspective_taker.quaternion2rotmat(updated_R)

        else:
            # Save rotation angles
            # rotang = torch.stack(updated_R)
            # angles:
            ang_diff = updated_R - self.ideal_angle
            ang_loss = 2 - (torch.cos(torch.deg2rad(ang_diff)) + 1)
            print(f'loss of rotation angles: \n  {ang_loss}, \n  with norm {torch.norm(ang_loss)}')
            self.rv_losses.append(torch.norm(ang_loss))
            # Compute rotation matrix
            rotmat = self.perspective_taker.compute_rotation_matrix_(
                updated_R[0], updated_R[1], updated_R[2])[0]

        # Calculate and save rotation losses
        # matrix: 
        dif_R = torch.mm(self.ideal_rotation, torch.transpose(rotmat, 0, 1))
        mat_loss = torch.arccos(0.5 * (torch.trace(dif_R) - 1))
        mat_loss = torch.rad2deg(mat_loss)
        # if self.rotation_type == 'qrotate' and quat_loss == 0.0:
        #     mat_loss = torch.Tensor([0.0])
        print(f'loss of rotation matrix: {mat_loss}')
        self.rm_losses.append(mat_loss.cpu())

    def loss_translation(self, updated_C):
        trans_loss = torch.norm(updated_C - self.ideal_translation)
        self.c_losses.append(trans_loss.cpu())
        print(f'loss of translation bias: {trans_loss}')

    def loss_dimension(self):
        dim_loss = []
        for i in range(self.tuning_length + 1):
            dim_loss.append(torch.sum(self.Ds[i] - self.Ds_opt[i]).cpu())
        dim_loss = torch.sum(torch.Tensor(dim_loss))
        self.d_losses.append(dim_loss)
        print(f'loss of dimension values: {dim_loss}')

    ############################################################################
    ##########  EVALUATION #####################################################

    def intersave_matrices(self, fbm, fba, frame):

        frame = '%04d' % (int('00000') + frame)

        if self.num_observations == 15:
            fbm_plt = self.evaluator.plot_binding_matrix(
                fbm,
                self.feature_names,
                'binding matrix $B$'
            )
            fba_plt = self.evaluator.plot_binding_matrix(
                fba,
                self.feature_names,
                'binding activations $A$'
            )
        else:
            fbm_plt = self.evaluator.plot_binding_matrix_nxm(
                fbm,
                self.feature_names,
                self.num_observations,
                self.get_additional_features(),
                'binding matrix $B$'
            )
            fba_plt = self.evaluator.plot_binding_matrix_nxm(
                fba,
                self.feature_names,
                self.num_observations,
                self.get_additional_features(),
                'binding activations $A$'
            )

        posture_type = 'dancer'

        path = "Testing/Grafics/intersave_BMs/FBM/"
        fbm_plt.savefig(path + f'{posture_type}_B_{frame}.png')
        fbm_plt.savefig(path + f'{posture_type}_B_{frame}.pdf')
        torch.save(fbm, path + f'{posture_type}_B_{frame}.pt')

        path = "Testing/Grafics/intersave_BMs/FBA/"
        fba_plt.savefig(path + f'{posture_type}_A_{frame}.png')
        fba_plt.savefig(path + f'{posture_type}_A_{frame}.pdf')
        torch.save(fba, path + f'{posture_type}_A_{frame}.pt')

        plt.close('all')

    def get_result_history(
            self,
            observations,
            at_final_predictions):

        if self.nxm:
            pred_errors = self.evaluator.prediction_errors_nxm(
                observations,
                self.additional_features,
                self.num_observations,
                at_final_predictions,
                self.mse
            )
            self.bm_losses = torch.stack(self.bm_losses)

        else:
            pred_errors = self.evaluator.prediction_errors(
                observations,
                at_final_predictions,
                self.mse)

        return [pred_errors,
                self.at_losses,
                self.bm_dets,
                self.bm_losses,
                self.rm_losses,
                self.rv_losses,
                self.c_losses]
