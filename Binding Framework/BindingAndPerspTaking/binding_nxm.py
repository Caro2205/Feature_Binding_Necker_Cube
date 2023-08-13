"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""
import itertools
import random
from tempfile import TemporaryDirectory
import numpy as np
import torch
from torch import nn

from scipy.signal import butter, lfilter
from scipy.interpolate import UnivariateSpline

class BINDER_NxM():
    """
    Performs Binding task. 

    Initial parameters:
        - num_observations = M: How many features are observed?
        - num_featrues = N: On how many features should the observations be bound? 
                            (i.e. number of input features of the coreLSTM)
    
    Important functions: 
        - scale_binding_matrix: determine binding matrix from given activations by either 
            > using sigmoidal activation function
            > using row- and columnwise softmax with inverted temperature
        - bind: binding M observations to N input features using the given binding matrix
        - update_binding_matrix: update given binding activations with given gradients using
            > SGD with given learning rate and momentum
            > if sign_damp=True: sign damping with given alpha
            > if self.prescale=True: tanh-scaaling using self.sigma
    """

    def __init__(self, num_observations, num_features):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_features = num_features
        self.num_observations = num_observations
        self.nxm = (num_features != num_observations)

        self.bin_momentum = self.init_momentum()
        self.bin_sign_damp_binding = self.init_sign_damping()
        
        self.set_prescale('None')
        self.set_sigma(10)

        self.t = 0
        self.t_sign = 0
        self.ts = []

        #self.temps_col =[]
        #self.temps_row = []
        self.prev_avg_col = []
        self.prev_avg_row = []

        self.all_filtered_losses = []


    ######## Initializations ################################################################
    #########################################################################################


    def init_momentum(self):
        if self.nxm: 
            return torch.zeros(self.num_features+1, self.num_observations).to(self.device)
        else:
            return torch.zeros(self.num_features, self.num_features).to(self.device)


    def init_sign_damping(self):
        if self.nxm: 
            return torch.zeros(self.num_features+1, self.num_observations).to(self.device)
        else:
            return torch.zeros(self.num_features, self.num_features).to(self.device)


    def init_binding_matrix(self):
        init_val = 1.0/self.num_features
        binding_matrix = torch.Tensor(self.num_features, self.num_observations)
        binding_matrix.requires_grad = False
        binding_matrix = binding_matrix.fill_(init_val)
        return binding_matrix


    def set_binding_matrix(self, versionBM):
        #versionBM = 'wholeMatrixUniform' #"wholeMatrixRandom", "random"
        val = 5
        # val = self.sigma
        binding_matrix = torch.ones(self.num_features, self.num_observations) * -val
        binding_matrix.requires_grad = False

        if versionBM == 'tie':
            for i in [0,1,2,3,4,5,9,10,11,12,13,14]:
                binding_matrix[i,i] = 0.0
            for i in [6,7,8]:
                binding_matrix[i,i] = val
            for i,j in [(0,3), (1,4), (2,5), (9,12), (10,13), (11,14)]:
                binding_matrix[i,j] = 0.0
                binding_matrix[j,i] = 0.0

        elif versionBM == 'tie_80_20':            
            for i in [0,1,2,3,4,5,9,10,11,12,13,14]:
                binding_matrix[i,i] = 0.1 * val
            for i in [6,7,8]:
                binding_matrix[i,i] = val
            for i,j in [(0,3), (1,4), (2,5), (9,12), (10,13), (11,14)]:
                binding_matrix[i,j] = 0.9 * val
                binding_matrix[j,i] = 0.9 * val

        elif versionBM == 'contra':
            for i in [6,7,8]:
                binding_matrix[i,i] = val
            for i,j in [(0,3), (1,4), (2,5), (9,12), (10,13), (11,14)]:
                binding_matrix[i,j] = val
                binding_matrix[j,i] = val

        elif versionBM == 'original':
            for i in range(self.num_features):
                binding_matrix[i,i] = val

        elif versionBM == 'random':
            for i in range(self.num_features):
                binding_matrix[i, i] = random.randint(-5, 5)

        elif versionBM == 'wholeMatrixRandom':
            for i, j in itertools.product(range(self.num_features), range(self.num_features)):
                binding_matrix[i, j] = random.randint(-5, 5)

        elif versionBM == 'wholeMatrixUniform':
            for i, j in itertools.product(range(self.num_features), range(self.num_features)):
                binding_matrix[i, j] = 0.5

        return binding_matrix

        
    def set_prescale(self, prescale): 
        self.prescale = prescale


    def set_sigma(self, sigma):
        self.sigma = sigma


    ######## Binding Matrix Computations ####################################################
    #########################################################################################

    def ideal_nxm_binding(self, additional_features, ideal_matrix):
        zeros = np.zeros((self.num_features, 1))
        for i in additional_features: 
            ideal_1 = ideal_matrix[:, :i]
            ideal_2 = ideal_matrix[:, i+1:]
            ideal_matrix = np.hstack([ideal_1, zeros, ideal_2])
        
        dummy_line = np.zeros((1, self.num_observations))
        for i in additional_features: 
            dummy_line[0, i] = 1
        
        ideal_matrix = np.vstack([ideal_matrix, dummy_line])
        
        return torch.Tensor(ideal_matrix)
    

    def scale_binding_matrix(self, 
        bm_activations, 
        scale_mode):

        if scale_mode == 'sigmoid':
            # compute sigmoidal
            return torch.sigmoid(bm_activations)

        elif scale_mode == 'rcwSM': 
                
            # get row-wise softmax with inverted temperature
            rwB = self.temperatured_softmax(bm_activations, self.temp_val_row, 1)

            # get column-wise softmax with inverted temperature
            cwB = self.temperatured_softmax(bm_activations, self.temp_val_col, 0)

            if self.nxm:
                # replace last line by ones-row
                rwB = torch.cat([
                    rwB[:-1],
                    torch.ones(1, self.num_observations).to(self.device)
                ])
                
                # combine row-wise and column-wise softmax
                Q = rwB * cwB 

                # take square-root from the binding-matrix (without outcast-line)
                Q[:-1] = torch.sqrt(Q[:-1])

            else: 
                # combine row-wise and column-wise softmax
                Q = torch.sqrt(rwB * cwB)
            
            return Q

        else: 
            if scale_mode != 'unscaled': 
                print('ERROR: Unknown mode! Return unscaled binding matrix.')
            return bm_activations


    def bind(self, input, bind_matrix):
        return torch.matmul(bind_matrix.float(), input.float())     #input


    def update_binding_matrix(self, matrix, gradient, learning_rate, momentum, sign_damp=False, alpha=0.0):
        # calculate momentum term 
        mom = momentum * self.bin_momentum

        # calculate change in matrix 
        if sign_damp:
            # update sign damping variable 
            self.bin_sign_damp_binding = alpha * self.bin_sign_damp_binding + (1-alpha) * torch.sign(gradient)
            upd = -learning_rate * gradient * torch.square(self.bin_sign_damp_binding) + mom
        else: 
            upd = -learning_rate * gradient + mom

        # reset momentum
        self.bin_momentum = upd

        if self.prescale == 'tanh_fixed':
            # update binding matrix
            updMatrix = matrix + upd
            # scale matrix values 
            return self.sigma * torch.tanh(updMatrix / self.sigma)
        elif self.prescale == 'clamp':
            # update binding matrix
            updMatrix = matrix + upd
            # scale matrix values 
            return torch.clamp(updMatrix, -self.sigma, self.sigma)
        else:
            # update binding matrix
            return matrix + upd


    ######## Temperature Computations #######################################################
    #########################################################################################

    def temperatured_softmax(self, bm, t, dimint):
        return nn.functional.softmax(bm*t, dim=dimint)


    def set_temperature(self, temp_params):

        ## Temperature values are fixed 
        if len(temp_params)==2:
            print("Fixed temperature used.")

            self.temp_val_row = temp_params[0]
            self.temp_val_col = temp_params[1]

        elif len(temp_params)==1:
            print("Smooth turned-down temperature used.")
            [(temp_row, temp_col)] = temp_params
            self.temp_change = "smooth_down"

            self.max_temp_val_row = temp_row[0]
            self.max_temp_val_col = temp_col[0]

            self.temp_lambda_row = temp_row[1]
            self.temp_lambda_col = temp_col[1]

            self.temp_val_row = 1 / 1000
            self.temp_val_col = 1 / 1000

            # self.temp_val_row = 0
            # self.temp_val_col = 0

            self.temp_down_cnt_row = 0
            self.temp_down_cnt_col = 0


        ## Temperature values are turned up 
        elif len(temp_params)==5:
            print("Step turned-up temperature used.")
            self.temp_change = "step_up"

            [temperature, 
            temperature_step_col, 
            temperature_step_row, 
            temperature_fct, 
            temperature_fct_relative] = temp_params

            self.max_temp_val_row = temperature[0]
            self.max_temp_val_col = temperature[1]

            self.temp_val_row = 0
            self.temp_val_col = 0

            self.temp_step_col = temperature_step_col
            self.temp_step_row = temperature_step_row

            self.fct_temperature_turnup = temperature_fct
            self.fct_realtive_temperature_turnup = temperature_fct_relative

            if self.fct_temperature_turnup == 'sigmoid':
                self.sig_cnt_row = torch.Tensor([0]).to(self.device)
                self.sig_cnt_col = torch.Tensor([0]).to(self.device)
                self.sig_drift_row = 4
                self.sig_drift_col = 4
                self.sig_offset_row = 0
                self.sig_offset_col = 0
            elif self.fct_temperature_turnup == 'elu':
                self.elu_cnt_row = torch.Tensor([0]).to(self.device)
                self.elu_cnt_col = torch.Tensor([0]).to(self.device)
                self.elu_drift_row = 4
                self.elu_drift_col = 4
                self.elu_alpha_row = 0.1
                self.elu_alpha_col = 0.1

        ## No temperature specified
        else:
            print("No temperature used.")
            
            self.temp_val_row = 1
            self.temp_val_col = 1

    
    def reset_temp(self):
        self.temp_val_row = 1 / 1000
        self.temp_val_col = 1 / 1000



    def get_temp_row(self):
        return self.temp_val_row
    def get_temp_col(self):
        return self.temp_val_col

    def filtered_loss(self, current_loss, cycle):
        alpha = 0.95
        if cycle == 0:
            filtered_loss = current_loss
        else:
            last_filtered_loss = self.all_filtered_losses[-1]
            filtered_loss = alpha * last_filtered_loss + (1 - alpha) * current_loss

        self.all_filtered_losses.append(filtered_loss)

    def get_filtered_losses(self):
        return self.all_filtered_losses

    def filtered_mov_avg_temp_adaption(self, cycle):
        threshold = 0.001
        incr_factor = 0.005
        decr_factor = 0.04
        max_temp = 4

        losses = self.all_filtered_losses

        if cycle < 200 and self.temp_val_row < max_temp:
            self.temp_val_col += incr_factor
            self.temp_val_row += incr_factor
            print('Temperature has been increased to ' + str(self.temp_val_col))
        else:
            last_10 = losses[-10:]
            moving_average = np.mean(last_10)
            self.prev_avg_col.append(moving_average)
            if self.prev_avg_col.__len__() > 1 and abs(moving_average - self.prev_avg_col[-2]) < threshold and self.temp_val_col > 0.11 and losses[-1] <= 0.2: # moving_average
                self.temp_val_col -= decr_factor
                self.temp_val_row -= decr_factor
                print('Temperature has been decreased to ' + str(self.temp_val_col))
            elif self.temp_val_col < max_temp:
                self.temp_val_col += incr_factor
                self.temp_val_row += incr_factor
                print('Temperature has been increased to ' + str(self.temp_val_col))


    def mov_avg_temp_adaption_col(self, losses, cycle):
        threshold = 0.0001
        incr_factor = 0.02
        decr_factor = 0.06
        max_temp = 4

        if cycle < 200 and self.temp_val_row < max_temp:
            self.temp_val_col += incr_factor
            print('Temperature has been increased to ' + str(self.temp_val_col))
        else:
            last_10 = losses[-10:]
            moving_average = np.mean(last_10)
            self.prev_avg_col.append(moving_average)
            test = self.prev_avg_col
            if self.prev_avg_col.__len__() > 1 and abs(losses[-1] - self.prev_avg_col[-2]) < threshold and self.temp_val_col > 0.11 and losses[-1] <= 0.2: # moving_average
                self.temp_val_col -= decr_factor
                print('Temperature has been decreased to ' + str(self.temp_val_col))
            elif self.temp_val_col < max_temp:
                self.temp_val_col += incr_factor
                print('Temperature has been increased to ' + str(self.temp_val_col))

    def mov_avg_temp_adaption_row(self, losses, cycle):
        threshold = 0.0001
        incr_factor = 0.02
        decr_factor = 0.06
        max_temp = 4

        if cycle < 200 and self.temp_val_row < max_temp:
            self.temp_val_row += incr_factor
        else:
            last_10 = losses[-10:]
            moving_average = np.mean(last_10)
            self.prev_avg_row.append(moving_average)

            if self.prev_avg_row.__len__() > 1 and abs(losses[-1] - self.prev_avg_row[-2]) < threshold and self.temp_val_row > 0.11 and losses[-1] <= 0.2: # moving_average
                self.temp_val_row -= decr_factor
            elif self.temp_val_row < max_temp:
                self.temp_val_row += incr_factor



    def decr_temp_col_linear(self):
        if self.temp_val_col < 3.7:   #< 5:
            self.temp_val_col += 0.02 #0.0051 #0.02
            print("Temp_val_col has been linearly decreased to:")
            print(self.temp_val_col)

    def decr_temp_row_linear(self):
        if self.temp_val_row < 3.7:#< 5:
            self.temp_val_row += 0.02

            #0.0051 #0.02
            print("Temp_val_row has been linearly decreased to:")
            print(self.temp_val_row)

    def set_temp_constant(self):
        self.temp_val_col = 4.5
        self.temp_val_row = 4.5


    def get_max_temp_row(self):
        return self.max_temp_val_row


    def get_max_temp_col(self): 
        return self.max_temp_val_col


    def low_pass_filter(self, data):
        cutoff = 0.5
        order = 2
        data = data.cpu().numpy()
        data = np.transpose(data)

        b, a = butter(order, cutoff)
        return lfilter(b, a, data, axis=-1)


    def second_derivative(self, data):
        d_shape = data.shape
        d_spline_2d = []
        x = [*range(d_shape[1])]

        for i in range(d_shape[0]):
            d_spline = UnivariateSpline(x, data[i], s=0)
        
            d_spline_2d.append(d_spline.derivative(n=2)(x))

        return torch.Tensor(d_spline_2d)





