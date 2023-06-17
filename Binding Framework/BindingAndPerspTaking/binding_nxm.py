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
        x = bm*t
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

        if self.temp_change=="step_up":
            self.temp_val_row = 0
            self.temp_val_col = 0

            if self.fct_temperature_turnup == 'sigmoid':
                self.sig_cnt_row = torch.Tensor([0]).to(self.device)
                self.sig_cnt_col = torch.Tensor([0]).to(self.device)

        elif self.temp_change=="smooth_down":
            self.temp_val_row = 1 / 1000
            self.temp_val_col = 1 / 1000

            # self.temp_val_row = 0
            # self.temp_val_col = 0

            self.temp_down_cnt_row = 0
            self.temp_down_cnt_col = 0


    def decr_temp_row(self, i):
        if self.temp_val_row < 3:
            self.temp_down_cnt_row += i
            self.temp_val_row = (self.temp_lambda_row * self.temp_down_cnt_row) / self.max_temp_val_row
            # self.temp_val_row = (1 + self.temp_lambda_row * self.temp_down_cnt_row) / self.max_temp_val_row
            # self.temp_val_row = self.max_temp_val_row / (1 + self.temp_lambda_row * self.temp_down_cnt_row)
            print("temp_val_row has been decreased to:")
            print(self.temp_val_row)
        # else:
        #     print(self.temp_val_row)
        #     exit()


    def decr_temp_col(self, i):
        if self.temp_val_col < 4:
            self.temp_down_cnt_col += i
            self.temp_val_col = (self.temp_lambda_col * self.temp_down_cnt_col) / self.max_temp_val_col
            # self.temp_val_col = (1 + self.temp_lambda_col * self.temp_down_cnt_col) / self.max_temp_val_col
            # self.temp_val_col = self.max_temp_val_col / (1 + self.temp_lambda_col * self.temp_down_cnt_col)
            print("Temp_val_col has been decreased to:")
            print(self.temp_val_col)

    def decr_temp_col_linear(self):
        if self.temp_val_col < 5:   #< 5:
            self.temp_val_col += 0.02 #0.0051 #0.02
            print("Temp_val_col has been linearly decreased to:")
            print(self.temp_val_col)

    def decr_temp_row_linear(self):
        if self.temp_val_row < 5:#< 5:
            self.temp_val_row += 0.02

            #0.0051 #0.02
            print("Temp_val_row has been linearly decreased to:")
            print(self.temp_val_row)


    def incr_temp_row(self):
        if self.fct_temperature_turnup == 'linear' and self.temp_val_row <= self.max_temp_val_row:
            self.temp_val_row += self.temp_step_row
        elif self.fct_temperature_turnup == 'sigmoid':
            self.temp_val_row = self.max_temp_val_row * nn.functional.sigmoid((self.sig_cnt_row - self.sig_drift_row)/self.max_temp_val_row) + self.sig_offset_row
            self.sig_cnt_row += self.temp_step_row
            print(self.sig_cnt_row)
        elif self.fct_temperature_turnup == 'elu':
            elu = lambda x : self.elu_alpha_row * (torch.exp(x)-1)
            # elu = nn.ELU(alpha=self.elu_alpha_row)
            if self.temp_val_row < self.max_temp_val_row:
                self.temp_val_row = elu(self.elu_cnt_row - self.elu_drift_row) + self.elu_alpha_row
                # self.temp_val_row = elu(self.elu_cnt_row/self.max_temp_val_row - self.elu_drift_row) + self.elu_alpha_row
            else:
                self.temp_val_row = self.max_temp_val_row
            self.elu_cnt_row += self.temp_step_row
            print(self.elu_cnt_row)

        print(f'\n>>>>> Increased temperature (row) to {self.temp_val_row} [{self.fct_temperature_turnup}] <<<<<')


    def incr_temp_col(self):
        if self.fct_temperature_turnup == 'linear' and self.temp_val_col <= self.max_temp_val_col:
            self.temp_val_col += self.temp_step_col
        elif self.fct_temperature_turnup == 'sigmoid':
            self.temp_val_col = self.max_temp_val_col * nn.functional.sigmoid((self.sig_cnt_col - self.sig_drift_col)/self.max_temp_val_col) + self.sig_offset_col
            self.sig_cnt_col += self.temp_step_col
            print(self.sig_cnt_col)
        elif self.fct_temperature_turnup == 'elu':
            elu = lambda x : self.elu_alpha_col * (torch.exp(x)-1)
            # elu = nn.ELU(alpha=self.elu_alpha_col) 
            if self.temp_val_col < self.max_temp_val_col:
                self.temp_val_col = elu(self.elu_cnt_col - self.elu_drift_col) + self.elu_alpha_col
                # self.temp_val_col = elu(self.elu_cnt_col/self.max_temp_val_col - self.elu_drift_col) + self.elu_alpha_col
            else:
                self.temp_val_col = self.max_temp_val_col
            self.elu_cnt_col += self.temp_step_col
            print(self.elu_cnt_col)

        print(f'\n>>>>> Increased temperature (col) to {self.temp_val_col} [{self.fct_temperature_turnup}] <<<<<')


    def incr_temp_relative_row(self, grads):
        self.temp_val_row += self.get_relative_temperature_step(grads)

        print(f'\n>>>>> Increased temperature (row) to {self.temp_val_row} [{self.fct_realtive_temperature_turnup}] <<<<<')


    def incr_temp_relative_col(self, grads):
        self.temp_val_col += self.get_relative_temperature_step(grads)

        print(f'\n>>>>> Increased temperature (col) to {self.temp_val_col} [{self.fct_realtive_temperature_turnup}] <<<<<')


    def get_relative_temperature_step(self, gradients): 
        if self.fct_realtive_temperature_turnup == 'lpfilter_deriv':
            t_filter_alpha = 0.5
            t_sign_alpha = 0.6
            grads_filtered = self.low_pass_filter(gradients)
            grads_sec_deriv = self.second_derivative(grads_filtered)

            # t_new = grads_sec_deriv[-1]
            t_new = grads_sec_deriv[-10:]
            t_new_sum = torch.sum(t_new)

            self.t_sign = torch.square(t_sign_alpha * self.t_sign + (1-t_sign_alpha) * torch.sign(t_new_sum))

            self.t = self.t_sign * (t_filter_alpha * self.t + (1-t_filter_alpha) * torch.abs(t_new_sum))

            print(f'New temperature increment: {self.t}')
            
            return self.t.item()

        else:
            if self.fct_realtive_temperature_turnup == 'sum_of_abs_diff':
                n = gradients.size()[0] - 1
                d_grads = gradients[:n] - gradients[-n:]
                t = torch.sum( torch.abs(torch.sum(d_grads, dim=0)), dim=0 )
            elif self.fct_realtive_temperature_turnup == 'mean':
                t = torch.mean(gradients)

            self.ts.append(t)
            if t < 0.0001:
                return self.temp_step_col
            else:
                print(gradients)
                print(self.ts)
                return 1/t


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





