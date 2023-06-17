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
sys.path.append('/Users/MartinButz2/Documents/CODE/Python/BindingDancersAndCubes/BindingAndPerspectiveTaking-Research-main')
# Before run: replace ... with current directory path

# class imports 
from BPAT_Inference import BPAT_Inference


class BPAT_PARALLEL_TEMPTURNUP_BM_FORCE(BPAT_Inference):

 
    """
    	Subclass of BPAT inference. 

        Performs BPAT inference in parallel manner,
        i.e. parameters are infered simultaniously in every tuning cycle. 

        Temperature is turned up according to the parameters set in
            experiment_interface(_oi). 

        Binding matrix can be fixed in specific initialization. (see experiment_interface(_oi)).
            Fixed binding matrix can be hold for start_BM_infer time steps. 
            After that, inference of binding activations starts. 

        Depth coordinates are predicted after a specified number of time_steps: 
            - self.give_attractor: bool
                correct value for depth is given in the first forward pass of the tuning horizon.
            - self.attractor_span: int
                number if time steps for which the correct depth value is provided during inference.
        Different depth perception can be induced by providing a specific depth information in
            corresponding contra features. Values are exchanged in set_z_contra_attractor. 
            - self.start_attractor_z: int 
                time step defining the start of the period over which the depth information is 
                provided
            - self.end_attractor_z: int 
                time step defining the end of the period over which the depth information is 
                provided
            - self.attractor_feature: [int]
                list of indices specifying the features the depth information cues should be taken from
            - self.contra_feature: [int]
                list of indices specifying the features in which the depth information cues should be 
                induced. Order corresponds to the list of attractor information.  

        Note that the different mechanisms can also be tested separately. For this, the corresponding 
            parameters must be set accordingly.         

    """


    def __init__(self):
        ## General parameters 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        autograd.set_detect_anomaly(True)

        torch.set_printoptions(precision=8)

        ### BM inference ###
        self.start_BM_infer = 80 + 10
        # self.start_BM_infer = 100

        ### depth prediction ###
        # attractor parameters
        self.give_attractor = False
        print(f"PredZ inference with give_attractor={self.give_attractor}")
        self.attractor_span = 0

        # self.pred_z = False
        self.pred_dim = 2
        # self.pred_z_loss = nn.SmoothL1Loss(reduction='sum', beta=0.001)
        # self.pred_z_loss = nn.MSELoss()
        self.pred_z_loss = lambda x, y : x-y    # x:input, y:target
        # self.pred_z_loss = lambda x, y : torch.sum(x-y)    # x:input, y:target

        # inducing different depth perception
        self.start_attractor_z = 210
        self.end_attractor_z = self.start_attractor_z + 80
        # self.attractor_feature = [0,1,2,9,10,11]
        # self.contra_feature = [3,4,5,12,13,14]
        self.attractor_feature = [14]
        self.contra_feature = [11]

        self.contra_order = torch.tensor(
            [3, 4, 5, 
             0, 1, 2, 
             6, 7, 8, 
             12, 13, 14, 
             9, 10, 11]
        ).to(self.device)


        super().__init__()

    
    def exchange_coord(self, obs, exc_vals):
        obs = self.preprocessor.convert_data_CoreModel_to_AT(obs)
        obs_t = torch.transpose(obs, 0, 1)

        if self.num_input_dimensions == 3: 
            index = torch.tensor([self.pred_dim]).to(self.device)
            obs_d = obs_t.index_copy(0, index, exc_vals.unsqueeze(0))
        elif self.num_input_dimensions == 6: 
            index = torch.tensor([self.pred_dim, self.pred_dim+3]).to(self.device)
            obs_d = obs_t.index_copy(0, index, exc_vals)
        else: 
            print('ERROR: Inference of coordinates only defined for 3 or 6 input dimensions.')

        obs_d = torch.transpose(obs_d, 0, 1)
        return self.preprocessor.convert_data_AT_to_CoreModel(obs_d)


    # def set_pred_dim(self, pred_bool):
    #     self.pred_z = pred_bool


    def update_dimension_values(self):
        upd_d = []

        for i in range(self.tuning_length+1):
            mom = self.d_momentum*self.bin_momentum_dimension[i]        
            upd = - self.dim_learning_rate * self.D_grads[i] + mom

            self.bin_momentum_dimension[i] = upd
            upd_d.append(self.Ds[i] + upd)

        return upd_d


    def extract_z(self, pred):
        pred_rs = torch.transpose(
            pred.clone().detach().reshape(self.num_input_features, self.num_input_dimensions), 
            0,1)

        if self.num_input_dimensions==3: 
            return pred_rs[self.pred_dim]
        else: 
            return torch.cat([pred_rs[self.pred_dim].unsqueeze(0), pred_rs[self.pred_dim+3].unsqueeze(0)])

        # if self.num_input_dimensions==3: 
        #     return torch.Tensor([pred_rs[self.pred_dim].numpy()])
        # else: 
        #     return torch.Tensor([pred_rs[self.pred_dim].numpy(), pred_rs[self.pred_dim+3].numpy()])
        # CPU


    def set_z_contra_attractor(self, z_coord, obs):
        
        if self.num_input_dimensions==3: 
            pred_dim_indx = [self.pred_dim]
        else:
            pred_dim_indx = [self.pred_dim, self.pred_dim+3]
        
        # shape z_coord: [2, num_features]
        # shape obs:       [num_features, num_dimensions]

        for cf, af in zip(self.contra_feature, self.attractor_feature):
            z_coord[:, cf] = -obs[af, pred_dim_indx]

        return z_coord



    ############################################################################
    ##########  INFERENCE  #####################################################
    
    def run_inference(self, 
            observations, 
            do_binding, 
            do_rotation,
            do_translation
        ):

        """
            Performs BPAT inference of the binding and perspective taking parameters. 
            
            Parameters: 
                - observations: tensor of shape (time steps, observed features, dimensions)
                    observational input for multiple time steps. 
                    Note that the number of observations must always exceed the length of the tuning horizon. 
                - do_binding, do_rotation, do_translation: bool
                    indicators whether the respective tasks should be performed

            Returns list of results, containing: 
                - at_final_inputs
                - at_final_predictions 
                - final_binding_matrix 
                - final_binding_entries
                - final_rotation_values 
                - final_rotation_matrix
                - final_translation_values

        """

        print(observations.shape)

        if self.bm_version == "contra":
            # order = torch.tensor([3, 4, 5, 0, 1, 2, 6, 7, 8, 12, 13, 14, 9, 10, 11])
            # observations = observations.gather(1,order.unsqueeze(1).expand(observations.shape))

            observations[:,:,self.pred_dim] *= -1 
            observations[:,:,self.pred_dim+3] *= -1

        
        self.at_final_pred_errors_z = torch.tensor([]).to(self.device) 
        at_final_predictions = torch.tensor([]).to(self.device)
        at_final_inputs = torch.tensor([]).to(self.device)
        at_optimal_inputs = torch.tensor([]).to(self.device)
        
        at_bin_gradients = torch.tensor([]).to(self.device)
        at_rot_gradients = torch.tensor([]).to(self.device)
        at_trans_gradients = torch.tensor([]).to(self.device)

        if self.num_input_dimensions == 3: 
            z_coord = torch.zeros(self.num_observations).to(self.device)
        else: 
            z_coord = torch.zeros(2, self.num_observations).to(self.device)

        temp_range_col = 1
        temp_range_row = 1


        ###############################################################################################
        ############ INITIALIZATIONS
        # Initialize variables for binding matrices, rotation parameters, and/or translation bias.
        ###############################################################################################

        ###########################  BINDING  #################################
        if do_binding:
            self.init_binding()
            
        ###########################  ROTATION  ################################
        if do_rotation:
            self.init_rotation()

        ###########################  TRANSLATION  #############################
        if do_translation:
            self.init_translation()

        #######################################################################

        ## Core state
        # define scaler
        state_scaler = 1.0

        # init state
        at_h = torch.zeros(1, self.core_model.hidden_size).to(self.device)
        at_c = torch.zeros(1, self.core_model.hidden_size).to(self.device)

        at_h.requires_grad = True
        at_c.requires_grad = True

        init_state = (at_h, at_c)
        state = (init_state[0], init_state[1])

        ###############################################################################################
        ############ FORWARD PASS
        # Perform one forward pass on the first H observations of the first tuning horizon. 
        ###############################################################################################

        for i in range(self.tuning_length):
            o = observations[self.obs_count].to(self.device)

            self.at_observations = torch.cat((
                self.at_observations, 
                o.reshape(1, self.num_observations, self.num_input_dimensions)), 0)
            self.obs_count += 1

            ###########################  BPT  #####################################
            x, bm, rotmat = self.perform_bpt(do_binding, do_rotation, do_translation, i, o)
            #######################################################################
            
            if self.give_attractor:
                x = x
            else:
                x = self.exchange_coord(x, z_coord)

            state = (state[0] * state_scaler, state[1] * state_scaler)
            new_prediction, state = self.core_model(x, state)

            if not self.give_attractor:
                z_coord = self.extract_z(new_prediction)  

            self.at_states.append(state)
            self.at_predictions = torch.cat((self.at_predictions, new_prediction.reshape(1,self.input_per_frame)), 0)
            self.at_lstm_inputs = torch.cat((self.at_lstm_inputs, x), 0)


        if self.give_attractor and self.attractor_span == 0:
            z_coord = self.extract_z(new_prediction)


        ###############################################################################################
        ############ ACTIVE TUNING
        # Perform active tuning algorithm for the rest of the samples. 
        ###############################################################################################

        while self.obs_count < self.num_frames:  
            o = observations[self.obs_count].to(self.device)     

            ###########################  BPT  #####################################
            x, bm, rotmat = self.perform_bpt(do_binding, do_rotation, do_translation, -1, o)
            #######################################################################

            if self.obs_count >=  self.attractor_span:

                if self.obs_count == self.start_attractor_z:
                    self.ideal_binding = self.ideal_binding.gather(
                        0,
                        self.contra_order.unsqueeze(1).expand(self.ideal_binding.shape)
                    )

                    print(f'Reset ideal binding matrix to contra matrix: \n{self.ideal_binding}')

                if self.obs_count >= self.start_attractor_z and self.obs_count<self.end_attractor_z:
                    z_coord = self.set_z_contra_attractor(z_coord, o)

                x = self.exchange_coord(x, z_coord)
                    
            else:
                x = x


            ## Generate current prediction 
            with torch.no_grad():
                state = self.at_states[-1]
                state = (state[0] * state_scaler, state[1] * state_scaler)
                new_prediction, state = self.core_model(x, state)

            ## For #tuning_cycles 
            for cycle in range(self.tuning_cycles):
                print('----------------------------------------------')

                # Get prediction
                p = self.at_predictions[-1]

                # Calculate error 
                targets = torch.cat([self.at_lstm_inputs.view(self.tuning_length,self.input_per_frame)[1:], x], dim=0)
                loss = self.at_loss(self.at_predictions, targets)

                # Propagate error back through tuning horizon 
                loss.backward() 

                ############ UPDATE PARAMETERS
                ###############################################################################################
                with torch.no_grad():
                    # self.at_losses.append(loss.clone().detach())
                    self.at_losses.append(loss.clone().detach().cpu())
                    print(f'frame: {self.obs_count} cycle: {cycle} loss: {loss}')

                    ###########################  BINDING  #################################
                    if do_binding:

                        grad_B = self.get_grad_binding(self.grad_calc_binding).to(self.device)
                        if self.nxm:
                            at_bin_gradients = torch.cat(
                                [at_bin_gradients, 
                                grad_B.reshape(1,self.num_observations*self.num_input_features+self.num_observations)]
                            )
                        else:
                            at_bin_gradients = torch.cat(
                                [at_bin_gradients, 
                                grad_B.reshape(1,self.num_observations*self.num_input_features)]
                            )
                        
                        if self.obs_count >= self.start_BM_infer:
                            # Update parameters in time step t-H with saved gradients 
                            upd_B = self.binder.update_binding_matrix(
                                self.Bs[0], 
                                grad_B, 
                                self.at_learning_rate_binding, 
                                self.bm_momentum, 
                                True, 
                                self.b_signdamp
                            )
                        else:
                            upd_B = self.Bs[0].clone().detach()

                        # Compare binding matrix to ideal matrix
                        self.loss_binding(upd_B)

                        # save outcast-line gradients 
                        if self.nxm:
                            self.oc_grads.append(grad_B[-1].cpu())

                        # Zero out gradients for all parameters in all time steps of tuning horizon
                        for i in range(self.tuning_length+1):
                            self.Bs[i].requires_grad = False
                            self.Bs[i].grad.data.zero_()

                        # Update all parameters for all time steps 
                        for i in range(self.tuning_length+1):
                            self.Bs[i].data = upd_B.clone().data
                            self.Bs[i].requires_grad = True

                    ###########################  ROTATION  ################################
                    if do_rotation:
                        
                        grad_R = self.get_grad_rotation(self.grad_calc_rotation).to(self.device)

                        if self.rotation_type == 'qrotate': 
                            at_rot_gradients = torch.cat([at_rot_gradients, grad_R.reshape(1,4)])
                            # Update parameters in time step t-H with saved gradients 
                            upd_R = self.perspective_taker.update_quaternion(
                                self.Rs[0], grad_R, self.at_learning_rate_rotation, self.r_momentum, True, self.r_signdamp)
                            print(f'updated quaternion: {upd_R}')

                            # Zero out gradients for all parameters in all time steps of tuning horizon
                            for i in range(self.tuning_length+1):
                                self.Rs[i].requires_grad = False
                                self.Rs[i].grad.data.zero_()

                            # Update all parameters for all time steps 
                            for i in range(self.tuning_length+1):
                                quat = upd_R.clone()
                                quat.requires_grad_()
                                self.Rs[i] = quat

                        else: 
                            at_rot_gradients = torch.cat([at_rot_gradients, grad_R.reshape(1,3)])
                            # Update parameters in time step t-H with saved gradients 
                            upd_R = self.perspective_taker.update_rotation_angles_(
                                self.Rs[0], grad_R, self.at_learning_rate_rotation, self.r_momentum)
                            print(f'updated angles: {upd_R}')
                            
                            # Zero out gradients for all parameters in all time steps of tuning horizon
                            for i in range(self.tuning_length+1):
                                for j in range(self.num_spatial_dimensions):
                                    self.Rs[i][j].requires_grad = False
                                    self.Rs[i][j].grad.data.zero_()

                            # Update all parameters for all time steps 
                            for i in range(self.tuning_length+1):
                                angles = []
                                for j in range(3):
                                    angle = upd_R[j].clone()
                                    angle.requires_grad_()
                                    angles.append(angle)
                                self.Rs[i] = angles

                        # Calculate and save rotation losses
                        self.loss_rotation(upd_R)
                    
                    ###########################  TRANSLATION  #############################
                    if do_translation:
                        
                        grad_C = self.get_grad_translation(self.grad_calc_translation).to(self.device)
                        at_trans_gradients = torch.cat([at_trans_gradients, grad_C.reshape(1,3)])
                        
                        # Update parameters in time step t-H with saved gradients 
                        upd_C = self.perspective_taker.update_translation_bias_(
                            self.Cs[0], grad_C, self.at_learning_rate_translation, self.c_momentum, True, self.c_signdamp)
                        
                        # Compare translation bias to ideal bias
                        self.loss_translation(upd_C)
                        
                        # Zero out gradients for all parameters in all time steps of tuning horizon
                        for i in range(self.tuning_length+1):
                            self.Cs[i].requires_grad = False
                            self.Cs[i].grad.data.zero_()
                        
                        # Update all parameters for all time steps 
                        for i in range(self.tuning_length+1):
                            translation = upd_C.clone()
                            translation.requires_grad_()
                            self.Cs[i] = translation

                    #######################################################################

                    # Initial state
                    g_h = at_h.grad.to(self.device)
                    upd_h = init_state[0] - self.at_learning_rate_state * g_h
                    at_h.data = upd_h.clone().detach().requires_grad_()
                    at_h.grad.data.zero_()

                    # print(f'updated init_state: {init_state}')
                

                ############ FORWARD PASS
                # Perform forward pass from t-H to t with new parameters. 
                ###############################################################################################
                init_state = (at_h, at_c)
                state = (init_state[0], init_state[1])

                z_coord = self.extract_z(self.at_lstm_inputs[0].detach()) 

                self.at_predictions = torch.tensor([]).to(self.device)
                self.at_lstm_inputs = torch.tensor([]).to(self.device)

                for i in range(self.tuning_length):

                    at_o = self.at_observations[i]

                    ###########################  BPT  #####################################
                    x, bm, rotmat = self.perform_bpt(do_binding, do_rotation, do_translation, i, at_o)
                    #######################################################################

                    if self.obs_count > self.attractor_span:

                        if self.obs_count > self.start_attractor_z:
                            z_coord = self.set_z_contra_attractor(z_coord,at_o)

                        x = self.exchange_coord(x, z_coord)

                    else:
                        x = x

                    state = (state[0] * state_scaler, state[1] * state_scaler)
                    upd_prediction, state = self.core_model(x, state)

                    self.at_predictions = torch.cat((self.at_predictions, upd_prediction.reshape(1,self.input_per_frame)), 0)
                    self.at_lstm_inputs = torch.cat((self.at_lstm_inputs, x), 0)
                    
                    # for last tuning cycle update initial state to track gradients 
                    if cycle==(self.tuning_cycles-1) and i==0: 
                        at_h = state[0].clone().detach().requires_grad_().to(self.device)
                        at_c = state[1].clone().detach().requires_grad_().to(self.device)
                        init_state = (at_h, at_c)
                        state = (init_state[0], init_state[1])

                    
                    # if self.obs_count > self.attractor_span:
                    z_coord = self.extract_z(upd_prediction) 

                    self.at_states[i] = state 


                # Update current input
                ###########################  BPT  #####################################
                x, bm, rotmat = self.perform_bpt(do_binding, do_rotation, do_translation, -1, o) 
                #######################################################################

                if self.obs_count > self.attractor_span:

                    if self.obs_count > self.start_attractor_z:
                        z_coord = self.set_z_contra_attractor(z_coord,o)

                    x = self.exchange_coord(x, z_coord)
                

                

            # END tuning cycle
            #######################################################################        

            with torch.no_grad():
                # extract optimal values
                opt_z_x = self.binder.bind(o, self.ideal_binding)
                if self.obs_count >=  self.start_attractor_z:
                    opt_z_x[:,self.pred_dim] = - opt_z_x[:,self.pred_dim]
                    
                opt_z = self.extract_z(opt_z_x).to(self.device)

                ## Save final prediction errors
                final_pred_error_z = self.pred_z_loss(z_coord, opt_z)
                print(f"z-pred-loss: {final_pred_error_z}")
                print(f"z-coord: {z_coord}")
                # TODO !!!!!!!!!!!!!!!!
                final_prediction = upd_prediction
                final_input = x.clone().detach().to(self.device)
                final_opt_input = opt_z_x.detach().to(self.device)
                # final_prediction = self.at_predictions[0].clone().detach().to(self.device)
                # final_input = x.clone().detach().to(self.device)


                ## Save FBM and FBA every TL steps
                if self.obs_count % 1 == 0:
                    self.intersave_matrices(bm, self.Bs[0], self.obs_count-self.tuning_length)

                
                ## Update temperature for binding if turned up
                if self.temp_turnup:
                    if self.obs_count == self.temp_reset_frame:
                        self.binder.reset_temp()

                    else:
                        if temp_range_col == self.range_temp_turnup_col:
                            if self.binder.temp_change=="step_up":
                                self.binder.incr_temp_col()
                            elif self.binder.temp_change=="smooth_down":
                                self.binder.decr_temp_col(self.range_temp_turnup_col)

                            temp_range_col = 1
                        else:
                            temp_range_col += 1

                        if temp_range_row == self.range_temp_turnup_row:
                            if self.binder.temp_change=="step_up":
                                self.binder.incr_temp_row()
                            elif self.binder.temp_change=="smooth_down":
                                self.binder.decr_temp_row(self.range_temp_turnup_row)

                            temp_range_row = 1
                        else: 
                            temp_range_row += 1


            ## Generate updated prediction 
            state = self.at_states[-1]
            state = (state[0] * state_scaler, state[1] * state_scaler)
            new_prediction, state = self.core_model(x, state)
            z_coord = self.extract_z(new_prediction) 


            ## Reorganize storage variables            
            # observations
            self.at_observations = torch.cat(
                (self.at_observations[1:], 
                o.reshape(1, self.num_observations, self.num_input_dimensions)), 0)
            
            # lstm inputs 
            self.at_lstm_inputs = torch.cat((self.at_lstm_inputs[1:], x), 0)

            at_final_inputs = torch.cat(
                (at_final_inputs, final_input.reshape(1,self.input_per_frame)), 
                0
            )

            at_optimal_inputs = torch.cat(
                (at_optimal_inputs, final_opt_input.reshape(1,self.input_per_frame)), 
                0
            )

            # predictions
            at_final_predictions = torch.cat(
                (at_final_predictions, final_prediction.reshape(1,self.input_per_frame)), 
                0
            )

            self.at_final_pred_errors_z = torch.cat(
                (self.at_final_pred_errors_z, final_pred_error_z.reshape(1,-1)), 
                0
            )
        
            self.at_predictions = torch.cat(
                (self.at_predictions[1:], 
                new_prediction.reshape(1,self.input_per_frame)), 0)


            ##############################################
            self.obs_count += 1



        # END active tuning
        #######################################################################


        ###############################################################################################
        ############ FINISH INFERENCE
        ###############################################################################################
        
        ### store rest of predictions in at_final_predictions ###

        # for i in range(self.tuning_length): 
            # at_final_predictions = torch.cat(
            #     (at_final_predictions, 
            #     self.at_predictions[i].reshape(1,self.input_per_frame)), 0)

            # ###########################  BPT  #####################################
            # x_F, bm, rotmat = self.perform_bpt(do_binding, do_rotation, do_translation, i, self.at_observations[i], bm, rotmat)
            # #######################################################################
            
            # at_final_inputs = torch.cat(
            #     (at_final_inputs, 
            #     x_F.reshape(1,self.input_per_frame)), 0)


        ###########################  BINDING  #################################
        # get final binding matrix
        if do_binding:
            final_binding_matrix = self.binder.scale_binding_matrix(
                self.Bs[-1].clone().detach(), self.scale_mode)
            print(f'final binding matrix: {final_binding_matrix}')
            final_binding_entries = self.Bs[-1].clone().detach()
            print(f'final binding entires: {final_binding_entries}')

        else: 
            final_binding_entries, final_binding_matrix = None, None

        ###########################  ROTATION  ################################
        # get final rotation matrix
        if do_rotation:
            if self.rotation_type == 'qrotate': 
                final_rotation_values = self.Rs[0].clone().detach()
                # get final quaternion
                print(f'final quaternion: {final_rotation_values}')
                final_rotation_matrix = self.perspective_taker.quaternion2rotmat(final_rotation_values)
            else:
                final_rotation_values = [
                    self.Rs[0][i].clone().detach() 
                    for i in range(self.num_spatial_dimensions)]
                print(f'final euler angles: {final_rotation_values}')
                final_rotation_matrix = self.perspective_taker.compute_rotation_matrix_(
                    final_rotation_values[0], 
                    final_rotation_values[1], 
                    final_rotation_values[2])
            
            print(f'final rotation matrix: \n{final_rotation_matrix}')

        else: 
            final_rotation_matrix, final_rotation_values = None, None

        ###########################  TRANSLATION  #############################
        # get final translation bias
        if do_translation:
            final_translation_values = self.Cs[0].clone().detach()
            print(f'final translation bias: {final_translation_values}')

        else: 
            final_translation_values = None

        #######################################################################


        return [at_final_inputs,
                at_optimal_inputs,
                at_final_predictions, 
                final_binding_matrix, 
                final_binding_entries,
                final_rotation_values, 
                final_rotation_matrix, 
                final_translation_values,
                at_bin_gradients.cpu(), 
                at_rot_gradients.cpu(),
                at_trans_gradients.cpu()]




    ############################################################################
    ##########  EVALUATION #####################################################

    def get_result_history(
        self, 
        optimal_inputs, 
        at_final_predictions):

        if self.nxm:
            pred_errors = self.evaluator.prediction_errors_nxm(
                optimal_inputs, 
                self.additional_features, 
                self.num_observations, 
                at_final_predictions[:-1], 
                self.mse
            )
            self.bm_losses = torch.stack(self.bm_losses)

        else:

            pred_errors = self.evaluator.prediction_errors(
                optimal_inputs[:-1], 
                at_final_predictions[1:], 
                self.mse)
                
        return [pred_errors,
                self.at_losses, 
                self.at_final_pred_errors_z, 
                self.bm_dets, 
                self.bm_losses, 
                self.rm_losses, 
                self.rv_losses,
                self.c_losses]

