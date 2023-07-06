"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""

# packet imports
from builtins import super
import numpy as np
import pandas as pd
import torch
import copy
from torch import nn, autograd
import matplotlib.pyplot as plt

import sys
import os
import csv

sys.path.append('C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Python Projects/Code/Binding Framework')
# Before run: replace ... with current directory path

# class imports
from BPAT_Inference import BPAT_Inference

# eigener Import
#from Testing.testing_module.TESTING_procedure_abstract import TEST_PROCEDURE


class Control_BPAT_NeckerCubeStatic(BPAT_Inference):
    """
    	Subclass of BPAT inference.

        Performs BPAT inference in parallel manner,
        i.e. parameters are inferred simultaneously in every tuning cycle.

        Temperature is turned up according to the parameters set in
            experiment_interface(_oi).

        Depth coordinates are predicted after a specified number of time_steps:
            - self.give_attractor: bool
                correct value for depth is given in the first forward pass of the tuning horizon.
            - self.attractor_span: int
                number of time steps for which the correct depth value is provided during inference.
        Different depth perception can be induced by providing specific depth information in
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

        ### depth prediction ###
        # attractor parameters
        self.give_attractor = True  # False  # macht True einen unterschied??
        print(f"PredZ inference with give_attractor={self.give_attractor}")
        self.attractor_span = 200  # 70 #0

        # self.pred_z = False
        self.pred_dim = 2
        # self.pred_z_loss = nn.SmoothL1Loss(reduction='sum', beta=0.001)
        # self.pred_z_loss = nn.MSELoss()
        self.pred_z_loss = lambda x, y: x - y  # x:input, y:target
        # self.pred_z_loss = lambda x, y : torch.sum(x-y)    # x:input, y:target

        # inducing different depth perception
        self.start_attractor_z = 210
        self.end_attractor_z = self.start_attractor_z + 80
        # self.attractor_feature = [0,1,2,9,10,11]
        # self.contra_feature = [3,4,5,12,13,14]
        self.attractor_feature = [0, 1, 2, 3, 4, 5, 6, 7]   # indices form which attractor feature should be taken from
        self.contra_feature = [7, 6, 5, 4, 3, 2, 1, 0]      # indices to which taken feature should be inserted

        # beschreibt reihenfolge der "ausgefüllten" y-Werte der neunen BM
        # für den Necker Cube:
        self.contra_order = torch.tensor(
            [7, 6, 5, 4,
             3, 2, 1, 0]
        ).to(self.device)

        super().__init__()


    def extract_z(self, pred):
        pred_rs = torch.transpose(
            pred.clone().detach().reshape(self.num_input_features, self.num_input_dimensions),
            0, 1)

        if self.num_input_dimensions == 3:
            return pred_rs[self.pred_dim]
        else:
            return torch.cat([pred_rs[self.pred_dim].unsqueeze(0), pred_rs[self.pred_dim + 3].unsqueeze(0)])


    def add_vis_marker(self, x):
        result = torch.empty(1,32)
        k = 0
        for j in range(32):
            if (j + 1) % 4 == 0:
                result[0, j] = 1
            else:
                result[0, j] = x[0, k]
                k += 1

        return result

    ############################################################################
    ##########  INFERENCE  #####################################################

    def run_inference(self,
                      observations,
                      do_binding,
                      do_rotation,
                      do_translation,
                      result_path
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
        print("#############################################################")
        print("Running NeckerCubeStatic Binding only........................")
        print("#############################################################")

        print('Shape of observations:')
        print(observations.shape)

        self.at_final_pred_errors_z = torch.tensor([]).to(self.device)
        at_final_predictions = torch.tensor([]).to(self.device)
        at_final_inputs = torch.tensor([]).to(self.device)
        at_optimal_inputs = torch.tensor([]).to(self.device)

        at_bin_gradients = torch.tensor([]).to(self.device)
        at_rot_gradients = torch.tensor([]).to(self.device)
        at_trans_gradients = torch.tensor([]).to(self.device)

        z_coord = torch.zeros(8).to(self.device)

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
        #### Resetting Binding Matrix and Cube Constellation.
        reset_frame = 1000 #500

        #######################################################################
        #### create observations without visibility markers, vis markers are only used in model

        observations_vis = observations.clone()
        #o1_vis = observations_vis[0]
        #o2_vis = observations_vis[1]
        print(observations)
        observations = np.delete(observations, -1, axis=2)
        #observations[0] = np.delete(observations[0], list(range(3, observations[0].shape[1], 4)), axis=1)
        #observations[1] = np.delete(observations[1], list(range(3, observations[1].shape[1], 4)), axis=1)
        print(observations)


        ###############################################################################################
        ############ Setting Cube Observations ########################################################
        ###############################################################################################
        o1 = observations[0]   # NC Variante 1
        o2 = observations[1]  # NC Variante 2

        print(observations)
        print(o1)

        o1_without_z = torch.clone(o1)
        o2_without_z = torch.clone(o2)
        o2_with_o1_z = torch.clone(o2)
        o1_switched = torch.clone(o1)
        o1_switched[0] = o1[3]
        o1_switched[1] = o1[2]
        o1_switched[2] = o1[1]
        o1_switched[3] = o1[0]
        o1_switched[4] = o1[4]
        o1_switched[5] = o1[5]
        o1_switched[6] = o1[6]
        o1_switched[7] = o1[7]

        for corner in o1_without_z:
            corner[2] = 0
        for corner in o2_without_z:
            corner[2] = 0

        for i in range(8):
         o2_with_o1_z[i][2] = o1[7-i][2]

        print("Cube 1:")
        print(o1)
        print("Cube 1 without z:")
        print(o1_without_z)
        print("Cube 1 switched:")
        print(o1_switched)
        print("Cube 2:")
        print(o2)

    #    print("Cube 2:")
    #    print(o2)
    #    print("Cube 2 with z values of Cube 1:")
    #    print(o2_with_o1_z)

        ## print the inintial binding matrix]
        feature_names = ['LefDowFar', 'RigDowFar', 'RigUpFar', 'LefUpFar', 'LefUpClo', 'RigUpClo', 'RigDowClo',
                         'LefDowClo']

        ########################### Printing and Saving Initial binding Matrix ########################
        print("Initial Binding Matrix:")
        print(self.Bs[0])
        ## plotting the initial binding matrix and writing it to png file .file.
        initial_matrix = self.evaluator.plot_binding_matrix(
            self.Bs[0],  # final_binding_matrix,
            feature_names,
            'Binding matrix showing relative contribution of observed feature to input feature')

        initial_matrix.savefig(result_path + "initial_binding_matrix_neuron_act.png")


        #o1_flat =  self.preprocessor.convert_data_AT_to_VAE(o1)
        #test = self.core_model.forward(o1_flat.float(), "testing")
        #print(self.preprocessor.convert_data_VAE_to_AT(test))
        ##print(o1)

        rec_losses = []
        fbe = []
        #noise_cyles = list(range(0, 400))
        #for i in (range(reset_frame, reset_frame+400)):
        #    noise_cyles.append(i)
#########################################################################################################################
        for cycle in range(self.tuning_cycles):  # es wird dann immer der gleiche Würfel verwendet

            ### wechsel des input cubes:
            if cycle < reset_frame:
                o = o1_without_z
                #o = o1
                o_target = o1
                o_target_flat = self.preprocessor.convert_data_AT_to_VAE(o_target)
            elif cycle >= reset_frame:
                o = o2_without_z
                #o = o2
                o_target = o1
                o_target_flat = self.preprocessor.convert_data_AT_to_VAE(o_target)

            # calculate and print ORE (optimal reconstruction error)
            if cycle == 0:
                ideal_binding_matrix = self.ideal_binding
                ore_x, useless_bm = self.perform_bpt_binding_only(True, idx=0, obs=o_target, bm=ideal_binding_matrix)
                print(type(ore_x))
                print(ore_x)
                print(ore_x.shape)
                ore_x_vis = self.add_vis_marker(ore_x)
                print(type(ore_x_vis))
                print(ore_x_vis)
                ore_pred = self.core_model.forward(ore_x_vis, "testing")
                ore_pred_masked = torch.clone(ore_pred)
                o_target_flat_masked = torch.clone(o_target_flat)
                for i in range(2, ore_pred_masked.size(dim=1), 3):
                    ore_pred_masked[0][i] = 0
                    o_target_flat_masked[0][i] = 0

                ORE = self.at_loss(ore_pred.float(), o_target_flat.float())
                print('Optimal Reconstruction Error: ' + str(ORE))
                ORE_masked = self.at_loss(ore_pred_masked.float(), o_target_flat_masked.float())
                print('Optimal Reconstruction Error without z: ' + str(ORE_masked))


            ######## set z-coordiantes of observation to predicted z-coordintates of last cycle ########
            if (cycle != 0) and (cycle != reset_frame):
                for i in range(o.size(dim=0)):
                    prev_predicted_z = self.preprocessor.convert_data_VAE_to_AT(upd_prediction)[i][2].item()
                    o[i][2] = prev_predicted_z

                    # add noise to observation
                    if True: # changed noise adding here
                        o[i][0] += np.random.normal(0, 1) * 0.01
                        o[i][1] += np.random.normal(0, 1) * 0.01

            #########################################################################################
            print('------------- Tuning Cycle: ' + str(cycle) + ' of ' + str(
                self.tuning_cycles) + '---------------------------------')

            #########################################################################################
            x, bm = self.perform_bpt_binding_only(do_binding, idx=0, obs=o)

            x_vis = self.add_vis_marker(x)
            #########################################################################################
            upd_prediction = self.core_model.forward(x_vis, "testing")
            #upd_prediction = upd_prediction.squeeze()

#            print("Input to Binding:")
#            print(o)
#            print("Binding matrix:")
#            print(bm)
#            print("Input to VAE:")
#            print(self.preprocessor.convert_data_VAE_to_AT(x))
#            print("Prediction forward:")
#            print(self.preprocessor.convert_data_VAE_to_AT(upd_prediction))


           ################ Calculate error  ############################################
           ############### create masked predictions and targets for loss ###############
            upd_prediction_masked = torch.clone(upd_prediction)
            o_target_flat_masked = torch.clone(o_target_flat)

            for i in range(2, upd_prediction_masked.size(dim=1), 3):
                upd_prediction_masked[0][i] = 0
                o_target_flat_masked[0][i] = 0
            ############################################################################

            #loss = self.at_loss(upd_prediction.float(), o_target_flat.float())      #MSE

            #loss with calculated with mask on z-values
            loss = self.at_loss(upd_prediction_masked.float(), o_target_flat_masked.float())

            print(f'frame: {self.obs_count} cycle: {cycle} loss: {loss}')
            rec_losses.append(loss.item())

            # Propagate error back through tuning horizon
            loss.backward()

            ############ UPDATE PARAMETERS
            ###############################################################################################
            with torch.no_grad():
                # self.at_losses.append(loss.clone().detach())
                self.at_losses.append(loss.clone().detach().cpu())

                ###########################  BINDING  #################################

                grad_B = self.get_grad_binding(self.grad_calc_binding).to(self.device)  # Derive the gradient of the binding matrix.
                at_bin_gradients = torch.cat(
                    [at_bin_gradients,
                     grad_B.reshape(1, self.num_observations * self.num_input_features)]
                )
                upd_B = self.binder.update_binding_matrix(
                    self.Bs[0],
                    grad_B,
                    self.at_learning_rate_binding,  # anpassen wie sie BM beeinflusst
                    self.bm_momentum,
                    True,
                    self.b_signdamp
                )

                # Temperature till reset_frame decrease then set back to initial temperature then decrease again
                if cycle != reset_frame:  # reset_frame
                    self.binder.decr_temp_col_linear()
                    self.binder.decr_temp_row_linear()
                elif cycle == reset_frame:
                    # reset temperature
                    self.binder.reset_temp()
                    print("Temperature has been reset")

                    #reset ideal Binding Matrix
                    self.ideal_binding = self.ideal_binding.gather(
                        0, self.contra_order.unsqueeze(1).expand(self.ideal_binding.shape))  # ideal BM geändert
                    print(f'Reset ideal binding matrix to contra matrix: \n{self.ideal_binding}')

                    ########################


                self.loss_binding(upd_B, fbe)

                #####################################################################
                matrix = self.binder.scale_binding_matrix(self.Bs[0], self.scale_mode)
                #####################################################################
                ## plotting binding matrix after each %100 cycle
                if cycle % 100 == 0:
                    name = "Binding_Matrix_frame_" + str(self.obs_count) + "_cycle_" + str(cycle) + ".png"

                    #matrix = self.binder.scale_binding_matrix(self.Bs[0], self.scale_mode)
                    matrix = self.evaluator.plot_binding_matrix(matrix, feature_names,
                                                                'Binding matrix showing relative contribution of observed feature to input feature')

                    matrix.savefig(result_path + name)
                    plt.close(matrix)
                    #########################################

                    # Compare binding matrix to ideal matrix
                    print("New Binding Matrix: ")
                    torch.set_printoptions(precision=4)
                    print(upd_B)

                # save outcast-line gradients       #nicht relevant
                # if self.nxm:
                #     self.oc_grads.append(grad_B[-1].cpu())

                # Zero out gradients for all parameters in all time steps of tuning horizon
                for i in range(self.tuning_length + 1):
                    self.Bs[i].requires_grad = False
                    self.Bs[i].grad.data.zero_()
                    self.Bs[i].data = upd_B.clone().data            #hier wird self.Bs auf neue BM gesetzt
                    self.Bs[i].requires_grad = True

        #######################################################################
        #  END of tuning cycles
        #######################################################################

        filename = result_path + "/reconstruction_losses.txt"
        with open(filename, "w") as f:
            for i, loss in enumerate(rec_losses):
                print(i + 1, loss, file=f)
        f.close()

        filename = result_path + "/feature_binding_losses.txt"
        with open(filename, "w") as f:
            for i, loss in enumerate(fbe):
                print(i + 1, loss, file=f)
        f.close()

        print("FINALIZING STUFF NOW:... ")
        with torch.no_grad():
            # extract optimal values
            print("FINALIZING STUFF NOW:... ")
            opt_z_x = self.binder.bind(o, self.ideal_binding)
            if self.obs_count >= self.start_attractor_z:
                opt_z_x[:, self.pred_dim] = - opt_z_x[:, self.pred_dim]

            opt_z = self.extract_z(opt_z_x).to(self.device)

            ## Save final prediction errors
            final_pred_error_z = self.pred_z_loss(z_coord, opt_z)
            print(f"z-pred-loss: {final_pred_error_z}")
            print(f"z-coord: {z_coord}")

            path = result_path + "prediction_errors.csv"
            print("Writing prediction errors to: "+path)
            df = pd.DataFrame(final_pred_error_z)
            df.to_csv(path)

            fig = self.evaluator.plot_prediction_errors(final_pred_error_z)
            figname = "prediction_errors_frame_" + str(self.obs_count) + ".png"
            fig.savefig(result_path + figname)

            final_prediction = upd_prediction
            final_input = x.clone().detach().to(self.device)
            final_opt_input = opt_z_x.detach().to(self.device)

            #self.intersave_matrices(bm, self.Bs[0], self.obs_count - self.tuning_length)

        ## Generate updated prediction
        new_prediction = self.core_model(x_vis, "testing")
        #z_coord = self.extract_z(new_prediction)

        ## Reorganize storage variables
        # observations
        self.at_observations = torch.cat(
            (self.at_observations[1:],
             o.reshape(1, self.num_observations, self.num_input_dimensions)), 0)

        # lstm inputs
        self.at_model_inputs = torch.cat((self.at_model_inputs[1:], x), 0)

        at_final_inputs = torch.cat(
            (at_final_inputs, final_input.reshape(1, self.input_per_frame)),
            0
        )

        at_optimal_inputs = torch.cat(
            (at_optimal_inputs, final_opt_input.reshape(1, self.input_per_frame)),
            0
        )

        # predictions
        at_final_predictions = torch.cat(
            (at_final_predictions, final_prediction.reshape(1, self.input_per_frame)),
            0
        )

        self.at_final_pred_errors_z = torch.cat(
            (self.at_final_pred_errors_z, final_pred_error_z.reshape(1, -1)),
            0
        )

        self.at_predictions = torch.cat(
            (self.at_predictions[1:],
             new_prediction.reshape(1, self.input_per_frame)), 0)

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
            print(f'final binding entries: {final_binding_entries}')

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
