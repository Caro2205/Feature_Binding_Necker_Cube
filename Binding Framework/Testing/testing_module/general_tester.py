"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""

import numpy as np
import torch 
import os


from Testing.testing_module.TESTING_procedure_abstract import TEST_PROCEDURE
from BPAT_subclass_parallel import BPAT_PARALLEL
from BPAT_subclass_sequential import BPAT_SEQUENTIAL
from BPAT_subclass_parallel_tempTurnUp import BPAT_PARALLEL_TEMPTURNUP
from BPAT_subclass_parallel_tempTurnUp_BMforce import BPAT_PARALLEL_TEMPTURNUP_BM_FORCE
from Control_BPAT_NeckerCubeStatic import Control_BPAT_NeckerCubeStatic

#from TESTING_procedure_abstract import TEST_PROCEDURE


"""

    This class performs the detailed steps of the experiment.
    It holds the following 'milestone' methods:
        - load_data: 
            > load data from given source 
            > modify data for inference as determined by 'modification'
                (set BP-booleans for inference, rotate data, translate data)
            > determine optimal inference results for rotation and translation
        - prepare_inference: 
            > initialize important variables in the BPAT 
                (e.g. load the model, create the storage tensors, etc.)
            > set the optimal parameters for binding, rotation, and translation
                which are used as target in the inference
            > construct the parameter_information.txt from all given experiment
                parameters and save it in the experiment folder
        - evaluate: 
            > get experiment data (losses, gradients, etc.)
            > plot results
            > store results, some in csv-files, some in pt files 
                (Note: csv files will later be used to create comparing plots 
                between experiment trials)
        - run: 
            > create folder for experiment trial
            > prepare, run and evaluate experiment trial via calling respective functions
            > organize and return results of experiment trial


"""

class TESTER(TEST_PROCEDURE): 

    ###################################################################################################
    ### Initializations
    ###################################################################################################

    def __init__(self, num_features, num_observations, num_dimensions, experiment_name):
        super().__init__(num_features, num_observations, num_dimensions)
        experiment_path = "Testing/Grafics/"+experiment_name+'/'
        super().create_trial_directory(experiment_path)
        self.trial_path = self.result_path

        self.BPAT = None
        self.infer_coord = False

        self.do_binding = False
        self.do_rotation = False
        self.rotation_type = None
        self.do_translation = False
        
        print('Initialized test environment.')


    """
        Initialize BPAT as an instance of the class determined by 
        given structure. 
    """
    def set_BPAT_structure(self, structure_type):
        self.structure = structure_type
        if structure_type == 'sequential':
            self.BPAT = BPAT_SEQUENTIAL()
        elif structure_type == 'parallel':
            self.BPAT = BPAT_PARALLEL()
        elif structure_type == 'parallel_temp_turnup':
            self.BPAT = BPAT_PARALLEL_TEMPTURNUP()
        elif structure_type == 'parallel_temp_turnup_BMforce':
            self.BPAT = BPAT_PARALLEL_TEMPTURNUP_BM_FORCE()
        elif structure_type == 'necker_cube_static_bind':
            self.BPAT = Control_BPAT_NeckerCubeStatic()
        else:
            print('error: No valid structure type! Please check values.')
            exit()


    ###################################################################################################
    ### Methods for preparing the experiment
    ###################################################################################################

    def load_data(self, modifications=None, sample_nums=None):
        self.init_modification_params()

        if self.gestalten: 
            dim_spare = self.num_dimensions
            self.num_dimensions = 3

        data_modification = []
        for action, modify, specify in modifications: 
            if action == 'bind':
                self.do_binding = True

            elif action == 'rotate': 
                self.do_rotation = True

                if modify == 'rand' or modify == 'det' or modify == 'set':
                    if modify == 'rand': 
                        modification_quat = torch.rand(1,4).view(1,4)
                        modification = self.PERSP_TAKER.norm_quaternion(modification_quat)
                        if specify == 'eulrotate': 
                            modification = torch.rand(3).view(3,1)*360

                        print(f'Randomly modified rotation of observed features: {specify} by {modification}')

                    elif modify == 'det': 
                        modification_quat = torch.Tensor([0.7333946 ,0.1242057, 0.6644438, -0.0722476]).view(1,4)    # eul: 45, 83, 29 | ^85.6565338
                        modification = self.PERSP_TAKER.norm_quaternion(modification_quat)
                        if specify == 'eulrotate': 
                            modification = torch.rad2deg(self.PERSP_TAKER.qeuler(modification,'xyz').view(3,1))
                        
                        print(f'Deterministically modified rotation of observed features: {specify} by {modification}')
                    
                    else: 
                        self.set_modification = True
                        (specify, modification) = specify
                        modification_quat = modification.view(1,4)
                        modification = self.PERSP_TAKER.norm_quaternion(modification_quat)
                        if specify == 'eulrotate': 
                            modification = torch.rad2deg(self.PERSP_TAKER.qeuler(modification,'xyz').view(3,1))
                        
                        print(f'Set rotation of observed features: {specify} by {modification}')

                    self.new_rotation = modification
                    rerot_quat = self.PERSP_TAKER.inverse_rotation_quaternion(modification_quat)
                    if specify == 'qrotate':
                        self.rerotate = rerot_quat
                        self.rerotation_matrix = self.PERSP_TAKER.quaternion2rotmat(self.rerotate)

                    elif specify == 'eulrotate': 
                        self.rerotate = torch.rad2deg(self.PERSP_TAKER.qeuler(rerot_quat,'xyz').view(3,1))
                        self.rerotation_matrix = self.PERSP_TAKER.compute_rotation_matrix_(
                            self.rerotate[0], self.rerotate[1], self.rerotate[2]).view(3,3)

                    data_modification += [(specify, modification)]
                else: 
                    modification = None
                    self.new_rotation = None
                    if specify == 'qrotate':
                        self.rerotate = torch.zeros(1,4)
                        self.rerotate[0,0] = 1.0
                    elif specify == 'eulrotate':
                        self.rerotate = torch.zeros(3).view(3,1)
                    self.rerotation_matrix = torch.Tensor(np.identity(self.num_dimensions))
                
                self.rotation_type = specify

            elif action == 'translate': 
                self.do_translation = True

                if modify == 'rand' or modify == 'det' or modify == 'set':
                    if modify == 'rand': 
                        lower_bound = specify[0] * torch.ones(self.num_dimensions)
                        upper_bound = specify[1] * torch.ones(self.num_dimensions)
                        range_size = upper_bound - lower_bound
                        modification = torch.mul(torch.rand(self.num_dimensions), range_size)
                        print(f'Randomly modified translation of observed features: {modification}')

                    elif modify == 'det': 
                        modification = torch.Tensor([0.2, -0.8, 0.4])
                        print(f'Deterministically modified translation of observed features: {modification}')

                    else:
                        self.set_modification = True
                        modification = specify
                        print(f'Set translation of observed features: {modification}')

                    self.new_translation = modification
                    self.retranslate = self.PERSP_TAKER.inverse_translation_bias(modification)

                    data_modification += [('translate', modification)]
                else: 
                    modification = None
                    self.new_translation = None
                    self.retranslate = torch.zeros(self.num_dimensions)
                    print(self.num_dimensions)
            
            else: 
                print('Received unknown modification. Skipped.')

        if self.gestalten: 
            self.num_dimensions = dim_spare
        
        if data_modification == []:
            data_modification = None
        
        data_paths = self.get_data_paths()
        data = self.load_data_all(data_paths, sample_nums, data_modification)
        
        return data


    ###################################################################################################

    def prepare_inference(self, 
        rotation_type, 
        current_rotation,
        current_translation, 
        num_frames, 
        model_path, 
        layer_norm,
        tuning_length, 
        num_tuning_cycles, 
        at_loss_function, 
        at_loss_parameters, 
        at_learning_rates,
        at_learning_rate_state, 
        at_momenta, 
        at_signdamps):


        super().prepare_inference(
            self.BPAT, 
            num_frames, 
            model_path,
            layer_norm, 
            tuning_length, 
            num_tuning_cycles, 
            at_loss_function, 
            at_learning_rates, 
            at_learning_rate_state, 
            at_momenta,
            at_signdamps)

        # set ideal comparision parameters
        ideal_binding, ideal_rotation, ideal_translation = None, None, None
        info_string = ''

        info_string += f'Performed inference for {num_frames} frames with the following parameters:\n\n'
        info_string = self.construct_info_string(info_string, at_loss_parameters)

        # binding
        if self.do_binding:
            ideal_binding = torch.Tensor(np.identity(self.num_features))
            if self.BPAT.nxm:
                oc_line = torch.zeros(self.num_observations)
                for i in self.BPAT.get_additional_features():
                    ideal_binding = torch.cat([
                        ideal_binding[:,:i], 
                        torch.zeros(self.num_features, 1), 
                        ideal_binding[:,i:]], dim=1)
                    oc_line[i] = 1
                oc_line = oc_line.view(1, self.num_observations)
                ideal_binding = torch.cat([ideal_binding, oc_line], dim=0)
        
        # rotation
        if self.do_rotation:
            rerot, rerot_matrix = None, None
            if current_rotation is not None: 
                rerot = self.rerotate
                rerot_matrix = self.rerotation_matrix
            else: 
                if rotation_type == 'qrotate':
                    rerot = torch.zeros(1,4)
                    rerot[0,0] = 1.0
                elif rotation_type == 'eulrotate':
                    rerot = torch.zeros(3).view(3,1)
                rerot_matrix = torch.Tensor(np.identity(3))

            ideal_rotation = (rerot, rerot_matrix)

            info_string += f' - modification of body rotation with {rotation_type} by \n\t{self.new_rotation if current_rotation is not None else None}\n'
            info_string += f' - optimally infered rotation: \n\t{rerot}\n\n'

        # translation
        if self.do_translation:
            if current_translation is not None: 
                retrans = self.retranslate
            else: 
                retrans = torch.zeros(3)

            ideal_translation = retrans       

            info_string += f' - modification of body translation: {self.new_translation if current_translation is not None else None}\n'
            info_string += f' - optimally infered translation: \n\t{retrans}\n\n'
        
        self.BPAT.set_comparison_values(ideal_binding, ideal_rotation, ideal_translation)
        
        # parameter information
        info_string += f'\n#### Binding ####\n'
        info_string += f' - scaler:  \t{self.BPAT.scale_mode}\n'
        info_string += f' - prescaling:\t{self.BPAT.binding_prescale}\n'
        info_string += f' - sigma:\t{self.BPAT.binding_sigma}\n'
        info_string += f'# Temperature #\n'
        if self.BPAT.temp_turnup:
            info_string += f' - turn-up max column:  \t{self.BPAT.binder.max_temp_val_col}\n'
            info_string += f' - turn-up max rows:  \t\t{self.BPAT.binder.max_temp_val_row}\n'
            info_string += f' - turn-up range column:  \t{self.BPAT.range_temp_turnup_col}\n'
            info_string += f' - turn-up range rows:  \t{self.BPAT.range_temp_turnup_row}\n'
            if self.BPAT.binder.temp_change=="step_up":
                info_string += f' - turn-up steps column:  \t{self.BPAT.binder.temp_step_col}\n'
                info_string += f' - turn-up steps rows:  \t{self.BPAT.binder.temp_step_row}\n'
                info_string += f' - turn-up gradient range column:  \t{self.BPAT.rel_temp_grad_range_col}\n'
                info_string += f' - turn-up gradient range rows:  \t{self.BPAT.rel_temp_grad_range_row}\n'

                info_string += f' - turn-up function:  \t\t{self.BPAT.binder.fct_temperature_turnup}\n'
                info_string += f' - turn-up function:  \t\t{self.BPAT.binder.fct_realtive_temperature_turnup}\n'
            info_string += f' - turn-up reset:  \t\t{self.BPAT.temp_reset_frame}\n'
        else: 
            info_string += f' - columns:  \t{self.BPAT.binder.temp_val_col}\n'
            info_string += f' - rows:  \t\t{self.BPAT.binder.temp_val_row}\n'
            
            
        info_string += f'# NxM Binding #\n'
        info_string += f' - NxM: \t\t\t{self.BPAT.nxm}\n'
        info_string += f' - distractors:  \t\t{self.BPAT.distractors}\n'
        info_string += f' - additional features:  \t{self.BPAT.additional_features}\n'
        info_string += f' - outcast initial value:  \t{self.BPAT.outcast_init_value}\n'
        info_string += f' - outcast enhancement:  \t{self.BPAT.nxm_enhance_ocvals}\n'
        info_string += f' - outcast scale:  \t\t{self.BPAT.nxm_last_line_scale}\n'

        info_string += f'\n#### Rotation ####\n'
        info_string += f' - rotation type:  \t{self.BPAT.rotation_type}\n'

        info_string += f'\n#### General parameters ####\n'
        info_string += f' - learning rates:\n' 
        info_string += f'\t> binding: \t\t{self.BPAT.at_learning_rate_binding}\n'
        info_string += f'\t> rotation: \t\t{self.BPAT.at_learning_rate_rotation}\n'
        info_string += f'\t> translation: \t\t{self.BPAT.at_learning_rate_translation}\n'
        info_string += f'\t> state:\t\t{self.BPAT.at_learning_rate_state}\n'
        info_string += f' - momenta:\n' 
        info_string += f'\t> binding: \t\t{self.BPAT.bm_momentum}\n'
        info_string += f'\t> rotation: \t\t{self.BPAT.r_momentum}\n'
        info_string += f'\t> translation: \t\t{self.BPAT.c_momentum}\n'
        info_string += f' - sign damping:\n' 
        info_string += f'\t> binding: \t\t{self.BPAT.b_signdamp}\n'
        info_string += f'\t> rotation: \t\t{self.BPAT.r_signdamp}\n'
        info_string += f'\t> translation: \t\t{self.BPAT.c_signdamp}\n'
        info_string += f' - gradient calculation:\n' 
        info_string += f'\t> binding: \t\t{self.BPAT.grad_calc_binding}\n'
        info_string += f'\t> rotation: \t\t{self.BPAT.grad_calc_rotation}\n'
        info_string += f'\t> translation: \t\t{self.BPAT.grad_calc_translation}\n'
        info_string += f' - gradient weightings (not important for meanOfTunHor):\n' 
        info_string += f'\t> binding: \t\t{self.BPAT.grad_bias_binding}\n'
        info_string += f'\t> rotation: \t\t{self.BPAT.grad_bias_rotation}\n'
        info_string += f'\t> translation: \t\t{self.BPAT.grad_bias_translation}\n'

        self.write_to_file(info_string, self.result_path+'parameter_information.txt')
        print('Ready to run AT inference for binding task! \nInitialized parameters with: \n' + info_string)



    ###################################################################################################
    ### Evaluation of experiment
    ###################################################################################################

    def evaluate(self, 
        at_optimal_inputs, 
        # observations,
        final_predictions, 
        at_bin_gradients, 
        at_rot_gradients,
        at_trans_gradients,
        final_binding_matrix,
        final_binding_entries, 
        final_rotation_values, 
        final_rotation_matrix, 
        final_translation_values, 
        feature_names):

        results, figures, fig_names = super().evaluate(self.BPAT, at_optimal_inputs, final_predictions)
        res_i = len(fig_names)

        res_names = []
        csv_names = []
        csv_names += fig_names
        pt_results = []
        
        grad_figures = []
        grad_figure_names = []

        ## Save figures
        if self.do_binding: 
            grad_figures += [self.BPAT.evaluator.plot_at_gradients(at_bin_gradients, 'Binding gradients')]
            grad_figure_names += ['bin_grads']

            fig_names += ['determinante_history']
            csv_names += ['determinante_history']
            figures += [self.BPAT.evaluator.plot_at_losses(results[res_i], 'History of binding matrix determinante')]
            res_i += 1
        
            if self.num_features != self.num_observations: 
                figures += [self.BPAT.evaluator.plot_at_losses(
                    results[res_i][:,0], 
                    'History of binding matrix loss (FBE) for cleared matrix'
                )]
                figures += [self.BPAT.evaluator.plot_at_losses(
                    results[res_i][:,1], 
                    'History of binding matrix loss (FBE) for outcast line and additional cloumns'
                )]
                figures += [self.BPAT.evaluator.plot_at_losses(
                    results[res_i][:,2], 
                    'History of binding matrix loss (FBE) for whole matrix'
                )]
                res_fbe_nxm = res_i
                res_i += 1

                figures += [self.BPAT.evaluator.plot_binding_matrix_nxm(
                    final_binding_matrix, 
                    feature_names, 
                    self.num_observations,
                    self.BPAT.get_additional_features(),
                    'Binding matrix showing relative contribution of observed feature to input feature'
                )]
                figures += [self.BPAT.evaluator.plot_binding_matrix_nxm(
                    final_binding_entries, 
                    feature_names, 
                    self.num_observations,
                    self.BPAT.get_additional_features(),
                    'Binding matrix entries showing contribution of observed feature to input feature'
                )]
                figures += [self.BPAT.evaluator.plot_outcast_gradients(
                    self.BPAT.get_oc_grads(), 
                    feature_names, 
                    self.num_observations,
                    self.BPAT.get_additional_features(),
                    'Gradients of outcast line for observed features during inference'
                )]
                fig_names += [
                    'fbe_cleared_history', 'fbe_oc+af_history', 'fbe_whole_history', 
                    'final_binding_matirx', 'final_binding_neurons_activities','outcat_line_gradients']
                # csv_names += ['fbe_cleared_history', 'fbe_oc+af_history', 'fbe_whole_history', 'outcat_line_gradients']
                csv_names += ['fbe_whole_history']

            else:
                figures += [self.BPAT.evaluator.plot_at_losses(results[res_i], 'History of binding matrix loss (FBE)')]
                res_i += 1

                figures += [self.BPAT.evaluator.plot_binding_matrix(
                    final_binding_matrix, 
                    feature_names, 
                    'Binding matrix showing relative contribution of observed feature to input feature'
                )]
                figures += [self.BPAT.evaluator.plot_binding_matrix(
                    final_binding_entries, 
                    feature_names, 
                    'Binding matrix entries showing contribution of observed feature to input feature'
                )]
                fig_names += ['fbe_history', 'final_binding_matirx', 'final_binding_neurons_activities']
                csv_names += ['fbe_history']
            
            res_names += ['final_binding_matirx', 'final_binding_neurons_activities']
            pt_results += [final_binding_matrix, final_binding_entries]
        else:
            results = results[:-5] + results[-3:]

        if self.do_rotation:
            grad_figures += [self.BPAT.evaluator.plot_at_gradients(at_rot_gradients, 'Rotation gradients')]
            grad_figure_names += ['rot_grads']

            figures += [self.BPAT.evaluator.plot_at_losses(results[res_i], 'History of rotation matrix loss (MSE)')]
            res_i += 1
            figures += [self.BPAT.evaluator.plot_at_losses(results[res_i], 'History of rotation values loss')]
            res_i += 1

            fig_names += ['rotmat_loss_history', 'rotval_loss_history']
            csv_names += ['rotmat_loss_history', 'rotval_loss_history']
            res_names += ['final_rotation_values', 'final_rotation_matrix']
            pt_results += [final_rotation_values, final_rotation_matrix]
        else:
            results = results[:-3] + results[-1:]
        
        if self.do_translation:
            grad_figures += [self.BPAT.evaluator.plot_at_gradients(at_trans_gradients, 'Translation gradients')]
            grad_figure_names += ['trans_grads']

            figures += [self.BPAT.evaluator.plot_at_losses(results[res_i], 'History of translation loss (MSE)')]
            res_i += 1

            fig_names += ['transba_loss_history']
            csv_names += ['transba_loss_history']
            res_names += ['final_translation_values']
            pt_results += [final_translation_values]
        else: 
            results = results[:-1]


        figures_2 = figures[1:]
        fig_names_2 = fig_names[1:]
        self.save_figures(figures_2, fig_names_2)   #(figures, fig_names)
        self.save_figures(grad_figures, grad_figure_names)
        
        if self.num_features != self.num_observations:
            nxm_c_names = ['fbe_cleared_history', 'fbe_oc+af_history', 'outcat_line_gradients']
            res = []
            res += [results[res_fbe_nxm][:, 0]]
            res += [results[res_fbe_nxm][:, 1]]
            res += [self.BPAT.get_oc_grads()]
            self.save_results_to_csv(res, nxm_c_names)
            
            results[res_fbe_nxm] = results[res_fbe_nxm][:, 2]

        self.save_results_to_csv(results, csv_names)
        self.save_results_to_pt(pt_results, res_names)

        return results, csv_names


    ###################################################################################################
    ### Run experiment trial with the given parameters
    ###################################################################################################

    def run(self,
        experiment_dir,
        modification,
        sample_nums, 
        model_path, 
        layer_norm,
        tuning_length, 
        num_tuning_cycles, 
        at_loss_function,
        loss_parameters,
        at_learning_rates, 
        at_learning_rate_state, 
        at_momenta,
        at_signdamps):

        print('*************************************************************************************')

        experiment_results = []
        self.BPAT.set_dimensions(self.num_dimensions)
        print(f'Use model: {model_path}')

        with torch.no_grad():
            data = self.load_data(modification, sample_nums)

        res_path = ""
        if self.do_binding:
            res_path += 'b_'
        if self.do_rotation:
            res_path += 'r_'
        if self.do_translation:
            res_path += 't_'
        self.prefix_res_path = self.trial_path + res_path
        res_path = self.prefix_res_path + experiment_dir
        if experiment_dir != "":
            os.mkdir(res_path)
            print('Created directory: '+ res_path)

        sample_names = []
        for (name, observations, feat_names) in data:
            sample_names += [name]
            obs_shape = observations.shape
            num_frames = observations.size()[0]
            self.BPAT.set_data_parameters_(
                num_frames, self.num_observations, self.num_features, self.num_dimensions)


            self.result_path = res_path+name+'/'
            os.mkdir(self.result_path)

            new_rotation = self.new_rotation
            new_translation = self.new_translation
            if '_' not in name:
                if new_rotation is not None:
                    new_rotation = None

                if new_translation is not None:
                    new_translation = None
            
            self.prepare_inference(
                self.rotation_type, 
                new_rotation,
                new_translation, 
                num_frames, 
                model_path, 
                layer_norm,
                tuning_length, 
                num_tuning_cycles, 
                at_loss_function, 
                loss_parameters, 
                at_learning_rates,
                at_learning_rate_state, 
                at_momenta,
                at_signdamps)

            # if self.gestalten:
            #     self.render_gestalt(observations)
            #     self.render_gestalt(data[1][1])
            # else:
            #     self.render(observations)
            #     self.render(data[1][1])

            # exit()

            [at_final_inputs,
                at_optimal_inputs,
                at_final_predictions, 
                final_binding_matrix,
                final_binding_entries, 
                final_rotation_values, 
                final_rotation_matrix, 
                final_translation_values,
                at_bin_gradients, 
                at_rot_gradients,
                at_trans_gradients,
                ] = self.BPAT.run_inference(
                    observations, 
                    self.do_binding, 
                    self.do_rotation,
                    self.do_translation,
                    self.result_path) if not self.infer_coord else self.BPAT.run_inference(
                    observations, 
                    self.do_coordinate,
                    self.do_binding, 
                    self.do_rotation,
                    self.do_translation,
                    self.result_path)


            # if self.gestalten:
            #     # if self.illusion == None:
            #     #     self.render_gestalt(at_final_inputs.view(num_frames, self.num_features, self.num_dimensions))
            #     #     self.render_gestalt(at_final_predictions.view(num_frames, self.num_features, self.num_dimensions))
            #     # else:
            #     self.render_gestalt(at_final_inputs.view(num_frames-tuning_length, self.num_features, self.num_dimensions))
            #     self.render_gestalt(at_final_predictions.view(num_frames-tuning_length, self.num_features, self.num_dimensions))
            # else:
            #     # if self.illusion == None:
            #     #     self.render(at_final_inputs.view(num_frames, self.num_features, self.num_dimensions))
            #     #     self.render(at_final_predictions.view(num_frames, self.num_features, self.num_dimensions))
            #     # else:
            #     self.render(at_final_inputs.view(num_frames-tuning_length, self.num_features, self.num_dimensions))
            #     self.render(at_final_predictions.view(num_frames-tuning_length, self.num_features, self.num_dimensions))

            self.save_results_to_pt([
                at_optimal_inputs,
                at_final_inputs,
                at_bin_gradients, 
                at_rot_gradients,
                at_trans_gradients], ['optimal_inputs', 'final_inputs', 'bin_grads', 'rot_grads', 'trans_grads'])

            # rerotate observations to compare with final predictions 
            if new_rotation is not None:
                data_shape = observations.shape
                if self.gestalten:
                    if self.dir_mag_gest:
                        mag = observations[:,:, -1].view(num_frames, self.num_observations, 1)
                        observations = observations[:,:, :-1]
                    observations = torch.cat([
                        observations[:,:, :self.num_dimensions], 
                        observations[:,:, self.num_dimensions:]], dim=2)
                    observations = observations.view((num_frames)*self.num_observations*2, 3)
                else: 
                    observations = observations.view(num_frames*self.num_observations, self.num_dimensions)
                    
                if self.rotation_type == 'qrotate':
                    observations = self.PERSP_TAKER.qrotate(observations, self.rerotate)   
                else:
                    rotmat = self.PERSP_TAKER.compute_rotation_matrix_(self.rerotate[0], self.rerotate[1], self.rerotate[2])
                    observations = self.PERSP_TAKER.rotate(observations, rotmat)   

                if self.gestalten:
                    observations = observations.reshape(num_frames, self.num_observations, 6)
                    if self.dir_mag_gest:
                        observations = torch.cat([observations, mag], dim=2)
                else:
                    observations = observations.view(data_shape)
                

            # retranslate observations to compare with final predictions 
            if new_translation is not None:
                non_pos = observations[:,:,3:]
                observations = observations[:,:,:3]
                observations = observations.view((num_frames)*self.num_observations, 3)
                self.PERSP_TAKER.translate(observations, self.retranslate)
                observations = observations.view((num_frames), self.num_observations, 3)
                observations = torch.cat([observations, non_pos], dim=2)


            res, res_names = self.evaluate(
                at_optimal_inputs, 
                # observations, 
                at_final_predictions, 
                at_bin_gradients, 
                at_rot_gradients,
                at_trans_gradients,
                final_binding_matrix,
                final_binding_entries, 
                final_rotation_values, 
                final_rotation_matrix, 
                final_translation_values, 
                feat_names)
            print('Evaluated current run.')


            super().terminate()
            print('Terminated current run.')


            experiment_results += [[name, res, final_binding_matrix, final_binding_entries]]

        return sample_names, res_names, experiment_results
        
