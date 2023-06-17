"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""

from os import TMP_MAX
import torch 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class BPAT_evaluator():

    """
    Evaluator to perform general evaluation of BPAT performance. 

    """

    def __init__(self, 
                 num_frames=None,           
                 num_observations=None,    
                 num_features=15,           
                 preprocessor=None):       

        """ 
        Initialize with given parameters: 

            - number of sample frames
            - number of observed features
            - number of input features for LSTM
            - already initialized preprocessor

        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ## Set parameters
        self.num_frames = num_frames
        # symmetrical binding
        if num_observations is None: 
            self.num_observations = num_features
        # asymmetrical binding
        else:
            self.num_observations = num_observations
        
        self.num_features = num_features
        self.preprocessor = preprocessor


    def set_parameters(self, num_feat, num_obs):
        self.num_features = num_feat
        self.num_observations = num_obs


    def prediction_errors(self, 
                          observations, 
                          final_predictions, 
                          loss_function):

        """
        Symmetrical binding function.
        Compare given predictions to given observations. 
        Return corresponding prediction error for every time step. 

        """
        prediction_error = []
        for i in range(observations.shape[0]-1):
        # for i in range(self.num_frames-1):
            with torch.no_grad():
                obs_t = self.preprocessor.convert_data_AT_to_VAE(observations[i + 1]).to(self.device)
                pred_t = final_predictions[i].to(self.device)
                loss = loss_function(pred_t, obs_t[0])
                prediction_error.append(loss.cpu())

        return prediction_error


    def prediction_errors_nxm(self, 
                          observations, 
                          additional_features,
                          num_observed_features,
                          final_predictions, 
                          loss_function):

        """
        Asymmetrical binding function.
        Compare given predictions to given observations. 
        Return corresponding prediction error for every time step. 

        """
        prediction_error = []
        for i in range(self.num_frames-1):
            with torch.no_grad():
                obs = observations[i+1]
                # obs = [obs[i] for i in range(num_observed_features) if (i not in additional_features)]
                # obs_t = self.preprocessor.convert_data_AT_to_LSTM(torch.stack(obs)).to(self.device)
                obs_t = self.preprocessor.convert_data_AT_to_VAE(obs).to(self.device)
                pred_t = final_predictions[i].to(self.device)
                loss = loss_function(pred_t, obs_t[0])
                prediction_error.append(loss.cpu())

        return prediction_error



    ###########
    ### Evaluation functions for feature binding:
    ###########
    def FBE(self, bm, ideal):
        return torch.sum(torch.sqrt(torch.sum(torch.square(bm-ideal), dim=0)))


    def clear_nxm_binding_matrix(self, bm, additional_features):
        '''
        Remove additional features from binding matrix. 
        Retruned matrix only includes bindings between input features for LSTM. 
        '''
        j = 0
        bm_sq = bm.clone().detach().cpu()
        bm_sq = bm_sq[:-1]
        for i in additional_features:
            i = i-j
            j += 1
            bm_1 = bm_sq[:,:i]
            bm_2 = bm_sq[:,i+1:]
            bm_sq = np.hstack([bm_1, bm_2])
        
        bm_sq = torch.Tensor(bm_sq).to(self.device)

        return bm_sq


    def FBE_nxm_additional_features(self, bm, ideal, additional_features):
        '''
        Computes the feature binding error only for the bindings which involve additional features. 
        '''
        fbe = 0
        oc_fbe = 0
        for j in range(self.num_observations):
            if j in additional_features:
                a = torch.square(bm[self.num_features,j]-ideal[self.num_features,j])
                b = 0
                for i in range(self.num_features):
                    b += torch.square(bm[i,j])
                fbe += torch.sqrt(a+b)
            else: 
                oc_fbe += torch.square(bm[-1,j])
        fbe += torch.sqrt(oc_fbe)
        
        return fbe

    """
    ###########
    Visualization functions: 
    ###########
    """    

    def plot_prediction_errors(self, prediction_error):
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        axes.plot(prediction_error, 'r')
        axes.grid(True)
        axes.set_xlabel('frames')
        axes.set_ylabel('prediction error')
        axes.set_title('Prediction error after active tuning')
        #plt.show()
        return fig



    def plot_at_losses(self, losses, title):
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        axes.plot(losses, 'r')
        axes.grid(True)
        axes.set_xlabel('active tuning runs')
        axes.set_ylabel('loss')
        axes.set_title(title)
        # plt.show()
        return fig

    
    def plot_at_gradients(self, gradients, title): 
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
        axes.plot(gradients)
        axes.grid(True)
        axes.set_xlabel('gradients')
        axes.set_ylabel('gradient value')
        axes.set_title(title)
        # plt.show()
        return fig


    def plot_zpred_loss(self, zpred_loss, title): 
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
        axes.plot(zpred_loss)
        axes.grid(True)
        axes.set_xlabel('time steps')
        axes.set_ylabel('final z-pred loss')
        axes.set_title(title)

        axes.legend(['LefDowFar', 'RigDowFar', 'RigUpFar', 'LefUpFar',
                     'LefUpClo', 'RigUpClo', 'RigDowClo', 'LefDowClo'])
        # plt.show()
        return fig


    def help_visualize_devel(self, observations,final_predictions):
        at_final_pred_plot = final_predictions.reshape(self.num_frames, 15, 3)

        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.scatter3D(observations[0,:,0], 
                     observations[0,:,1], 
                     observations[0,:,2])
        ax.scatter3D(at_final_pred_plot[0,:,0], 
                     at_final_pred_plot[0,:,1], 
                     at_final_pred_plot[0,:,2])
        plt.show()

        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.scatter3D(observations[self.num_frames-1,:,0], 
                     observations[self.num_frames-1,:,1], 
                     observations[self.num_frames-1,:,2])
        ax.scatter3D(at_final_pred_plot[self.num_frames-1,:,0], 
                     at_final_pred_plot[self.num_frames-1,:,1], 
                     at_final_pred_plot[self.num_frames-1,:,2])
        plt.show()

    
    def plot_binding_matrix(self, binding_matrix=None, feature_names=None, title=None):
        observ_num = len(feature_names)
        observ_order = torch.tensor(range(observ_num))
        observ_nums = np.arange(len(feature_names))

        matrix_rc = {
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'text.color': 'black'
        }

        sns.set(rc=matrix_rc)

        palette = "viridis"
        sns.set_palette(palette)

        bm = binding_matrix.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(4.9, 4.9))  #method adjusted for evaluation images
        # plt.subplots_adjust(bottom=0.25)
        # plt.subplots_adjust(left=0.2)
        # plt.subplots_adjust(right=1.01)
##############################################################
        # cax = sns.heatmap(
        #     bm,
        #     ax=ax,
        #     cmap=palette,
        #     vmin=0.0, vmax=1.0,
        #     cbar=True
        #     )
        # cbar = cax.collections[0].colorbar
        # cbar.ax.tick_params(color='black')
        # cbar.ax.tick_params(labelcolor='black')

        cax = sns.heatmap(
            bm,
            ax=ax,
            cmap=palette,
            vmin=0.0, vmax=1.0,
            cbar=False,
            xticklabels=False,
            yticklabels=False
            )
#########################################################

        # cax = ax.matshow(bm)            # draws matrix
        # cb = fig.colorbar(cax, ax=ax, shrink=0.71)   # draws colorbar

        ## Adds numbers to plot
        # for (i, j), z in np.ndenumerate(bm): 
            # ndenumerate function for generating multidimensional index iterator.
            # NOTE i is y-coordinate (row) and j is x-coordinate (column)
            # ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=5)
            # adds a text into the plot where i and j are the coordinates
            # and z is the assigned number 

        ## adding titles
        # ax.set_xticks(observ_nums)
        # ax.set_xticklabels([feature_names[i] for i in observ_order], size=10, color='black')
        # # ax.set_xticklabels(feature_names)
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # ax.set_xlabel('observed feature', size = 15, color='black')
        # ax.xaxis.set_label_position('bottom')
        # ax.set_yticks(observ_nums)
        # ax.set_yticklabels(feature_names, size=10, color='black')
        # plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        # ax.set_ylabel('input feature', size = 15, color='black')


        #plt.title(title, size = 20, fontweight='bold', color='white')
        # plt.show()

        return fig

    
    def plot_binding_matrix_nxm(self, 
        binding_matrix, 
        feature_names, 
        num_observed_features, 
        additional_features, 
        title): 

        bm = binding_matrix.detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(5, 4))
        plt.subplots_adjust(bottom=-0.05)
        plt.subplots_adjust(left=0.2)
        cax = ax.matshow(bm)            # draws matrix
        cb = fig.colorbar(cax, ax=ax, shrink=0.71)   # draws colorbar

        ## Adds numbers to plot
        for (i, j), z in np.ndenumerate(bm): 
            # ndenumerate function for generating multidimensional index iterator.
            ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=5)
            # NOTE i is y-coordinate and j is x-coordinate
            # adds a text into the plot where i and j are the coordinates
            # and z is the assigned number 

        
        ## adding titles
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names, size=10)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
        ax.set_xlabel('observed feature', size = 15, fontweight='bold')
        ax.xaxis.set_label_position('top') 
        feature_names = [feature_names[i] for i in range(num_observed_features) if (i not in additional_features)]
        ax.set_yticks(np.arange(len(feature_names)))
        ax.set_yticklabels(feature_names, size=10)
        plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_ylabel('input feature', size = 15, fontweight='bold')


        plt.title(title, size = 15, fontweight='bold')
        # plt.show()

        return fig


    def plot_outcast_gradients(self, oc_grads, feature_names, num_observed_features, additional_features, title): 
        oc = torch.stack(oc_grads)
        add_feature_grads = [oc[:, i] for i in range(num_observed_features) if (i in additional_features)]
        input_feature_grads = [oc[:, i] for i in range(num_observed_features) if (i not in additional_features)]
        
        fig = plt.figure()
        axes = fig.add_axes([0.1, 0.1, 0.8, 0.8]) 
        for grad in add_feature_grads:
            axes.plot(grad, 'r')

        for grad in input_feature_grads:
            axes.plot(grad, 'b')

        axes.grid(True)
        axes.set_xlabel('active tuning runs')
        axes.set_ylabel('gradients for entries')
        axes.set_title(title)
        # plt.show()

        return fig




def main():

    num_obs = 15
    num_feat = 15
    
    # num_obs = 17
    # num_feat = 16

    evaluator = BPAT_evaluator()    #BAPTAT_evaluator()

    if num_obs==15:
        feature_names = ['LefDowFar', 'RigDowFar', 'RigUpFar', 'LefUpFar', 'LefUpClo', 'RigUpClo', 'RigDowClo', 'LefDowClo']
        order = np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    else:
        feature_names = ['LefDowFar', 'RigDowFar', 'RigUpFar', 'LefUpFar', 'LefUpClo', 'RigUpClo', 'RigDowClo', 'LefDowClo']

        order = np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        # order = np.array([ 0, 1, 4, 3, 2, 5, 10, 7, 8, 9, 6, 11, 12, 13, 14, 15, 16])

    new_order = None
    order = torch.tensor(order, dtype=torch.int64)

    path = ...
    samples = ...

    gestalt_variant = ['Pos', 'PosVel', 'PosDirMag']

    param = ...
    param_vals = ...

    for par in param_vals:

        fbm = torch.zeros(num_feat, num_obs)
        fba = torch.zeros(num_feat, num_obs)
        for gest in gestalt_variant: 
            for s in samples: 
                fbm += torch.load(path+gest+'/b_'+param+'_'+par+'/'+s+'/final_binding_matirx.pt')
                fba += torch.load(path+gest+'/b_'+param+'_'+par+'/'+s+'/final_binding_neurons_activities.pt')

                # fbm += torch.load(path+param+par+'/b_gestalt variant_'+gest+'/'+s+'/final_binding_matirx.pt')
                # fba += torch.load(path+param+par+'/b_gestalt variant_'+gest+'/'+s+'/final_binding_neurons_activities.pt')
                
                # fbm += torch.load(path+'gestalt variant_'+gest+'/'+s+'/final_binding_matirx.pt')
                # fba += torch.load(path+'gestalt variant_'+gest+'/'+s+'/final_binding_neurons_activities.pt')

                # fbm += torch.load(path+par+'/b_r_t_gestalt variant_'+gest+'/'+s+'/final_binding_matirx.pt')
                # fba += torch.load(path+par+'/b_r_t_gestalt variant_'+gest+'/'+s+'/final_binding_neurons_activities.pt')

            # fbm /= len(samples) 
            # fba /= len(samples) 

        fbm /= len(samples) * len(gestalt_variant)
        fba /= len(samples) * len(gestalt_variant)

        fbm = fbm.gather(1, order.unsqueeze(0).expand(fbm.shape))
        fba = fba.gather(1, order.unsqueeze(0).expand(fba.shape))

        if num_obs==15:
            fbm_plt = evaluator.plot_binding_matrix(
                fbm, 
                feature_names,
                'Binding matrix showing relative contribution of observed feature to input feature', 
                new_order
            )
            fba_plt = evaluator.plot_binding_matrix(
                fba, 
                feature_names,
                'Activations of bias neurons in binding matrix', 
                new_order
            )
        else:
            fbm_plt = evaluator.plot_binding_matrix_nxm(
                fbm, 
                feature_names,
                num_obs, 
                [8, 16],
                'Binding matrix showing relative contribution of observed feature to input feature', 
                new_order
            )
            fba_plt = evaluator.plot_binding_matrix_nxm(
                fba, 
                feature_names,
                num_obs, 
                [8, 16],
                'Activations of bias neurons in binding matrix', 
                new_order
            )


        # fbm_plt.savefig(path + gest + '_fbm.png')
        # fbm_plt.savefig(path + gest + '_fbm.pdf')
        fbm_plt.savefig(path + par + '_fbm.png')
        fbm_plt.savefig(path + par + '_fbm.pdf')

        # fba_plt.savefig(path + gest + '_fba.png')
        # fba_plt.savefig(path + gest + '_fba.pdf')
        fba_plt.savefig(path + par + '_fba.png')
        fba_plt.savefig(path + par + '_fba.pdf')

        plt.close('all')
                    


if __name__ == "__main__":
    main()




