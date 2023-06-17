"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""

import pandas as pd   
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('...')      
# Before run: replace ... with current directory path

class TEST_STATISTICS(): 

    """
        Plots comparisons and histories for evaluation values. 
    """

    def __init__(self, num_features, num_observations, num_dimensions): 
        self.num_features = num_features
        self.num_observations = num_observations
        self.num_dimensions = num_dimensions

        self.img_height = 4
        self.img_length = 6

        self.title_fontsize =14
        self.label_fontsize = 12

        self.legend_fontsize = 13

        print('Initialized statistical evaluator.')


    def set_parameters(self, num_feat, num_obs, num_dim):
        self.num_features = num_feat
        self.num_observations = num_obs
        self.num_dimensions = num_dim


    def plot_histories(self, history_dfs, path, variation, results, titles): 
        for i in range(len(history_dfs)):
            dt = history_dfs[i]
            fig, ax = plt.subplots(figsize = (self.img_length,self.img_height))
            plt.subplots_adjust(bottom=0.15)
            # plt.subplots_adjust(right=0.65)
            ax = sns.lineplot(
                x="inference step", 
                y=results[i], 
                hue=variation, 
                data = dt 
            )
            ax.set_title(
                titles[i], 
                fontdict= { 'fontsize': self.title_fontsize, 'fontweight':'bold'})

            # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            plt.setp(ax.get_legend().get_texts(), fontsize=8) 
            plt.setp(ax.get_legend().get_title(), fontsize=self.label_fontsize, fontweight='bold')
            ax.set_ylabel(results[i], fontsize=self.label_fontsize)
            ax.set_xlabel('inference step', fontsize=self.label_fontsize)

            fig.savefig(path+results[i]+'.png')  
            fig.savefig(path+results[i]+'.pdf')  
        
        plt.close('all')


    def plot_value_comparisons(self, history_dfs, path, variation, results, titles):
        for i in range(len(history_dfs)):
            dt = history_dfs[i]
            fig, ax = plt.subplots(figsize = (self.img_length,self.img_height))
            plt.subplots_adjust(bottom=0.15)
            # plt.subplots_adjust(right=0.75)
            ax = sns.boxplot(
                x=variation,
                y=results[i],
                whis=1.5,
                data=dt
            )
            ax.set_title(
                titles[i], 
                fontdict= { 'fontsize': self.title_fontsize, 'fontweight':'bold'})
            
            ax.set_xlabel(variation, fontsize=self.label_fontsize)
            ax.set_ylabel(results[i], fontsize=self.label_fontsize)
            
            fig.savefig(path+results[i]+'comp.png')  
            fig.savefig(path+results[i]+'comp.pdf')  

        plt.close('all')


    def load_csvresults_to_dataframe(self, experiment_path, variation, variation_values, samples, results):
        dfs = []
        for result in results:
            value_dfs = []
            for val in variation_values:
                val = f'{val}'
                val_path = experiment_path+variation+'_'+val+'/'
                sample_dfs = []
                for sample in samples:
                    sdf = pd.read_csv(val_path+sample+'/'+result+'.csv', index_col=False)
                    sdf.columns = ['inference step', result]
                    sample_dfs += [sdf]
                
                vdf = pd.concat(sample_dfs).sort_values('inference step').assign(variation=val)
                vdf.columns = ['inference step', result, variation]
                value_dfs += [vdf]

            df = pd.concat(value_dfs).sort_values('inference step')
            dfs += [df]
        
        return dfs


def main(): 
    # set the following parameters
    num_observations = 15
    num_input_features = 15
    num_dimensions = 3
    stats = TEST_STATISTICS(num_input_features, num_observations, num_dimensions)

    samples = [...]
    result_names = [...]

    param = ...
    param_vals = [...]
    
    titles = [...]

    path = ...

    dfs = stats.load_csvresults_to_dataframe(
        path, 
        param, 
        param_vals, 
        samples, 
        result_names
    )

    print('Created dataframes')


    stats.plot_value_comparisons(
            dfs, 
            path, 
            param, 
            result_names,
            titles 
        )
    print('Plotted comparisons')


    stats.plot_histories(
            dfs, 
            path, 
            param, 
            result_names, 
            titles
        )
    print('Plotted histories.')

  
if __name__ == "__main__":
    main()


