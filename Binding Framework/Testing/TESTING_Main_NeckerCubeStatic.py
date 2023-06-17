"""
Author: Kaltenberger
franziska.kaltenberger@student.uni-tuebingen.de
"""
import os

import sys

sys.path.append('C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Python Projects/Code/Binding Framework')
os.chdir('C:/Users/49157/OneDrive/Dokumente/UNI/8. Semester/Bachelorarbeit/Python Projects/Code/Binding Framework')
#sys.path.append('C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/TimGerneProjectCodeOnly')
#os.chdir('C:/Users/Tim/_Eigene_Dateien/Studium_Kogni/Kogni_Semester_7/Bachelor_Arbeit/TimGerneProjectCodeOnly')

# Before run: replace ... with current directory path
#sys.path.append('/home/kaltenf/Documents/BPAT/Code/repro_continue01/BPAT_continue_01')

from testing_module.experiment_interface_opt_illusions import EXPERIMENT_INTERFACE_OPT_ILLUSIONS


class TESTING_NeckerCubeStatic_TimGerneBSc(EXPERIMENT_INTERFACE_OPT_ILLUSIONS):

    def __init__(self, num_features, num_observations, num_dimensions, illusion, edge_points):
        experiment_name = f"binding_test_results_{illusion}"
        super().__init__(num_features, num_observations, num_dimensions, experiment_name, illusion, edge_points)


    def perform_experiment(self, sample_nums, modification, structure, dimension_values, rotation_type): 
        
        changed_parameter = 'dimensions'
        tested_values = dimension_values
        distractors = None
        temperature = 'smooth_turn_down'        # either: fixed, smooth_turn_down, or turn_up

        super().perform_experiment(
            sample_nums, 
            modification, 
            structure,
            changed_parameter, 
            tested_values,
            rotation_type, 
            distractors, 
            temperature)

def main(): 
    # set the following parameters
    num_observations = 8    # The number of feature observations that are provided
    num_input_features = 8  # The number of input features expected / processed by the model (should be <= num_observations)
    num_dimensions = 3      # The total number of values describing a feature (e.g. 3 for x,y,z)
    
    rotation_type = 'qrotate'   # either: 'eulrotate' or 'qrotate'

    illusion = 'necker_cube_static'    # either: 'necker_cube' or 'dancer'
    edge_points = 8

### Tuning structure        #############################################################

    structure = 'necker_cube_static_bind'

    testNeckerCubeStaticTimeGerneBSc = TESTING_NeckerCubeStatic_TimGerneBSc(
        num_input_features, 
        num_observations, 
        num_dimensions, 
        illusion,
        edge_points) 

    
    modification = [
        ('bind', None, None)
    ]


    # sample_nums = [600, 600, 600, 600] 
    sample_nums = [200, 200, 200]   #[12, 12, 12]

    tested_dimensions = [3] #[6]

    testNeckerCubeStaticTimeGerneBSc.perform_experiment(
        sample_nums, 
        modification, 
        structure,
        tested_dimensions, 
        rotation_type)



if __name__ == "__main__":
    main()
