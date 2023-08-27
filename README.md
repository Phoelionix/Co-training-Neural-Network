# Co-trained neural network (MLP) - custom Numpy implementation 

(2022 project) 

Note that no ML packages are used, but the code is vectorised via numpy.

--- 

Underlying MultiLayer Perceptron (MLP) model structure is constructed via a vectorised NumPy implementation. 
File designed for jupyter environment.
The beginning of the python file contains the parameters needed to be modified to achieve different model types. See "Model variables" below. Hyperparameters are also included. 
Code is very messy. Users should execute the first, then second cell to train the model. The third cell stores data at the file path held in the variable by prediction_path.


encoding variable:
encoded (set True if embedded data is being used)

file path variables:
prediction_path
train_data_path
unlabelled_data_path
dev_data_path (for plotting accuracies)
test_data_path
path_for_temp_files (NOTE should also be the path to the dataset files. Stores copies of the training data. Needs to end in "/")

### Model variables:

#### Supervised (baseline) model: 
semi_supervised = False

#### Self-trained, semi-supervised model:
semi_supervised = True
co_training = False

#### Co-trained, semi-supervised model:
semi_supervised = True
co_training = True


(c) Spencer Passmore 2022
