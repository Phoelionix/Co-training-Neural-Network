# Co-trained neural network (MLP) - custom Numpy implementation 

(2022 project) 

Note that no ML packages are used, but the code is vectorised via numpy.

--- 

Underlying MultiLayer Perceptron (MLP) model structure is constructed via a vectorised NumPy implementation.  <br>
File designed for jupyter environment.  <br>
The beginning of the python file contains the parameters needed to be modified to achieve different model types. See "Model variables" below. Hyperparameters are also included.  <br> 
Code is very messy. Users should execute the first, then second cell to train the model. The third cell stores data at the file path held in the variable by prediction_path.  <br>


### encoding variable:
encoded (set True if embedded data is being used)

### file path variables:
prediction_path  <br>
train_data_path  <br>
unlabelled_data_path  <br>
dev_data_path (for plotting accuracies)  <br>
test_data_path  <br>
path_for_temp_files (NOTE should also be the path to the dataset files. Stores copies of the training data. Needs to end in "/")

### Model variables:

#### Supervised (baseline) model:
semi_supervised = False

#### Self-trained, semi-supervised model:
semi_supervised = True  <br>
co_training = False

#### Co-trained, semi-supervised model:
semi_supervised = True  <br>
co_training = True


(c) Spencer Passmore 2022
