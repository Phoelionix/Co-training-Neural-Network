#%% 
#PARAMS FOR YOU#####################################################
semi_supervised = False 
co_training = False

encoded = True
prediction_path = "blank"
train_data_path= "blank"
unlabelled_data_path = "blank"
dev_data_path = "blank"
test_data_path = "blank"
path_for_temp_files = "blank" # Also needs to be path to the dataset files, sorry.

weighted_learning = True  
####################################################################
# Learning Functions
import colorsys
from math import isnan
import numpy as np
import copy
import csv
import pandas as pd
import shutil
import gc # garbage collection
import random
from scipy.special import softmax
from scipy.special import expit
from matplotlib import pyplot as plt

#use_small_data = False  #Whether use dev data, overrides encoded param if True.

one_hot = False   # Whether the output is split into a "toxic" and "not toxic" one-hot encoding
linear_output_activation = False # Not a good idea, swings around between huge numbers D:!  Is fine if start with lower learning rate (exploding gradient problem?), BUT still not improving model accuracy even with square regression TODO figure out what is going on here....???







last_layer_activation_name = "blank"
def preprocess_test(fname):
    starting_id = -2  # will start at -2 if this wasn't set later
    with open(fname, newline='',encoding = file_encoding) as csvfile:

        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        labels_and_features = []
        # Get array of data, removing boilerplate of the ID (now represented by the index in the array) and the name of the features.
        for i,row in enumerate(spamreader):
            if i != 0:
                a = row[0]
                a = a.split(",")
                if i == 1:
                    starting_id = int(a[0])                
                a.pop(0)
                # For training data:
                for j, elem in enumerate(a):
                    if a[j] != "?":
                        a[j] = float(elem)
                    # Test data label
                    else:
                        a[j] = -4
                #a.pop(0)
                labels_and_features.append(a)   
    # Convert to numpy format (not necessary)
    np.asarray(labels_and_features)
    print("LENGTH OF TEST DATA:",len(labels_and_features))
    return(labels_and_features,starting_id) 

def preprocess(fname,recycle_toxic = True,max_number_of_instances = 1000000000):   
    with open(fname, newline='',encoding = file_encoding) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        labels = []
        non_toxic_rows = []
        toxic_rows = []
        shuffled_rows = []
        labels_and_features = []
        debug_num_toxic = 0
        debug_num_not_toxic = 0
        # Get array of data, removing boilerplate of the ID (now represented by the index in the array) and the name of the features.
        last_was_toxic = False
        for i,row in enumerate(spamreader):
            if i != 0 and i < max_number_of_instances:  #We only need to go through this many at most.
                a = row[0]
                a = a.split(",")
                a.pop(0)
                for j, elem in enumerate(a):
                    if a[j] != "?":
                        a[j] = float(elem)
                    # In this case it's the test data label(toxicity = "?")
                    else:
                        a[j] = -4       
                # For training data:                                 
                if a[0] != ["?"]:
                    if a[0] == 0:
                        non_toxic_rows.append(a)
                        debug_num_not_toxic += 1
                    if a[0] == 1:
                        toxic_rows.append(a)
                        debug_num_toxic += 1
    if recycle_toxic:
        j = 0 # the index within the toxic row array. i is the index of the non_toxic_row array.
        for i,a in enumerate(non_toxic_rows):
            tox_row = copy.deepcopy(toxic_rows[j])
            clean_row = copy.deepcopy(non_toxic_rows[i])
            if len(shuffled_rows) < max_number_of_instances:
                shuffled_rows.append(tox_row)
                shuffled_rows.append(clean_row)
            j += 1
            if j >= len(toxic_rows):
                j = 0
    else: #not implemented
        for i,a in enumerate(non_toxic_rows):
            clean_row = copy.deepcopy(non_toxic_rows[i])
            shuffled_rows.append(clean_row)
        for i,a in enumerate(toxic_rows):
            tox_row = copy.deepcopy(toxic_rows[i])
            shuffled_rows.append(tox_row)

    random.shuffle(shuffled_rows)

    for a in shuffled_rows:
        label = a[0]
        #TODO output these labels into a file to check it's 50/50
        #a.pop(0)
        labels.append(label)
        labels_and_features.append(a)
    if not recycle_toxic:
        # Ensure even number so we have same number of toxic and not toxic.
        if len(labels_and_features)%2 != 0:
            if labels[-1] == 0:
                debug_num_not_toxic -=1
            else:
                debug_num_toxic -=1
            labels_and_features.pop()
            labels.pop()                
    # Convert to numpy format (not necessary)
    np.asarray(labels_and_features)
    np.asarray(labels)
    print("LENGTH OF PROCESSED DATA:",len(labels_and_features))
    return(labels_and_features)     

def preprocess_specific_instance(fname,instance_index):
    with open(fname, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        # Get array of data, removing boilerplate of the ID (now represented by the index in the array) and the name of the features.
        last_was_toxic = False

        index = instance_index + 1
        # Get specific row
        for i,row in enumerate(spamreader):
            if index == i:
                continue
        a = row[0]
        a = a.split(",")
        a.pop(0)
        for j, elem in enumerate(a):
            if a[j] != "?":
                a[j] = float(elem)
            # Test data label
            else:
                a[j] = -4

        label = a[0]
        a.pop(0)
        feature = a
    return(feature,label)     

# NOT FAST AT ALL :(
def fast_process_specific_instance(fname,instance_index):
    print("WARNING THIS PREPROCESSING ISNT FAST AND IS BAD")
    with open(fname, newline='') as csvfile:
        #spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        
        # 
        index = instance_index
        # Get row
        df = pd.read_csv(fname,skiprows=index, nrows = 1)
        row = df.iloc[0].to_numpy()
        row = np.ndarray.tolist(row)
        a = row
        ''' not working, all strings :/
        if not isinstance(a[0],float):
            print("Warning: boilerplate passed through",a[0])
        '''
        a.pop(0)
        for j, elem in enumerate(a):
            if a[j] != "?":
                a[j] = float(elem)
            # Test data label
            else:
                a[j] = -4

        label = a[0]
        a.pop(0)
        feature = a
    return(feature,label)      #self.weights 


class Layer:
    def __init__(self,num_inputs,num_outputs,activation_type = "leaky_relu"):
        # Initialise weights to random value between -1 and 1
        self.weights = 2*np.random.rand(num_outputs,num_inputs) - 1
        self.bias = np.full(num_outputs,0.1)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.activation = Activation_Function(activation_type)
        self.new_cycle()
    def new_cycle(self):
        # Tracking the change to weights
        self.weights_change = np.zeros((self.num_outputs,self.num_inputs))
        self.bias_change = np.zeros(self.num_outputs)
    def normalise_weight_bias_changes(self,num_samples):
        self.weights_change /= num_samples
        self.bias_change /= num_samples
    def get_random_weight_value(self):
        return 2*np.random.rand(1) - 1  
    def get_random_bias_value(self):
        return self.get_random_weight_value()    
    def process(self,input_vector):
        # Perform matrix multiplication and then add bias. Equivalent to acting matrix of weights on the inputs:   1xnum_inputs X num_inputsxnum_outputs [transpsd]   + 1xnum_outputs - > 1xnum_outputs = output vector. 
        transposed_weights = np.matrix.transpose(self.weights)
        self.output = np.dot(input_vector,transposed_weights) + self.bias   # self.bias shape = (num_neurons)
        self.activation.process(self.output)      
    def process_vectorised(self,input_vector_array):
        #print("input vector array:")
        #print(input_vector_array)
        num_instances, num_cols = input_vector_array.shape
        if self.num_outputs != len(self.bias):
            print("Error, unexpected num_outputs, self.bias,",self.num_outputs,self.bias)
        bias_array = np.empty((num_instances,self.num_outputs))
        bias_array[:] = self.bias  # (num_instances,num_neurons)
        transposed_weights = np.matrix.transpose(self.weights)
        self.output_array = np.dot(input_vector_array,transposed_weights) + bias_array   #(num_instances,inputs) X (num_inputs,num_neurons) +(num_instances,num_neurons)  <=== this last one's rows are bias vectors, and it's same for every instance
        #print("output vector array:")
        #print(self.output_array)
        self.activation.process_vectorised(self.output_array)
        #print("activated output array:")
        #print(self.activation.output_array)
    def mutate(self,mutation_rate):
        for x,row in enumerate(self.weights):   #TODO check it is in fact rows then columns.
            for y,val in enumerate(row):
                if np.random.rand(1) < mutation_rate:
                    self.weights[x][y] = self.get_random_weight_value()
        for x, val in enumerate(self.bias):
            if np.random.rand(1) < mutation_rate:
                self.bias[x] = self.get_random_bias_value()
    '''
    # gradient of cost w.r.t. z              cost = |goal - z|.    goal > z: cost = goal - z so dc/z = -1    .   So   want a function F(z) = 1 if z - goal > 0; = -1 if z - goal < 0, 0 otherwise and a rectifier G(z)*F(z - goal) = G(z) = 0 if z > 1 or z < 0
    #                                                                                                            Alternately, can do F(G(z - goal)), but G is instead defined as G(x) = 0 if |x| > 1 
    #Gradient of cost against z, F(G(z - goal)), where G set to 0 if magnitude greater than 1 #TODO test performance by making G not do anything.
    '''
    def get_final_layer_dc_dz_WRONG(self,z,target_output_vector):
        G = copy.deepcopy(z) - copy.deepcopy(target_output_vector)
        G[abs(G) > 1] = 0   #TODO test removing this!
        F = np.heaviside(G,0.5)*2 - 1
        return F
    #TODO fix, important for speed.
    def get_final_layer_dc_dz(self,z,target_output_vector):
        F = np.zeros((len(z)))
        for i,goal in enumerate(target_output_vector):
            #We don't want to reward the algorithm for going below 0 and above 1 for the final value. TODO consider punishing/not rewarding all neurons that go below 0 or above 1.
            if goal != 0 and goal !=1:
                print("Error: Final layer's goal output is not binary!")
            if goal == 1 and z < 1:
                dc_dz = -1
            elif goal == 0 and z > 0:
                dc_dz = 1
            else:
                dc_dz = 0
            F[i] = dc_dz
        #print(F)
        return F
    def get_final_layer_dc_dz_fast(self,z,target_output_vector,mean_square_loss = True):
        z_copy = copy.deepcopy(z)
        squooshed_z = np.clip(z_copy,-1,1)
        goal_vector = target_output_vector
        # now squooshed_z and goal_vector are both, at most, of magnitude 1
        z_overshoot = squooshed_z - goal_vector   # this is proporitonal to the derivative of the mean square loss!
        # same sign and greater than or equal in magnitude to the goal

        if not mean_square_loss:
            # Linear loss function
            z_overshoot[z_overshoot ==0] = 0
            # the target is above (goal == 1, z < 1)
            z_overshoot[z_overshoot < 0] = -1
            # the target is below (goal == 0, z > 0)
            z_overshoot[z_overshoot > 0] = 1 
        return z_overshoot
    def get_final_layer_dc_dz_vectorised(self,_output_batch,target_output_batch,mean_square_loss = True):
        

        output_batch = copy.deepcopy(_output_batch) #TODO try using output_batch.copy()
        squooshed_batch = output_batch.clip(-1,1)
        # now we have a target output vector and a squooshed output vector that represents the confidence in the prediction.
        target_overshoot = squooshed_batch - target_output_batch
        if not mean_square_loss:  # Generally just the derivative in the overshoot as it is the linear loss  
            target_overshoot[target_overshoot ==0] = 0
            target_overshoot[target_overshoot < 0] = -1
            target_overshoot[target_overshoot > 0] = 1 
        return target_overshoot



    def get_final_layer_binary_classification_dc_dz(self,z,target_output_vector):  # divides by zero...
        toxicity_guess = copy.deepcopy(z)
        toxicity_guess[toxicity_guess >= 0] = 1        
        toxicity_guess[toxicity_guess < 0 ] = 0
        dc_dz = (1 - target_output_vector)/(1-toxicity_guess) - (target_output_vector)/toxicity_guess
        return dc_dz
    def get_final_layer_mean_square_loss_dc_dz(self,z,target_output_vector):   # Gives nan values for some reason, but note that we don't expect this to work as it claims to the network that it's wrong when it gives e.g. toxicity = 2 and toxicity is true.
        toxicity_guess = copy.deepcopy(z)
        dc_dz = toxicity_guess - target_output_vector
        return dc_dz
    
    def back_propagate_vectorised(self,input_vector_batch,learning_rate,learning_rate_weights,final_layer,target_output_batch = None,dc_dz_array = None):
        bias_learning_rate_mod = 1
        z0 = self.output_array
        z = self.activation.output_array
        x = input_vector_batch
        if final_layer:          
            dc_dz = self.get_final_layer_dc_dz_vectorised(z,target_output_batch)
        else:
            dc_dz = dc_dz_array 
        dz_dz0 = self.activation.dz_dz0_vectorised(z0)
        dz0_db = 1
        dz_db = dz_dz0*dz0_db # use np.multiply for more complex thing
        dc_db = np.multiply(dz_db,dc_dz)

        num_instances_sampled = len(input_vector_batch)
        #BIAS
        if final_layer:
            dc_db_weighted = np.multiply(dc_db,learning_rate_weights)  # These weights haven't been normalised, since we don't want the weight to depend on the number of samples (and hence don't chuck it into the weights for np.average!)
        else:
            dc_db_weighted = dc_db # already taken into account by dc_dz being weighted
        #average_dc_db = np.average(dc_db,axis=0,weights=learning_rate_weights)
        average_dc_db = np.average(dc_db_weighted,axis=0)     # learning_rate_weights S = [num_instances,num_outputs]
        self.bias_change -= bias_learning_rate_mod*learning_rate*average_dc_db
        #WEIGHTS         
        #print("THIS IS X",x) #PPp
        if len(z.shape) == 1:
            z = z.reshape((len(z),num_outputs)) 
            print("WARNING WRONG DIM FOR Z")
        if len(x.shape) == 1:
            x = x.reshape((len(x),num_outputs))  
            print("WARNING WRONG DIM FOR x")
        #print(x)
        #print(z[0])
        #print(x[0])

        dz0_dw_matrix = np.zeros((len(x),len(z[0]),len(x[0])))  #num instances [num_instances], length of output of one instance [rows], length of input of one instance [columns]
        #print(dz0_dw_matrix.shape)
        dz0_dw_matrix[:,:] = x[:,None] + dz0_dw_matrix[:]        # replace every instance_index with a matrix of size [z[0],x[0]] which is just copies of the x rows.
        #print(dz0_dw_matrix.shape)
        dz0_dw_matrix = dz0_dw_matrix.transpose(0,2,1)     # this gives us [num instances X repeat_columns_of_x_with_length_of_input X length_output]

        #print(dc_dz)
        #print("hoopa")
        #print(dz_dz0)
        dc_dz0 = np.multiply(dz_dz0,dc_dz)    # this is rows of dc_dz0 corresponding to each instance [instance X outputlength]
        
        #WEIGHTED FROM HERE
        # Weight the dc_dz0 by the learning rate weights
        if final_layer:
            dc_dz0_weighted = np.multiply(dc_dz0,learning_rate_weights)
        else:
            dc_dz0_weighted = dc_dz0 # already taken into account by dc_dz being weighted
        
        dc_dz0_matrix = np.zeros((len(x),len(x[0]),len(z[0])))                                                                 
        dc_dz0_matrix[:,:] = dc_dz0_weighted[:,None] + dc_dz0_matrix[:]   # this gives us, for every instance, a matrix of rows equal to the vector dc_dz0                                                                         
        dc_dw_tensor = np.multiply(dz0_dw_matrix,dc_dz0_matrix) #multiplies element by element :)
        #Transpose because of the goofy way I set up everything
        dc_dw_tensor = dc_dw_tensor.transpose(0,2,1) # s = [instances, outputs, inputs]
        average_dc_dw_tensor = np.average(dc_dw_tensor,axis = 0) 
        self.weights_change -= learning_rate*average_dc_dw_tensor
        # END WEIGHTED

        # Determine 3D VECTORISED ARRAY of dc/dx to pass back, DIFFERENT shape to the non-vectorised case. Low on time so just reusing self.dc_dx_array soz.                                                                                         # We need to make the tensor of dc/dx, but now we can use dz/dz0! The number of rows of the dz0_dx_matrix is equal to the number of outputs z, so the elements of the dz0_dx matrix with zs corresponding to the same index will be along the columns. Everything is all good!       
        dz0_dx_matrix = self.weights   #shape=(outputs,inputs)                                                                   #dz_dx = np.multiply(dz0_dx_matrix*dz_dz0_matrix)  #shape = (inputs)

        #Note passing back weighted dc_dz0
        self.dc_dx_array = np.dot(dc_dz0_weighted,dz0_dx_matrix)    #(instances,outputs) X (outputs,inputs)  ==> s = (instances,inputs)




    def back_propagate(self,input_vector,learning_rate,final_layer,target_output_vector = None,dc_dz_array = None):
        # General process: We are iterating through each layer (backwards) providing the layer with target outputs, and storing the desired modifications to the weights/biases to gradient descent towards the target. Each layer provides a target output to the previous layer, 
        # OUTDATED - that is nudged in a way that also gradient descends towards the target output. Note we do not use the new weights/biases, since the new weights/biases will be determined by averaging the changes across multiple instances.
        # The modifications to the weights/biases are made after averaging the desired modifications through testing multiple instances.
        '''#if target_output_vector == None:
           target_output_vector = self.output'''

        bias_learning_rate_mod = 1
        #change_target_learning_rate_mod = 1
        #self.target_input = copy.deepcopy(input_vector)
        
        # c = error, z = activation output, z0 = layer output, before activation, w = weight, x = input at index j

        z0 = self.output             #1xlen
        z = self.activation.output   #1xlen
        x = input_vector 
        
        # gradient vector of cost against output
        if final_layer:
            dc_dz = self.get_final_layer_dc_dz_fast(z,target_output_vector)
        else:
            dc_dz = dc_dz_array
        # gradient vector of activation function, z0 = the raw input vector to the function.
        dz_dz0 = self.activation.dz_dz0(z0)
        # gradient vector of output against bias
        dz0_db = 1
        dz_db = dz_dz0*dz0_db
        # gradient vector of cost against bias (element-wise multiplication)
        dc_db = np.multiply(dz_db,dc_dz)

        # Calculate and make changes
        #BIAS
        self.bias_change -= bias_learning_rate_mod*learning_rate*dc_db
        #WEIGHTS
        # We need to create the tensor (dz/dw)i,j = d/dwj(dz/dz0 *z0i) = xj*(dz/dz0)i    , where the final j is the number of inputs. 
        
        #We could use this, (edit we dont use it lol) then multiply by element with dz0/dw, which would give us dz/dw, but let's save it for later instead as then we don't have to mess around too much to get dc/dw from dz/dw.
        dz_dz0_matrix = np.zeros((len(x),len(z)))
        dz_dz0_matrix[:] = dz_dz0 

        dz0_dw_matrix = np.zeros((len(z),len(x)))
        dz0_dw_matrix[:] = copy.deepcopy(x)
        dz0_dw_matrix = np.matrix.transpose(dz0_dw_matrix) # this gives us a matrix of columns equal to the vector x

        dc_dz0 = np.multiply(dz_dz0,dc_dz) 
        dc_dz0_matrix = np.zeros((len(x),len(z)))                                                                        
        dc_dz0_matrix[:] = copy.deepcopy(dc_dz0)     # this gives us a matrix of rows equal to the vector dc_dz0      #WHY DOES THIS NEED TO BE cOPIED FOR LATER???             
        dc_dw_tensor = np.multiply(dz0_dw_matrix,dc_dz0_matrix)#np.multiply(dz0_dw_matrix,dc_dz0_matrix)
        #Transpose because of the goofy way I set up everything
        dc_dw_tensor = np.matrix.transpose(dc_dw_tensor)
        self.weights_change -= learning_rate*dc_dw_tensor
        

        # Determine dc/dx to pass back.                                                                                          # We need to make the tensor of dc/dx, but now we can use dz/dz0! The number of rows of the dz0_dx_matrix is equal to the number of outputs z, so the elements of the dz0_dx matrix with zs corresponding to the same index will be along the columns. Everything is all good!       
        dz0_dx_matrix = self.weights   #shape=(outputs,inputs)                                                                   #dz_dx = np.multiply(dz0_dx_matrix*dz_dz0_matrix)  #shape = (inputs)
        #self.dc_dx_array = np.multiply(dc_dz0_matrix,dz0_dx_matrix)   # dc_dz0 has rows of vectors of length num_output, dz0_dx_matrix has rows of  
        #self.dc_dx_array = np.matrix.transpose(self.dc_dx_array)[0]    
        self.dc_dx_array = np.dot(dc_dz0,dz0_dx_matrix)    #(1,outputs) X (outputs,inputs)
        #self.dc_dx_array = np.matrix.transpose(self.dc_dx_array)  #UNECCESARY?

# Note that after the final layer's activation function, we apply the classifier's final_activation method.
class Activation_Function:
    def __init__(self,type="leaky_relu"):
        #TODO Can perhaps do perceptron implementation (heaviside)?
        if type == "relu":
            self.process = self.relu   
        elif type == "leaky_relu":
            self.process = self.leaky_relu
            self.dz_dz0 = self.dz_dz0_leaky_relu
            self.dz_dz0_vectorised = self.vectorised_dz_dz0_leaky_relu
            self.leaky_gradient = 1/100 # :'0
        elif type == "double_leaky_relu":
            self.process = self.double_leaky_relu
            self.dz_dz0 = self.dz_dz0_double_leaky_relu
            self.dz_dz0_vectorised = self.vectorised_dz_dz0_double_leaky_relu
            self.leaky_gradient = 1/100 # :'0
        elif type == "sigmoid":
            self.process = self.sigmoid
            self.dz_dz0 = self.dz_dz0_sigmoid
        elif type == "modified_sigmoid":
            self.process = self.sigmoid_modified
            self.dz_dz0 = self.dz_dz0_sigmoid_modified
            self.sigmoid_mod = 1.3
        elif type == "linear":
            self.process = self.linear
            self.dz_dz0 = self.dz_dz0_linear
        elif type == "softmax":
            self.process = self.softmax
            self.dz_dz0 = self.dz_dz0_softmax
        else:
            print("Error: activation function type of",type,"is not valid!")
    # takes in an array of neuron outputs (each row corresponds to on instance), and applies the activation function.
    def process_vectorised(self,input_vector_array):
        self.output_array = np.apply_along_axis(self.process,1,input_vector_array) 
        return self.output_array
    def linear(self,input_vector):
        self.output = copy.deepcopy(input_vector)            
        #print("linear output",self.output)
        if True in np.isnan(self.output):
            HOOGABOOGA
        return self.output
    #Rectified linear unit
    def relu(self, input_vector):
        self.output = np.maximum(0,input_vector)
        return self.output
    # Leaky relu, to avoid (vanishing gradient? or saturating gradient?) problem. According to wikipedia https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Potential_problems, performance is reduced with a leaky relu. 
    # The vanishing gradient issue occurs with too high a learning rate, so we can try reducing the leaky gradient a lot lower and use a lower learning rate (to effectively give a relu)
    def leaky_relu(self,input_vector):
        self.output = np.maximum(input_vector*self.leaky_gradient,input_vector)      
        return self.output
    # Designed for final activation layer.
    def double_leaky_relu(self, input_vector):   
        input_copy = (copy.deepcopy(input_vector) + 1)/2            #TODO consider this (+ 1)/2  does it reverse effective halving of learning rate by final activation? TODO need to be more general so that it applies for any final layer activation
        input_squooshed = np.maximum(input_copy*self.leaky_gradient,input_copy)
        input_squooshed = np.minimum(input_squooshed*self.leaky_gradient + 1, input_squooshed)
        self.output = input_squooshed
        return self.output
    def sigmoid(self,input_vector):
        self.output = expit(input_vector)
    # We modify the sigmoid so that the algorithm can make "confident" decisions, where dc/dz = 0
    def sigmoid_modified(self,input_vector):
        self.output = expit(input_vector)*self.sigmoid_mod
        return self.output
    def softmax(self,input_vector):
        self.output = softmax(input_vector)
        return self.output
    # Returns the rate of change of an value of an output vector by the activation function at certain index, w.r.t. the value of the input vector at the same index. i.e. how do the outputs change w.r.t. the inputs.
    # Note we multiply z by 2 and minus one, but the dz_dz0 is still proportional so it's fine.
    def dz_dz0_linear(self,_input_z0_vector):
        dz_dz0 = copy.deepcopy(_input_z0_vector)
        dz_dz0[:] = 1
        return  dz_dz0
    def dz_dz0_leaky_relu(self,_input_z0_vector):
        input_z0_vector = copy.deepcopy(_input_z0_vector)
        return (1-self.leaky_gradient)*np.heaviside(input_z0_vector,1) + self.leaky_gradient   #TODO test performance changing the x = 0 value to be 0 for the heaviside function, rather than 1.     
    def dz_dz0_double_leaky_relu(self,_input_z0_vector):
        input_z0_vector = (copy.deepcopy(_input_z0_vector) + 1)/2
        #print("z0_vect",input_z0_vector) ##PPPP
        mask1 = (0 <= input_z0_vector)*(input_z0_vector <= 1)
        input_z0_vector[mask1] = 1  #WHY THE <heck> WAS THIS THERE error, had this as input_z0_vector[:] = 1 
        input_z0_vector[input_z0_vector < 0] = self.leaky_gradient
        input_z0_vector[input_z0_vector > 1] = self.leaky_gradient
        return input_z0_vector
    def dz_dz0_sigmoid(self,_input_z0_vector):
        input_z0_vector = copy.deepcopy(_input_z0_vector)
        s = expit(input_z0_vector)
        return s*(1-s)
    def dz_dz0_sigmoid_modified(self,_input_z0_vector):
        input_z0_vector = copy.deepcopy(_input_z0_vector)
        s = expit(input_z0_vector)
        return s*(1-s)*self.sigmoid_mod
    def dz_dz0_softmax(self,_input_z0_vector):
        input_z0_vector = copy.deepcopy(_input_z0_vector)
        s = softmax(input_z0_vector)
        return s*(1-s)
    def vectorised_dz_dz0_leaky_relu(self,_input_z0_array):
        result = np.apply_along_axis(self.dz_dz0_leaky_relu,1,_input_z0_array) 
        return result
    def vectorised_dz_dz0_double_leaky_relu(self,_input_z0_array):
        result = np.apply_along_axis(self.dz_dz0_double_leaky_relu,1,_input_z0_array) 
        #print("dz_dz0_vector:",result) ##PPPP
        return result
        # Remember the arg is is the output array of the neuron



class Outputs:
    def __init__(self):
        self.toxicity = None       
        #self.non_toxicity = None

class Classifier:
    # view - set to 1 to use first half of features, set to 2 to use second half. Full view used if not provided. TODO This is a rushed implement.                                                                                                OLD a tuple indicating the range of data that the model will use. Full view used if not provided. 
    # For this project, due to me DYING with lack of time (help), runs with the identity labels (potentially not?) as features will always be the full view, so the view doesn't need to be modified depending on this. 
    def __init__(self,_name, neurons_per_layer,num_hidden_layers,num_inputs,num_outputs,_view = None):
        if _view != None:
            num_inputs/= 2   # VERY BAD VERY BAD CODING TODO but we only have 384 inputs so it's fine for THIS SPECIFIC case.
            num_inputs = int(num_inputs)
        self.view = _view
        self.name = _name
        self.layers = []
        input_layer = Layer(num_inputs,neurons_per_layer)
        self.layers.append(input_layer)
        global last_layer_activation_name
        for i in range(num_hidden_layers):
            hidden_layer = Layer(neurons_per_layer,neurons_per_layer)
            self.layers.append(hidden_layer)
        # last layer
        if one_hot:
            last_layer_activation_name = "softmax"
        elif linear_output_activation:
            last_layer_activation_name = "linear"
        else:
            last_layer_activation_name = "sigmoid"
            last_layer_activation_name = "double_leaky_relu"
        output_layer = Layer(neurons_per_layer,num_outputs,last_layer_activation_name)
        self.layers.append(output_layer)
        self.output = Outputs()
    def learn_and_reset_for_new_cycle(self,num_instances_sampled,printin=False,vectorised = False):
        for layer in self.layers:
            # Average the weights and bias changes by dividing by the number of iterations
            if not vectorised:
                layer.normalise_weight_bias_changes(num_instances_sampled)
            # Modify weights and biases
            layer.weights += layer.weights_change
            layer.bias += layer.bias_change
            if printin:
                print("layer change:")
                print(layer.weights_change)
                print(layer.bias_change)
            layer.new_cycle()
    def forward_only(self,input_vector):
        next_input_vector = input_vector
        for layer in self.layers:   #I had "for layer in model_1.layers" here -_-
            layer.process(next_input_vector)
            next_input_vector = layer.activation.output
            ''' Leakless Relu test
            if next_input_vector[0] < 0:
                print("Warning: next input vector has negative values")
                print(next_input_vector)
            '''
        self.output.toxicity = self.final_activation(next_input_vector[0])
        #print(self.output.toxicity)
        #self.output.non_toxicity = next_input_vector[1]
    def forward_pass_vectorised(self,input_vector_array):
        next_input_vector_array = input_vector_array
        for layer in self.layers:
            layer.process_vectorised(next_input_vector_array)
            next_input_vector_array = layer.activation.output_array
        self.output.outputs_vectorised = self.final_activation_vectorised(next_input_vector_array)

    # WARNING this needs to be linear, or else dz_dz0 won't be proportional to the REAL dz_dz0!
    def final_activation(self,vector):
        if linear_output_activation:
            return vector 
        # Effectively halves learning rate? Since we don't include the 2 factor in the dz_dz0 atm. TODO
        return 2*vector - 1
    def final_activation_vectorised(self,vector_array,return_all_outputs = True):
        # Output is in the form of [[-1.01],[-0.8],[0.4]] so take slice
        toxicity_array = vector_array[:,0]
        if return_all_outputs:
            result = np.copy(vector_array[:,:])
        else:
            result = toxicity_array
        if linear_output_activation:
            return result
        # Effectively halves learning rate? Since we don't include the 2 factor in the dz_dz0 atm. TODO
        return 2*result - 1

    def mutate(self,mutation_rate):
        old_weight_test = copy.deepcopy(self.layers[0].weights)
        for l,layer in enumerate(self.layers):
            old_layer = copy.deepcopy(layer)
            mod_mutation_rate = mutation_rate
            # Give more likelihood for later layers to mutate, since a small number of neurons in these layers gives them greater impact
            if l > 0:   #TODO refine this
                mod_mutation_rate = get_hidden_layer_mutation_rate(mutation_rate)
            layer.mutate(mod_mutation_rate)
            '''
            if np.array_equal(old_layer.weights, self.layers[l].weights):
                print("WARNING, same weights layer")
            '''
        '''
        if np.array_equal(old_weight_test,self.layers[0].weights):
            print("Warning: same weights")
        '''
    def back_propagate(self,original_input_vector,_target_output_vector,learning_rate,vectorised = False):
        learning_rate_weights = np.copy(_target_output_vector)  # S = [instances,outputs]
        #print("target output vector")
        #print(learning_rate_weights)
        if weighted_learning:
            #Weighted learning: magnify the less common output - e.g. twice as prevalent --> twice the weight. If all same, all have weight of 1.
            if not vectorised:
                print("warning, weighted learning not implemented for non-vectorised stuff")
            tot_count = len(_target_output_vector)
            prob_true = np.sum(learning_rate_weights,axis=0)/tot_count
            #Note prob_true == 0 or == 1 means it has a weighting of 0

            prob_true_matrix = np.zeros((learning_rate_weights.shape))
            prob_true_matrix[:] = prob_true

            mask1 = (learning_rate_weights == 0)
            mask2 = (learning_rate_weights == 1) 
            total_prob = ((1-prob_true_matrix)**2 + prob_true_matrix**2)*tot_count

            learning_rate_weights[mask1] = ((prob_true_matrix)/total_prob*total_prob*2)[mask1]
            learning_rate_weights[mask2] = ((1-prob_true_matrix)/total_prob*total_prob*2)[mask2]   
        else:
            learning_rate_weights[learning_rate_weights] = 1     
        
        #print("learning rate weights")
        #print(learning_rate_weights)
        for i in range(len(self.layers)):
            layer = self.layers[-(i+1)]
            final_layer = False
            if i == 0: 
                final_layer = True      
                _dc_dz = None
            else: 
                _target_output_vector = None    
                next_layer = self.layers[-(i+1-1)]
                _dc_dz = next_layer.dc_dx_array
            if i != len(self.layers) - 1:
                previous_layer = self.layers[-(i+2)]
                input_vector = previous_layer.activation.output
                if vectorised:
                    input_vector = previous_layer.activation.output_array
            else:
                input_vector = original_input_vector
            if vectorised:

                layer.back_propagate_vectorised(input_vector,learning_rate,learning_rate_weights,final_layer,target_output_batch = _target_output_vector,dc_dz_array = _dc_dz)   
            else:                
                layer.back_propagate(input_vector,learning_rate,final_layer,target_output_vector = _target_output_vector,dc_dz_array = _dc_dz)                  
            
class Accuracy_Tester():
    def __init__(self,_dev_fname,_output_fname):
        self.accuracies = []
        self.iteration_indices = []
        self.dev_fname = _dev_fname
        self.output_fname = _output_fname
        labels_and_instances,self.starting_id = preprocess_test(_dev_fname)
        self.instances = slice_instances_2dim(labels_and_instances)
    def store_accuracy(self,model,true_iteration):
        make_final_predictions(model,self.output_fname,self.instances,self.starting_id)
        acc = assess_final_predictions(self.dev_fname,output_fname)
        print("iteration (epoch)",true_iteration,"accuracy of model against dev data:",acc)
        self.accuracies.append(acc)  
        self.iteration_indices.append(true_iteration)  
    def plot(self,show_plot,custom_name_ext = None):
        # Get error rate
        y = self.accuracies
        for i in range(len(y)):
            y[i] = (1 - y[i])*100
        x = self.iteration_indices
        for i in range(len(x)):
            x[i] = x[i]/1000
        plt.title("Error rate evolution")
        plt.xlabel("Batch no. (1e4)")
        plt.ylabel("Test error (%)")
        plt.plot(x,y)
        if custom_name_ext == None:
            plt.savefig('accuracy_plot_' + str(run)+'.png')
        else:
            plt.savefig('accuracy_plot_' + custom_name_ext +'.png') 
        if show_plot:
            plt.show()


def get_hidden_layer_mutation_rate(base_mutation_rate):
    return 100*base_mutation_rate

def make_final_predictions(_model,output_fname,instances,starting_id,vectorised = True):
        _header = ["ID","Toxicity"]

        
        current_id = starting_id
        if vectorised:
            instances = np.array(instances)    #TODO may need to use new variable name o.o
            #### Get view of instances
            if _model.view == 1:
                instances_array_view = np.array(instances)[:,int(len(instances[0])/2):]
            elif _model.view == 2:
                instances_array_view = np.array(instances)[:,:int(len(instances[0])/2)]
            else:
                instances_array_view = np.array(instances)            
            _model.forward_pass_vectorised(instances_array_view)
            output_array = _model.output.outputs_vectorised
            output_array[output_array >= 0] = 1
            output_array[output_array < 1] = 0
            output_array = output_array.astype("int")
            output_array = output_array.astype("str")

            id_array = np.arange(starting_id,len(instances)+starting_id)
            print("LENGTHS")
            print(output_array[0])
            print(id_array[0])
            print(len(output_array))
            print(len(id_array))
            id_array = id_array.reshape((len(id_array),1))
            comb_arry = np.concatenate((id_array,output_array),axis=1)
            #df_1 = pd.DataFrame({'X':id_array,'Y':output_array})
            df_1 = pd.DataFrame(comb_arry)

            if len(comb_arry) > len(_header):
                _header += range(len(comb_arry[0]) - len(_header))
            for i in range(len(_header)):
                _header[i] = str(_header[i])
            df_1.to_csv(output_fname,header=_header,index = False,encoding=file_encoding)

        else:
            with open(output_fname,'w+',newline='',encoding = file_encoding) as f:
                writer = csv.writer(f)
                writer.writerow(_header)             
                for instance in instances:
                    _model.forward_only(instance)
                    #Determine the prediction from the encoding. Assume toxic when ambiguous.
                    if _model.output.toxicity >= 0:
                        toxicity = "1"
                    else:
                        toxicity = "0"
                    row = [current_id,toxicity] 
                    writer.writerow(row)
                    current_id += 1

def make_predictions_with_confidence(_model,output_fname,instances,instances_starting_id,vectorised = True):
    with open(output_fname,'w+',newline='',encoding = file_encoding) as f:
        #_header = ["ID","Toxicity"]
        if vectorised:
            instances_array = np.array(instances)
            #### Get view of instances
            if _model.view == 1:
                instances_array_view = np.array(instances)[:,int(len(instances[0])/2):]
            elif _model.view == 2:
                instances_array_view = np.array(instances)[:,:int(len(instances[0])/2)]
            else:
                instances_array_view = instances_array
            # Predict labels
            _model.forward_pass_vectorised(instances_array_view)
            output_array = _model.output.outputs_vectorised
            #output_array= 1/2*(output_array+1)     #TODO make this dependent on the final activation layer func
            # Get the average confidence for the outputs for each instance, and the confidence for the toxicity specifically.
            toxicity_confidence_array = output_array[:,0]
            confidence_array = np.average(np.abs(output_array),axis=1)
            probability = 2*min_confidence - 1
            #get the unlabelled confident instances, such that, assuming probabilities are accurately represented, we get only results that are (min_confidence*100)% likely to be true.
            idx = (confidence_array >= probability)*(toxicity_confidence_array>=probability) #(confidence_array<=min_confidence)*(confidence_array >= 1 + min_confidence)
            confident_indices = np.where(idx)
            confident_instances = np.take(instances_array,confident_indices,axis=0)[0]
            confident_instances = confident_instances.tolist()
            #Get the unlabelled instances that weren't confident
            idx2 = (confidence_array < probability)|(toxicity_confidence_array < probability)
            unconfident_indices = np.where(idx2)
            unlabelled_instances = np.take(instances_array,unconfident_indices,axis=0)[0]
            unlabelled_instances = unlabelled_instances.tolist()            
            # Create list of labels for confident guesses
            confident_labels = np.copy(output_array)
            confident_labels[confident_labels >= 0] = 1
            confident_labels[confident_labels < 1] = 0
            confident_labels = np.take(confident_labels,confident_indices,axis=0)[0]
            confident_labels = confident_labels.tolist()


            
            output_array = output_array.astype("str")

            id_array = np.arange(instances_starting_id,len(instances)+instances_starting_id)
            id_array = id_array.reshape((len(id_array),1))
            #print("LENGTHS")
            #print(len(output_array))
            #print(len(id_array))
            comb_arry = np.concatenate((id_array,output_array),axis=1)
            df_1 = pd.DataFrame(comb_arry)
            #df_1 = pd.DataFrame ({'X':id_array,'Y':output_array})

            df_1.to_csv(output_fname,index = False,encoding=file_encoding)
            #confident_instances = confident_instances[0]
            #confident_labels = confident_labels[0]
            return confident_instances, confident_labels,unlabelled_instances
        
        '''
        
        writer = csv.writer(f)
        header = ["ID","Toxicity"]
        writer.writerow(header)

        current_id = starting_id
        for instance in instances:
            chosen_model.forward_only(instance)
            if chosen_model.output.toxicity >= 0:
                toxicity = "1"
            else:
                toxicity = "0"
            row = [current_id,toxicity] 
            writer.writerow(row)
            current_id += 1
        '''

def assess_final_predictions(_actual_data,_predictions):
    df1 = pd.read_csv(_actual_data,dtype=str)
    df2 = pd.read_csv(_predictions,delimiter = ',',dtype = str)
    num_common_elements_array = df1.isin(df2).sum()
    num_elements = num_common_elements_array['ID']
    num_common_elements = num_common_elements_array['Toxicity']
    accuracy = num_common_elements/num_elements
    return accuracy

def assess_final_predictions_old(_actual_data,_predictions):
    accuracy = 0
    with open(_actual_data, newline='',encoding = file_encoding) as actual_data_file:
        actual_data_reader = csv.reader(actual_data_file, delimiter=' ', quotechar='|')
        real_labels = []
        # Get list of labels.
        for i,row in enumerate(actual_data_reader):
            if i != 0:
                a = row[0]
                a = a.split(",")
                label = a[1]
                real_labels.append(label)    

    with open(_predictions, newline='',encoding = file_encoding) as predictions_file:
        predictions_reader = csv.reader(predictions_file, delimiter=' ', quotechar='|')
        # get list of guess labels
        guesses = []
        for i, row in enumerate(predictions_reader):
            if i != 0:
                a = row[0]
                a = a.split(",")
                label = a[1]
                guesses.append(label)
    for i, label in enumerate(real_labels):
        if label == guesses[i]:
            accuracy += 1
    accuracy /= len(guesses)
    return accuracy


def assess_predictions_with_confidence(_actual_data,_predictions,min_confidence = 0):
    with open(_actual_data, newline='',encoding = file_encoding) as actual_data_file:
        actual_data_reader = csv.reader(actual_data_file, delimiter=' ', quotechar='|')
        real_labels = []
        # Get list of labels.
        for i,row in enumerate(actual_data_reader):
            if i != 0:
                a = row[0]
                a = a.split(",")
                label = a[1]
                real_labels.append(label)    

    with open(_predictions, newline='',encoding = file_encoding) as predictions_file:
        predictions_reader = csv.reader(predictions_file, delimiter=' ', quotechar='|')
        # get list of labels, but labels are given their confidence value, with a positive sign for toxic, negative sign for not toxic.
        guesses_with_confidence = []
        for i, row in enumerate(predictions_reader):
            if i != 0:
                a = row[0]
                a = a.split(",")    

#############################################################################main - ?Genetic algorithm?
def train_model(fname,model,instances,instance_labels,learning_shape,min_rate,max_rate,num_epochs,interval,vectorised_instances = False):


    ####
    if model.view == 1:
        instances = np.array(instances)[:,int(len(instances[0])/2):]
    elif model.view == 2:
        instances = np.array(instances)[:,:int(len(instances[0])/2)]
        
    num_instances = len(instances)    
    print("Start")


    print("training on: ", fname)

    original_data_fname = fname
    outputs = Outputs()
    accuracy_tester = Accuracy_Tester(dev_fname,output_fname)
    #/////
    preprocessing_all = True

    '''
    if not preprocessing_all:
        del instances
        del instance_labels
        gc.collect()
    '''

    print("Number of instances",num_instances)
    print("Batch size:",interval)
    num_iterations = int(num_instances/interval)
    print("Num total iterations:",num_iterations*num_epochs)
    best_accuracy = 0

    print("Learning shape:",learning_shape)

    average_accuracy = 0
    for repeat in range(num_epochs):
        if repeat != 0:
            # Shuffle the data together.
            zipped = list(zip(instances, instance_labels))
            random.shuffle(zipped)
            instances, instance_labels = zip(*zipped)
        for iteration in range(num_iterations):     

            true_iteration = num_iterations*repeat + iteration
            true_num_iterations = num_epochs * num_iterations

            #temporary until update to python 10
            python_10 = False # :(
            # linear learning rate
            if learning_shape == "L":
                learning_rate = max_rate - (max_rate - min_rate)*(true_iteration/true_num_iterations)
            # Decreasing exponential            
            if learning_shape == "E":
                _b = np.log(max_rate/min_rate)/true_num_iterations
                learning_rate = max_rate*np.exp(-_b*true_iteration)      #watch out, HE has appeared!! AAAA (-_b)
            if learning_shape == "C":
                learning_rate = "ERROR: not implemented :("
            '''
            if python_10:
                match learning_shape:
                    # linear learning rate
                    case "L":
                        learning_rate = max_rate - (max_rate - min_rate)*(true_iteration/true_num_iterations)
                    # Decreasing exponential
                    case "E":
                        _b = np.log(max_rate/min_rate)/true_num_iterations
                        learning_rate = max_rate*np.exp(-_b*true_iteration)
                    # Cyclical "Ideal: keep dabbing in to different potentials until we find it's too deep to escape."
                    case "C":
                        learning_rate = "ERROR: not implemented :("
            '''
            if true_iteration%10000 == 0 or (true_iteration < 10000 and true_iteration%500 == 0):
                print("Iteration:",true_iteration,"/",num_iterations*num_epochs,"learning rate:",learning_rate)

            #Print out
            if iteration%num_iterations == 0:
                print("Cycle:",repeat,"learning rate:",learning_rate)
            if (iteration !=0 or repeat != 0) and model.accuracy > best_accuracy:
                best_accuracy = model.accuracy
                print("New best accuracy:",best_accuracy) 

            model.accuracy = 0

            # VECTORISED
            if vectorised_instances:
                if not preprocessing_all:
                    print("ERROR orange :C")
                    return orange
                current_index = iteration*interval
                instances_batch = np.array(instances[current_index:min(current_index+interval,len(instances))])
                #print("INSTANCES BATCH:",instances_batch) #ppp
                labels_batch = np.array(instance_labels[current_index:min(current_index+interval,len(instance_labels))])
                labels_batch_slice = labels_batch # for some reason o.o  #= labels_batch[:,0]
                if num_outputs == 1:
                    labels_batch_slice = labels_batch_slice.reshape((len(labels_batch_slice),num_outputs))  

                model.forward_pass_vectorised(instances_batch)
                model.back_propagate(instances_batch,labels_batch_slice,learning_rate,vectorised = True)
                
            else:


                #TODO vectorise this?
                for i in range(interval):
                    #index = i
                    index = i+iteration*interval   # we go through new data so that we don't overfit! 
                    if preprocessing_all:
                        instance = instances[index]
                        label = instance_labels[index]
                    else:
                        instance,label = fast_process_specific_instance(fname,index)

                    model.forward_only(instance)
                    
                    # BACK PROPAGATION
                    if label != 0 and label != 1: 
                        print("Warning: Label not binary!")
                    model.back_propagate(instance,np.array([label]),learning_rate)
                    # Assess models' accuracy. Assume toxic when ambiguous.
                    if model.output.toxicity >= 0 and label == 1:
                        model.accuracy += 1/interval
                    if model.output.toxicity < 0 and label == 0:
                        model.accuracy += 1/interval
            average_accuracy += model.accuracy
            # Model learns at end of iteration (batch)
            print_changes = False
            print_period = 30
            print_factor = 100
            og_print_factor = print_factor
            if vectorised_instances and true_iteration > 2*print_period*print_factor + 1:
                print_factor = 1000
            if true_iteration%(print_period*print_factor) == 0:
                print_changes = True
            print_layer_changes = print_changes
            if not global_print_layer_changes:
                print_layer_changes = False 
            model.learn_and_reset_for_new_cycle(interval,print_layer_changes,vectorised_instances)   
            if not vectorised_instances:
                if print_changes:
                    print("average accuracy:",average_accuracy/(print_period*print_factor))    
                    average_accuracy = 0          
                
                if true_iteration%(print_factor) == 0:
                    print("Model accuracy:",model.accuracy)

            if true_iteration%(og_print_factor*256/interval/5*10) == 0:
                accuracy_tester.store_accuracy(model,true_iteration)
    print(accuracy_tester.accuracies)
    return model,accuracy_tester


#%%
if encoded:
    file_encoding = 'utf-8-sig'
else:
    file_encoding = 'utf-8'

prediction_path
tfname = train_data_path
unlabelled_fname = unlabelled_data_path
dev_fname = dev_data_path
test_fname = test_data_path
dataset_path = path_for_temp_files
output_fname = path_for_temp_files + "output_dev.csv" 
#%%
# Main
# Preprocessing




import json 

def create_model(model_name,instances,num_hidden_layers,neurons_per_layer,view = None):
    num_inputs = len(instances[0]) 
    num_instances = len(instances)  
    print("Blank model created:")
    print("neurons per hidden layer",neurons_per_layer)
    print("num hidden layers",num_hidden_layers)
    total_num_weights_and_biases = neurons_per_layer*(num_inputs + 1) + neurons_per_layer *(num_hidden_layers-1)*(neurons_per_layer + 1) + num_outputs*(neurons_per_layer+1)
    print("num_inputs:",num_inputs)
    return Classifier(model_name,neurons_per_layer,num_hidden_layers,num_inputs,num_outputs,_view = view)

def save_model(folder,model):
    json_fname = folder + model.name + ".json"
    out_file = open(json_fname,"w+")
    json.dump(model,out_file,indent=6)

def load_model(folder,model_name):
    json_fname = folder + model_name + ".json"
    in_file = open(json_fname)
    model = json.load(in_file)
    return model

lazy_and_wanna_try_combined_data = False
use_small_data = False #!!!!!!!! SMALL DATA PARAM !!!!!!!!
if use_small_data:
    encoded = False
if encoded:
    file_encoding = 'utf-8-sig'
else:
    file_encoding = 'utf-8'
# some unsorted variables
# Initial run
#temp fname - the source dataset for cloning into the working dataset at start.
# other file names
working_data_fname = dataset_path + "working_dataset.csv"  
working_unlabelled_fname = dataset_path + "working_unlabelled_dataset.csv"
confidence_fname = dataset_path + "confidence.csv"
#create_working_dataset(working_data_name,tfname)
dfname = tfname
#dfname = working_data_name

max_num_instances = 9999999999
recycle_toxic_data = False#True  # Need to do this if do not WEIGHT the data by the correct output. 

#Learning rate params
learning_shape = "E"   
min_rate = 0.001 #0.01 - 0.001
max_rate = 0.1#0.5#0.5 - 1  #warning: 0.5 too high for 5 layers (use 0.1), but ok for 3.

original_train_labels_and_instances = preprocess(dfname,max_number_of_instances=max_num_instances,recycle_toxic = recycle_toxic_data)
original_unlabelled_labels_and_instances, original_unlabelled_starting_id = preprocess_test(working_unlabelled_fname)




# OLD CELL (removed for memory saving)

#Run models
global_print_layer_changes = False # Whether to occasionally print the changes to weights/biases through back propagation. 
#Get view of train_instances
num_outputs = 25
def slice_instances_2dim(list,_num_outputs = None):
    if _num_outputs == None:
        _num_outputs = num_outputs
    a = np.array(list)
    b = a[:,_num_outputs:]
    return b
def slice_unlabelled_instances_2dim(list):
    a = np.array(list)
    b = a[:,0:]
    return b
def slice_labels_2dim(list,_num_outputs = None):
    if _num_outputs == None:
        _num_outputs = num_outputs
    a = np.array(list)
    b = a[:,0:_num_outputs]
    return b
train_labels_and_instances = original_train_labels_and_instances#np.copy(original_train_labels_and_instances)
#train_labels_and_instances = train_labels_and_instances.tolist()
# Copy stuff
import os
if os.path.exists(working_data_fname):
    os.remove(working_data_fname)
if os.path.exists(working_unlabelled_fname):
    os.remove(working_unlabelled_fname)
dfname = tfname
shutil.copyfile(dfname,working_data_fname)
dfname = working_data_fname
shutil.copyfile(unlabelled_fname,working_unlabelled_fname)


# Run models
# semi-supervised params
num_initial_training_runs = 13
min_confidence = 0.75 


# Create blank models
#batch size
interval = 256 # num instances = 37152, 37152/288 = 129, 37152/32 = 1161
num_epochs = 5#5-10-20-200  NUM EPOCHS
num_hidden_layers = 3 # Need to cite non-linear paper thing,  that we only need 2 layers
neurons_per_layer = 22#12#int(num_instances/(3*(num_inputs+num_outputs)))  # Preventing overfitting - see "how to choose the number of hidden layers and nodes in a feedforward neural network" stack exchange thread.  

if semi_supervised and co_training:
    model_view = 0
    num_initial_training_runs *= 1
    #num_initial_training_runs*=2  #ran out of time
else:
    model_view = None
if semi_supervised:
    # Create blank model
    model_1_name = "Model A"
    train_instances = slice_instances_2dim(train_labels_and_instances)
    train_labels = slice_labels_2dim(train_labels_and_instances)
    model_1 = create_model(model_1_name,train_instances,num_hidden_layers,neurons_per_layer,view = model_view)
    unlabelled_labels_and_instances, unlabelled_starting_id = original_unlabelled_labels_and_instances, original_unlabelled_starting_id#copy.deepcopy(original_unlabelled_labels_and_instances),copy.deepcopy(original_unlabelled_starting_id)
    unlabelled_instances = slice_unlabelled_instances_2dim(unlabelled_labels_and_instances)
    
# semi-supervised learning - aggregation of new data from unlabelled data
accuracy_tester_list = []
num_instances_added_list = []
if semi_supervised:
    if co_training:
        model_1.view += 1
        if model_1.view != 1 and model_1.view != 2:
            if model_1.view != 3:
                print("Warning: unexpected model view:",model_1.view, "Setting to 1.")
            model_1.view = 1
    for run in range(num_initial_training_runs):
        # choose model
        chosen_model = copy.deepcopy(model_1)
        print("Starting semi-supervised run",run,"/",num_initial_training_runs,"using",chosen_model.name)
        # Train model on working dataset, returns the model plus the accuracy_tester object
        #TODO dfname isnt actually used for anything other than printing
        chosen_model,accuracy_tester = train_model(dfname,chosen_model,train_instances,train_labels,learning_shape,min_rate,max_rate,num_epochs,interval,vectorised_instances=True)

        
        confident_instances,confident_labels,unlabelled_instances = make_predictions_with_confidence(chosen_model,confidence_fname,unlabelled_instances,unlabelled_starting_id)
        # Remove confident instances/labels from working_dev_fname

        # append confident instances/labels to train_instances and train_labels
        print("Num new instances being added to training dataset:", len(confident_instances))
        num_instances_added_list.append(len(confident_instances))
        if len(confident_instances) > 0:
            print("confident instance:")
            print(confident_instances[0])
            print("confident label:")
            print(confident_labels[0])
        for i, _val in enumerate(confident_instances):
            train_labels_and_instances.append(confident_labels[i] + confident_instances[i])   #TODO this should be vectorised...
        train_instances = slice_instances_2dim(train_labels_and_instances)
        train_labels = slice_labels_2dim(train_labels_and_instances) 
        # Move on if no new data to add
        if len(unlabelled_instances) > 0:
            print("unconfident instance",unlabelled_instances[0])
            print("num unlabelled instances remaining",len(unlabelled_instances))
        else:
            print("empty unconfident instances achieved:",unlabelled_instances)
            continue
        accuracy_tester_list.append(accuracy_tester)
        accuracy_tester.plot(False)
print("Num instances added list",num_instances_added_list)

# Final training run using all data and only toxicity is set as a label! Just outputting the toxicity
num_epochs = 20#5-20-200
num_outputs = 1
interval = 32
train_instances = slice_instances_2dim(train_labels_and_instances)
train_labels = slice_labels_2dim(train_labels_and_instances)
# need to create new model since num outputs and inputs is different, i.e. diff weight shape.
chosen_model = create_model("Final model",train_instances,num_hidden_layers,neurons_per_layer)
chosen_model,final_accuracy_tester = train_model(dfname,chosen_model,train_instances,train_labels,learning_shape,min_rate,max_rate,num_epochs,252,vectorised_instances=True)


#%%
# Output creation code
#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/#/
num_outputs = 1
file_analysed = dev_fname
print("Making predictions...")
dev_labels_and_instances,dev_starting_id = preprocess_test(file_analysed)
dev_instances = slice_instances_2dim(dev_labels_and_instances)
m = chosen_model 
make_final_predictions(m,output_fname,dev_instances,dev_starting_id)
print("Assessing accuracy...")
accuracy_of_model = assess_final_predictions(file_analysed,output_fname)
print("Accuracy of model:",accuracy_of_model)

fname_end = last_layer_activation_name + " acc " + str(accuracy_of_model) + " learning  " + str(max_rate) + "-" +str(min_rate)+ "_" + learning_shape + ", " + str(neurons_per_layer) + " neurons " + str(num_hidden_layers) + " hidden layers " + str(num_epochs) + " repeats " + str(interval) + " int" + " rec " + str(recycle_toxic_data) + " wt " + str(weighted_learning)
if lazy_and_wanna_try_combined_data:
    fname_end += " with combined data"

file_analysed = test_fname
test_labels_and_instances,test_starting_id = preprocess_test(test_fname)
test_instances = slice_instances_2dim(test_labels_and_instances)
make_final_predictions(m,output_fname,test_instances,test_starting_id)
    #OPTIONAL add new valid stuff to working_dataset.csv
    
#Graphing final run accuracies
a_test = final_accuracy_tester # final_accuracy_tester
a_test.plot(True,custom_name_ext = "final_run")
# save data in case it crashes o.o
####save_model(dataset_path,chosen_model)
# Load model (for debug purposes)
####chosen_model = load_model(dataset_path, model_1_name)

#%%
# EXTRA LONG FINAL RUN
num_epochs = 100#5-20-200
num_outputs = 1
interval = 32
train_instances = slice_instances_2dim(train_labels_and_instances)
train_labels = slice_labels_2dim(train_labels_and_instances)
# need to create new model since num outputs and inputs is different, i.e. diff weight shape.
chosen_model_2 = create_model("Final model 100 runs",train_instances,num_hidden_layers,neurons_per_layer)
chosen_model_2,final_accuracy_tester_2 = train_model(dfname,chosen_model,train_instances,train_labels,learning_shape,min_rate,max_rate,num_epochs,interval,vectorised_instances=True)


#%%
print("Num instances added list",num_instances_added_list)

# Final training run using all data and only toxicity is set as a label! Just outputting the toxicity
num_epochs = 5 #5-20-200  OUT OF TIME
num_outputs = 1
interval = 32
train_instances = slice_instances_2dim(train_labels_and_instances)
train_labels = slice_labels_2dim(train_labels_and_instances)
# need to create new model since num outputs and inputs is different, i.e. diff weight shape.
chosen_model = create_model("Final model",train_instances,num_hidden_layers,neurons_per_layer)
chosen_model,final_accuracy_tester = train_model(dfname,chosen_model,train_instances,train_labels,learning_shape,min_rate,max_rate,num_epochs,interval,vectorised_instances=True)
#%%
# Colour gradients
# src: https://bsouthga.dev/posts/color-gradients-with-python
def hex_to_RGB(hex):
  ''' "#FFFFFF" -> [255,255,255] '''
  # Pass 16 to the integer function for change of base
  return [int(hex[i:i+2], 16) for i in range(1,6,2)]


def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])
def color_dict(gradient):
  ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
  return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}


def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)

  return color_dict(RGB_list)

#%%
# Graphing semi-supervised accuracies
accuracy_tester_list[0].plot(False,custom_name_ext = "first")
accuracy_tester_list[9].plot(False,custom_name_ext = "tenth")
accuracy_tester_list[19].plot(False,custom_name_ext = "twentieth")
accuracy_tester_list[29].plot(True,custom_name_ext = "thirtieth")
#%%
#%% Plotting continuous line after 2000 batches (10 list elements)
num_instances_added = [441, 476, 674, 721, 2367, 3733, 5795, 6791, 5209, 3747, 3648, 12491, 5617]
num_instances_added_co = [922, 483, 1155, 1625, 3631, 4937, 4962]
acc_post_2k = [0.5642666666666667, 0.5416, 0.5616, 0.5536666666666666, 0.5442, 0.5586666666666666, 0.5456, 0.5526, 0.5451333333333334, 0.5505333333333333, 0.5465333333333333, 0.5507333333333333, 0.545, 0.5453333333333333, 0.5452666666666667, 0.5494666666666667, 0.5516, 0.5484666666666667]
acc_post_2k += [0.5611333333333334, 0.5751333333333334, 0.5599333333333333, 0.5666666666666667, 0.5581333333333334, 0.57, 0.5680666666666667, 0.5722, 0.579, 0.5922, 0.569, 0.5637333333333333, 0.5834, 0.5786, 0.5737333333333333, 0.5755333333333333, 0.5832, 0.5782]

acc_post_2k += [0.5691333333333334, 0.5672, 0.5693333333333334, 0.5614, 0.5592666666666667, 0.5767333333333333, 0.5780666666666666, 0.5763333333333334, 0.5656666666666667, 0.5652, 0.5736666666666667, 0.5724666666666667, 0.5670666666666667, 0.5712666666666667, 0.5778, 0.5698666666666666, 0.5696, 0.5762]

acc_post_2k += [0.5709333333333333, 0.5834666666666667, 0.5780666666666666, 0.5940666666666666, 0.5966, 0.5994666666666667, 0.5927333333333333, 0.5877333333333333, 0.6017333333333333, 0.6055333333333334, 0.5982, 0.5935333333333334, 0.6037333333333333, 0.6046666666666667, 0.5997333333333333, 0.6004, 0.6031333333333333, 0.6104]

acc_post_2k += [0.5589333333333333, 0.5516, 0.5803333333333334, 0.5594, 0.5894, 0.5708, 0.574, 0.5838666666666666, 0.5838666666666666, 0.5704, 0.5801333333333333, 0.5778666666666666, 0.5892, 0.5863333333333334, 0.5736, 0.5812666666666667, 0.5802, 0.5806]

acc_post_2k += [0.6496666666666666, 0.6647333333333333, 0.6587333333333333, 0.6732666666666667, 0.6488666666666667, 0.6541333333333333, 0.6713333333333333, 0.6713333333333333, 0.6528, 0.6599333333333334, 0.6631333333333334, 0.6666, 0.6614, 0.6657333333333333, 0.6581333333333333, 0.6554, 0.6641333333333334, 0.6647333333333333, 0.6638]

acc_post_2k+=[0.7306666666666667, 0.7074666666666667, 0.7214666666666667, 0.7134, 0.7106, 0.7077333333333333, 0.7047333333333333, 0.709, 0.7099333333333333, 0.708, 0.7093333333333334, 0.7096, 0.7098666666666666, 0.7048, 0.7086666666666667, 0.7054666666666667, 0.7084, 0.7096666666666667, 0.7094]

acc_post_2k += [0.7534666666666666, 0.749, 0.7562666666666666, 0.7499333333333333, 0.7446666666666667, 0.7503333333333333, 0.7388, 0.7370666666666666, 0.7526, 0.7508666666666667, 0.7436666666666667, 0.7473333333333333, 0.7420666666666667, 0.7502666666666666, 0.7421333333333333, 0.7471333333333333, 0.7458, 0.7484666666666666, 0.7476666666666667, 0.7464, 0.7484]

acc_post_2k += [0.7417333333333334, 0.7602666666666666, 0.7540666666666667, 0.7526, 0.7625333333333333, 0.7602666666666666, 0.7576, 0.7528, 0.7594, 0.7624666666666666, 0.7576666666666667, 0.7591333333333333, 0.7613333333333333, 0.7606666666666667, 0.7596, 0.7584666666666666, 0.7582666666666666, 0.7566, 0.7596666666666667, 0.7573333333333333, 0.7591333333333333, 0.7570666666666667]

acc_post_2k += [0.7721333333333333, 0.7694666666666666, 0.7718666666666667, 0.7704, 0.7692666666666667, 0.7645333333333333, 0.7673333333333333, 0.7675333333333333, 0.7692666666666667, 0.7681333333333333, 0.7665333333333333, 0.7654666666666666, 0.7671333333333333, 0.7709333333333334, 0.7719333333333334, 0.7724666666666666, 0.7696, 0.7682, 0.7709333333333334, 0.768, 0.7704, 0.7675333333333333, 0.7706666666666667]

acc_post_2k +=[0.7726666666666666, 0.7686, 0.7662, 0.7692666666666667, 0.7772, 0.7716666666666666, 0.7744, 0.7738666666666667, 0.7744, 0.7740666666666667, 0.7747333333333334, 0.7755333333333333, 0.7708666666666667, 0.7706666666666667, 0.7761333333333333, 0.7726, 0.7729333333333334, 0.7747333333333334, 0.7752666666666667, 0.7744, 0.7741333333333333, 0.7744, 0.7766, 0.7738666666666667]

acc_post_2k += [0.7648, 0.7720666666666667, 0.7687333333333334, 0.7693333333333333, 0.7724, 0.7709333333333334, 0.7704666666666666, 0.7702666666666667, 0.7686, 0.7703333333333333, 0.7701333333333333, 0.7666, 0.7722666666666667, 0.7676666666666667, 0.7734666666666666, 0.7703333333333333, 0.7705333333333333, 0.769, 0.7727333333333334, 0.7700666666666667, 0.7705333333333333, 0.7706, 0.7712666666666667, 0.7710666666666667]

my_lists = []
my_lists.append([0.5506, 0.43, 0.41546666666666665, 0.4206666666666667, 0.44866666666666666, 0.46586666666666665, 0.4992666666666667, 0.5520666666666667, 0.5594, 0.5745333333333333, 0.559, 0.5844, 0.5875333333333334, 0.5783333333333334, 0.5854666666666667, 0.5825333333333333, 0.5868, 0.5973333333333334, 0.5874666666666667, 0.5932666666666667, 0.5815333333333333, 0.5915333333333334, 0.5954666666666667, 0.592, 0.589, 0.5916, 0.5938, 0.5928]
)
my_lists.append([0.5506, 0.4130666666666667, 0.4284, 0.47173333333333334, 0.43906666666666666, 0.473, 0.46973333333333334, 0.47326666666666667, 0.46786666666666665, 0.48746666666666666, 0.5381333333333334, 0.5657333333333333, 0.5825333333333333, 0.5785333333333333, 0.5896, 0.5868666666666666, 0.5909333333333333, 0.5928666666666667, 0.5905333333333334, 0.5930666666666666, 0.598, 0.596, 0.5956666666666667, 0.5969333333333333, 0.596, 0.5945333333333334, 0.5969333333333333, 0.596]
)
my_lists.append([0.5506, 0.4298666666666667, 0.45466666666666666, 0.44993333333333335, 0.45693333333333336, 0.4318, 0.5763333333333334, 0.5983333333333334, 0.6096666666666667, 0.6296, 0.6042666666666666, 0.6179333333333333, 0.6181333333333333, 0.6176666666666667, 0.6249333333333333, 0.6164666666666667, 0.6319333333333333, 0.6191333333333333, 0.6218666666666667, 0.6245333333333334, 0.6176666666666667, 0.6257333333333334, 0.6221333333333333, 0.6253333333333333, 0.6193333333333333, 0.6183333333333333, 0.6216666666666667, 0.6258]
)
my_lists.append([0.5506, 0.44106666666666666, 0.45693333333333336, 0.43133333333333335, 0.45493333333333336, 0.486, 0.5466, 0.6193333333333333, 0.6254666666666666, 0.6380666666666667, 0.6224, 0.616, 0.6137333333333334, 0.6440666666666667, 0.6406, 0.6474, 0.6449333333333334, 0.6266666666666667, 0.6426, 0.6398666666666667, 0.6397333333333334, 0.6349333333333333, 0.6344, 0.6399333333333334, 0.6306, 0.6355333333333333, 0.636, 0.6376]
)
my_lists.append([0.5506, 0.45206666666666667, 0.4582, 0.4506, 0.4739333333333333, 0.5908, 0.649, 0.6404, 0.6508, 0.6693333333333333, 0.6445333333333333, 0.6469333333333334, 0.6307333333333334, 0.6210666666666667, 0.6228666666666667, 0.6771333333333334, 0.6404, 0.6514666666666666, 0.6524666666666666, 0.6549333333333334, 0.6496, 0.6407333333333334, 0.6439333333333334, 0.6486, 0.644, 0.6481333333333333, 0.6446666666666667, 0.6412, 0.6422666666666667]
)
my_lists.append([0.5506, 0.44206666666666666, 0.41873333333333335, 0.4295333333333333, 0.5748, 0.6096, 0.6913333333333334, 0.6940666666666667, 0.6465333333333333, 0.672, 0.6444666666666666, 0.661, 0.6159333333333333, 0.6327333333333334, 0.6612666666666667, 0.6629333333333334, 0.6559333333333334, 0.6564, 0.6649333333333334, 0.6622666666666667, 0.6590666666666667, 0.6634, 0.6777333333333333, 0.667, 0.6669333333333334, 0.6691333333333334, 0.668, 0.6652666666666667, 0.6689333333333334]
)
my_lists.append([0.5506, 0.4616, 0.46353333333333335, 0.43393333333333334, 0.6333333333333333, 0.6616666666666666, 0.6781333333333334, 0.6850666666666667, 0.6958, 0.6859333333333333, 0.6970666666666666, 0.6544, 0.6986, 0.6874, 0.6975333333333333, 0.6924, 0.6848, 0.6851333333333334, 0.6978, 0.6918666666666666, 0.6874666666666667, 0.6917333333333333, 0.6886, 0.6823333333333333, 0.6884, 0.6842, 0.6866, 0.6888666666666666, 0.6816666666666666, 0.6891333333333334]
)
co_acc_post_2k = []
for list in my_lists:
    l = list[10:]
    for a in range(len(l)):
        l[a] = 1 - l[a]
    co_acc_post_2k += l

c = linear_gradient("#7FFFD4","#097969",n=3)

# NO LONGER NUM INSTANCES ADDED BTW, but batch size
print(num_instances_added)
for a in range(len(num_instances_added)):
    num_instances_added[0] = 2000
    if a != 0:
        num_instances_added[a] = (num_instances_added[a-1] + num_instances_added[a] + 140000 - 256*2000)/256
        print(num_instances_added[a])
        if a < 7:
            num_instances_added_co[a] = (num_instances_added_co[a-1] + 140000 - 256*2000)/256

#plt.axvline(10)
for i in range(len(acc_post_2k)):
    acc_post_2k[i] = 1 - acc_post_2k[i]
for b in range(len(num_instances_added)):
    num_instances_added[b] = num_instances_added[b]/1000
for b in range(len(num_instances_added_co)):
    num_instances_added_co[b] =  num_instances_added_co[b]/1000
plt.vlines(num_instances_added, 0, 0.5,linestyles = 'dashed', colors=c['hex'][0])
plt.vlines(num_instances_added_co, 0, 0.5,linestyles = 'dashed',colors = c['hex'][1])
plt.title("Error rate evolution")
plt.xlabel("Batch no. - num early batches (1e3)")
plt.ylabel("Test error (%)")

# self-trained
x = []
for i in range(len(acc_post_2k)):
        x.append(2 + i/5)  # Start from batch 1000
x2 = []
for i in range(len(co_acc_post_2k)):
        x2.append(2 + i/5)  # Start from batch 1000
plt.plot(x,acc_post_2k,color = c['hex'][0],label="self-trained")
plt.plot(x2,co_acc_post_2k,color = c['hex'][1],label="co-trained")
plt.legend()
# co-trained
plt.savefig('Accs past 2000' +'.png') 

plt.show()





# %%
print(accuracy_tester.accuracies)
# %%
gi = [[3,2,1],[2,2,2]]
print(gi[0][0])
# %%
output_array = np.array([[0],[1],[2]])
gi = output_array[:,0]
print(gi)
# %%
"get 3 from list"
a = 1
k = [1,2,3,4,5,6]
k_batch = k[a:a+3]
print(k_batch)

#%%
'''
b = np.zeros((5,4,2))     #5 outputs per instance, 4 inputs per instance,2 instances
a = np.array([[10,11,12,13],[14,15,16,17]])       # instance 1 input: [10,11,12,13] instance 2 input: [14,15,16,17]

a_trans = np.transpose(a)
print(b)

'''

#%%
dz0_dw_matrix = np.zeros(len(x),(len(z[0]),len(x[0])))
dz0_dw_matrix[:,:] = x[:,None] + dz0_dw_matrix[:]

#dz_dz0 in [instance_index,]
dc_dz0 = np.multiply(dz_dz0,dc_dz)


#%%
dz0_dw_matrix = np.zeros((len(x),len(z[0]),len(x[0])))   #num instances [num_instances], length of output of one instance [rows], length of input of one instance [columns]
dz0_dw_matrix[:] = copy.deepcopy(x[:])                      # replace every instance_index with 

b = b.transpose(0,2,1)

# %%
# dz0_dw = x testing
b = np.zeros((2,5,4))     #2 instances, 5 outputs per instance, 4 inputs per instance
a = np.array([[10,11,12,13],[14,15,16,17]])       # instance 1 input: [10,11,12,13] instance 2 input: [14,15,16,17]
                        # Goal: want to replace every element corresponding to z for the same instance with the same instance input
#print(b)
b = a[:, None] + b[:]
print(b)
b = b.transpose(0,2,1)

print(b)
#for x,y in np.nditer([a,b]):
    #x = y

#print(b)

#%%
#dc_dz0 testing

c = np.zeros((2,4,5))     #2 instances, 4 inputs per instance, 5 outputs per instance, 
z = np.array([[0.1,0.2,0.3,0.4,0.5],[0.6,0.7,0.8,0.9,1.0]])            # instance 1 cost of output derivative: [0.1,0.2,0.3,0.4,0.5], instance 2 cost: [0.6,...,1.0]
c[:,:] = z[:, None] + c[:]
print(c)

#%%
#dc_dw_tensor testing
dc_dw_tensor = np.multiply(b,c)
#print(dc_dw_tensor)
tens_avg = np.average(dc_dw_tensor,axis = 0)
print(tens_avg)
#print(np.array2string(tens_avg,separator = ","))

#%%
dc_dz0 = np.multiply(dz_dz0,dc_dz)    # this is rows of dc_dz0 corresponding to each instance [instance X outputlength]
#NOT DONE, STILL TODO D:
dc_dz0_matrix = np.zeros(len(x),(len(x[0]),len(z[0])))                                                                        
dc_dz0_matrix[:,:] = dc_dz0[:,None] + dc_dz0_matrix[:]   

#%%

#dz_dz0 = self.activation.dz_dz0_vectorised(z0)
dc_dz0 = np.multiply(dz_dz0,dc_dz) 
dc_dz0_matrix = np.zeros((len(x),len(z)))                                                                        
dc_dz0_matrix[:] = copy.deepcopy(dc_dz0)     # this gives us a matrix of rows equal to the vector dc_dz0      #WHY DOES THIS NEED TO BE cOPIED FOR LATER???             
dc_dw_tensor = np.multiply(dz0_dw_matrix,dc_dz0_matrix)#np.multiply(dz0_dw_matrix,dc_dz0_matrix)
#Transpose because of the goofy way I set up everything
dc_dw_tensor = np.matrix.transpose(dc_dw_tensor)
self.weights_change -= learning_rate*dc_dw_tensor
#%%
dz0_dw_matrix = np.zeros((len(x),len(z[0]),len(x[0])))   #num instances [num_instances], length of output of one instance [rows], length of input of one instance [columns]
dz0_dw_matrix[:] = copy.deepcopy(x[:])                      # replace every instance_index with 


#not vectorised

dz0_dw_matrix = np.zeros((len(z),len(x)))
dz0_dw_matrix[:] = copy.deepcopy(x)
dz0_dw_matrix = np.matrix.transpose(dz0_dw_matrix) # this gives us a matrix of columns equal to the vector x


 #%%
l_array = np.array([[2,-4,0.5],[3,0.1,1]])
print(l_array.clip(-1,1))
# %%
a = np.array([3,2,1])
print(a)
print(a.reshape(len(a),1))
# %%

a = [[3,2],[4,5]]
print(np.average(a,axis = 1))
# %%
a = np.array([[2,1,3,2],[2,1,2,4]])

b = a[:,1:]
print(b)


# %%
a = np.array([[2,3],[2,5]])
mask = (a<5)*(a >= 2)
a[mask] = 1
print(a)


# %%
# %%
import numpy as np
min_confidence = 0.8
instances_array = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
output_array = np.array([[0.4,0.9,1],[1,1,1],[1,1,1]])
confidence_array = np.average(np.abs(output_array),axis=1)
print(confidence_array)
idx = (confidence_array<=999)*(confidence_array >= (1 + min_confidence)/2) #(confidence_array<=min_confidence)*(confidence_array >= 1 - min_confidence)
print(idx)
confident_indices = np.where(idx)
confident_instances = np.take(instances_array,confident_indices,axis=0)[0]
confident_instances = confident_instances.tolist()
print(confident_instances)

# %%
#yo
k = np.array([[0.1,0.4,0.6,2],[0.1,0.5,0.6,2],[0.1,0.6,0.6,2.1]])   #for each instance, contribute to bias change, 4 biases

average_k = np.average(k,axis=0)
print(average_k)


#%% GOOD
target_output__ = np.array([[1,0,1,1],[1,1,0,0],[1,1,1,1],[1,1,0,1],[1,0,1,1],[1,1,0,0],[1,1,1,1],[1,1,0,1]],dtype='float') 
my_weights = target_output__ 
#print(target_output__)
tot_count = len(target_output__)
prob_true = np.sum(my_weights,axis=0)/tot_count

prob_true_matrix = np.zeros((my_weights.shape))
print(my_weights)
prob_true_matrix[:] = prob_true

print(prob_true_matrix)

mask1 = (my_weights == 0)
mask2 = (my_weights == 1) 
total_prob = ((1-prob_true_matrix)**2 + prob_true_matrix**2)*tot_count

my_weights[mask1] = ((prob_true_matrix)/total_prob*total_prob*2)[mask1]
my_weights[mask2] = ((1-prob_true_matrix)/total_prob*total_prob*2)[mask2]
print(my_weights)
# %%
#TESASDAD
average_w = np.average(my_weights,axis=0)
average_w_matrix = np.zeros((my_weights.shape))
# %%

average_w_matrix[:] = average_w

#%%
#%%
target_output__ = np.array([[0,0,1,1],[0,1,0,0],[0,1,1,1]],dtype='float') 
my_weights = target_output__ 
tot_count = len(target_output__)
prob_true = np.sum(my_weights,axis=0)/tot_count
prob_true[prob_true == 0] = 1

prob_true_matrix = np.zeros((my_weights.shape))
prob_true_matrix[:] = prob_true

mask1 = (my_weights == 0)
mask2 = (my_weights == 1) 
print(mask1)
print(mask2)

print(my_weights)
print("prob true:")
print(prob_true_matrix)
print()

total_prob = ((1-prob_true_matrix)**2 + prob_true_matrix**2)*tot_count
print("total prob")
print(total_prob)
print()
my_weights[mask1] = ((prob_true_matrix)/total_prob*total_prob*tot_count)[mask1]
my_weights[mask2] = ((1-prob_true_matrix)/total_prob*total_prob*tot_count)[mask2]
print(my_weights)
# %%
#test
m1 = np.array([0,1,1,3])
m2 = np.array([0,0,1,1])
data = np.array([10,20,30,40])
mask = (m1 == 1)|(m2 == 1)
print(data[mask]) #should return 30
# %%
#e
print("test")
# %%
# Determination of optimal hyperparams
# bloody vscode kernel crashed

#Interval of 5 runs starting from first run. The acc list is the error rates, not the accuracies sorry.
err_rate_list = [0,0,0,0,0,0,0]
batch_count_list = [0,0,0,0,0,0,0]

batch_count_list[0] = 5460
err_rate_list[0] = [0.18993333333333334, 0.5206, 0.5434, 0.5267333333333334, 0.5414666666666667, 0.5689333333333333, 0.5650666666666667, 0.5784666666666667, 0.5636, 0.5853333333333334, 0.58, 0.5947333333333333, 0.5887333333333333, 0.5846666666666667, 0.5678666666666666, 0.5828666666666666, 0.5774, 0.5848666666666666, 0.5783333333333334, 0.5837333333333333, 0.5752666666666667, 0.574, 0.5825333333333333, 0.5848666666666666, 0.5824, 0.5784, 0.5842, 0.5806666666666667]
batch_count_list[1] = 5640
err_rate_list[1] = [0.18993333333333334, 0.5239333333333334, 0.5705333333333333, 0.5505333333333333, 0.5898, 0.6654666666666667, 0.6600666666666667, 0.6502666666666667, 0.6509333333333334, 0.6576, 0.6447333333333334, 0.6481333333333333, 0.6632666666666667, 0.6532, 0.6573333333333333, 0.6426, 0.6523333333333333, 0.6405333333333333, 0.6450666666666667, 0.6406, 0.6437333333333334, 0.6398, 0.6462, 0.6507333333333334, 0.6458666666666667, 0.6513333333333333, 0.6400666666666667, 0.6404666666666666, 0.6472]
batch_count_list[2] = 6630
err_rate_list[2] = [0.18993333333333334, 0.5008, 0.53, 0.2536, 0.6790666666666667, 0.7067333333333333, 0.7358666666666667, 0.7499333333333333, 0.7639333333333334, 0.7654, 0.7714666666666666, 0.767, 0.7680666666666667, 0.7746, 0.7786, 0.7724, 0.7693333333333333, 0.7704, 0.7784, 0.774, 0.7795333333333333, 0.7771333333333333, 0.7765333333333333, 0.7761333333333333, 0.7806666666666666, 0.7777333333333334, 0.7786, 0.7788666666666667, 0.7788666666666667, 0.7778666666666667, 0.778, 0.7798, 0.7798, 0.7772]
batch_count_list[3] = 7630
err_rate_list[3] = [0.18993333333333334, 0.5353333333333333, 0.5268666666666667, 0.26826666666666665, 0.6282, 0.7439333333333333, 0.774, 0.7482, 0.746, 0.7333333333333333, 0.749, 0.7626666666666667, 0.7629333333333334, 0.762, 0.7729333333333334, 0.7766, 0.7752, 0.7774, 0.7786666666666666, 0.7784666666666666, 0.7782666666666667, 0.7808, 0.7798666666666667, 0.7813333333333333, 0.7807333333333333, 0.7812, 0.7821333333333333, 0.7819333333333334, 0.7824666666666666, 0.7824, 0.7829333333333334, 0.7823333333333333, 0.7827333333333333, 0.7831333333333333, 0.7827333333333333, 0.7826, 0.7826666666666666, 0.7824, 0.7831333333333333]
batch_count_list[4] = 8310
err_rate_list[4] = [0.18993333333333334, 0.5504, 0.5629333333333333, 0.2899333333333333, 0.2816666666666667, 0.6990666666666666, 0.7548, 0.766, 0.7738666666666667, 0.7634, 0.7636666666666667, 0.7596666666666667, 0.7493333333333333, 0.7503333333333333, 0.7369333333333333, 0.7572, 0.7542666666666666, 0.7526666666666667, 0.7554666666666666, 0.7586, 0.7510666666666667, 0.7506, 0.7553333333333333, 0.7554, 0.757, 0.7551333333333333, 0.7578666666666667, 0.7572, 0.7592, 0.7574666666666666, 0.7598666666666667, 0.7582666666666666, 0.7614, 0.7608666666666667, 0.7610666666666667, 0.7584666666666666, 0.7621333333333333, 0.7605333333333333, 0.76, 0.7618, 0.7614666666666666, 0.7612666666666666]
batch_count_list[5] = 9210
err_rate_list[5] =[0.18993333333333334, 0.5645333333333333, 0.492, 0.2581333333333333, 0.37933333333333336, 0.7549333333333333, 0.7106666666666667, 0.7428, 0.7659333333333334, 0.7746666666666666, 0.7726666666666666, 0.7673333333333333, 0.7704, 0.7726, 0.7727333333333334, 0.7744666666666666, 0.7744666666666666, 0.7698666666666667, 0.773, 0.7736, 0.7719333333333334, 0.7711333333333333, 0.7723333333333333, 0.7737333333333334, 0.7700666666666667, 0.7725333333333333, 0.7702, 0.7726666666666666, 0.7710666666666667, 0.7716666666666666, 0.7729333333333334, 0.7714, 0.7724, 0.7708666666666667, 0.7717333333333334, 0.772, 0.7722666666666667, 0.7718666666666667, 0.7698666666666667, 0.7722666666666667, 0.7716666666666666, 0.7718, 0.7716666666666666, 0.772, 0.7714666666666666, 0.7718666666666667, 0.7716666666666666]
batch_count_list[6] = 9470
err_rate_list[6] = [0.18993333333333334, 0.5638, 0.49146666666666666, 0.23686666666666667, 0.23733333333333334, 0.7216666666666667, 0.7540666666666667, 0.7644666666666666, 0.7678, 0.7339333333333333, 0.7733333333333333, 0.7691333333333333, 0.7736666666666666, 0.7742, 0.7713333333333333, 0.7720666666666667, 0.7712, 0.7702666666666667, 0.7731333333333333, 0.7724666666666666, 0.7698666666666667, 0.7728, 0.7718666666666667, 0.7724, 0.7740666666666667, 0.7728, 0.7730666666666667, 0.7741333333333333, 0.772, 0.772, 0.7713333333333333, 0.7720666666666667, 0.7723333333333333, 0.7723333333333333, 0.7728666666666667, 0.7726666666666666, 0.7728, 0.772, 0.7719333333333334, 0.7719333333333334, 0.7719333333333334, 0.7724, 0.7724, 0.7723333333333333, 0.7723333333333333, 0.7725333333333333, 0.7728, 0.7724]

c = ["red","orange","gold","green","blue"]
c = linear_gradient("#7FFFD4","#097969")
c = linear_gradient("#EE4B2B","#097969",n=6)
c = linear_gradient("#097969","#00FFFF",n=6)
c = linear_gradient("#DE3163","#FFBF00",n=6)
#00FFFF

import math
batches_per_interval = 200  # This is just what it was when I ran it.
start_run = 2
start_batch = 8
for k in range(len(err_rate_list)):
    if k > start_run - 1:
        # Get error rate
        y = err_rate_list[k]
        for i in range(len(y)):
            if i > start_batch - 1:
                y[i] = (1 - y[i])*100
        for i in range(start_batch ):
            y.pop(0)
        x = []
        for i,val in enumerate(range(0,math.ceil(batch_count_list[k]/batches_per_interval))):
            if i > start_batch - 1:
                x.append(val/50)
        plt.title("Error rate evolution")
        plt.xlabel("Batch no. (1e3)")
        plt.ylabel("Test error (%)")
        plt.plot(x,y,color = c['hex'][k - start_run],label="Run " + str(k*5))
plt.legend(loc="upper right")
plt.savefig('accuracy_plot_' + 'recent_graph' +'.png') 
plt.show()

#final run, accuracies at END of cycle:
start_acc = 0.5640666666666667
cycle_0_acc = 0.5709333333333333
cycle_1_acc = 0.8092
cycle_2_acc = 0.8098
cycle_3_acc = 0.8121
cycle_4_acc = 0.8134
cycle_9_acc = 0.8181
cycle_17_acc = 0.8217
cycle_18_acc = 0.8224
cycle_19_acc = 0.82
cycle_20_acc = 0.8225
cycle_21_acc = 0.8215
cycle_22_acc = 0.8207
cycle_23_acc = 0.8229
cycle_24_acc = 0.8229
cycle_24_acc = 0.8229
cycle_25_acc = 0.8227
cycle_26_acc = 0.8229
# %%
del list
# %%
