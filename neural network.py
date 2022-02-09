import pandas
import numpy as np

def logistic(x):
    return 1.0/(1 + np.exp(-x))

def logistic_deriv(x):
    return logistic(x) * (1 - logistic(x))

LR = 1

I_dim = 2
H_dim = 10

epoch_count = 100

#np.random.seed(1)
weights_ItoH = np.random.uniform(-1, 1, (I_dim, H_dim))
weights_HtoO = np.random.uniform(-1, 1, H_dim)

preActivation_H = np.zeros(H_dim)
postActivation_H = np.zeros(H_dim)

training_data = pandas.read_excel('trainingDataDivide.xlsx')
target_output = training_data.output
training_data = training_data.drop(['output'], axis=1)
training_data = np.asarray(training_data)
training_count = len(training_data[:,0])

validation_data = pandas.read_excel('validationDataDivide.xlsx')
validation_output = validation_data.output
validation_data = validation_data.drop(['output'], axis=1)
validation_data = np.asarray(validation_data)
validation_count = len(validation_data[:,0])

#####################
#training
#####################
for epoch in range(epoch_count):
    for sample in range(training_count):
        for node in range(H_dim):
            preActivation_H[node] = np.dot(training_data[sample,:], weights_ItoH[:, node])
            postActivation_H[node] = logistic(preActivation_H[node])
            
        preActivation_O = np.dot(postActivation_H, weights_HtoO)
        postActivation_O = logistic(preActivation_O)
        
        FE = postActivation_O - target_output[sample]
        
        for H_node in range(H_dim):
            S_error = FE * logistic_deriv(preActivation_O)
            gradient_HtoO = S_error * postActivation_H[H_node]
                       
            for I_node in range(I_dim):
                input_value = training_data[sample, I_node]
                gradient_ItoH = S_error * weights_HtoO[H_node] * logistic_deriv(preActivation_H[H_node]) * input_value
                
                weights_ItoH[I_node, H_node] -= LR * gradient_ItoH
                
            weights_HtoO[H_node] -= LR * gradient_HtoO

#####################
#validation
#####################            
#correct_classification_count = 0
errorA = np.zeros(validation_count)
errorP = np.zeros(validation_count)
for sample in range(validation_count):
    for node in range(H_dim):
        preActivation_H[node] = np.dot(validation_data[sample,:], weights_ItoH[:, node])
        postActivation_H[node] = logistic(preActivation_H[node])
            
    preActivation_O = np.dot(postActivation_H, weights_HtoO)
    postActivation_O = logistic(preActivation_O)
        
    #if postActivation_O > 0.5:
    #    output = 1
    #else:
    #    output = 0     
    
    errorA[sample] = postActivation_O - validation_output[sample]
    errorP[sample] = abs(errorA[sample])/validation_output[sample]
    print("Output: " + str(postActivation_O))
    print("Expected: " + str(validation_output[sample]))
    print("Absolute Error: " + "{:.2f}".format(errorA[sample]))
    print("Percentage Error: " + "{:.2f}".format(errorP[sample]*100) + "%")
    print("")
    #if output == validation_output[sample]:
    #    correct_classification_count += 1

print("Average Absolute Error: " + str(np.mean(errorA)))
print("Average Percentage Error: " + str(np.mean(errorP)*100) + "%")
#print('Percentage of correct classifications:')
#print(correct_classification_count*100/validation_count)