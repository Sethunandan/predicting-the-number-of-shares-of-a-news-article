import pandas as pd
import numpy as np



np.random.seed(42)
class Scaler():
    # hint: https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    def __init__(self,file):
        self.file = file
        
    def __call__(self):
        
        for feature in self.file.columns:
            xmean = np.mean(self.file[feature],axis=0)
            xstd = np.std(self.file[feature],axis=0)
            self.file[feature] = (self.file[feature]-xmean)/(xstd)
            #xmin = self.file[feature].min(0)
            #xmax = self.file[feature].max(0)
            #self.file[feature] = (self.file[feature]-xmin)/(xmax-xmin)
        return self.file
        raise NotImplementedError

def get_features(csv_path,is_train=False,scaler=None):
    '''
    Description:
    read input feature columns from csv file
    manipulate feature columns, create basis functions, do feature scaling etc.
    return a feature matrix (numpy array) of shape m x n 
    m is number of examples, n is number of features
    return value: numpy array
    '''
    dataset = pd.read_csv(csv_path)
    
    
    
    if is_train:
        
        dataset = dataset.drop([' shares'],axis=1)
        dataset = dataset.drop([' LDA_00',' global_sentiment_polarity',' global_rate_negative_words',' avg_negative_polarity',' n_unique_tokens',' n_non_stop_unique_tokens'],axis=1)
    else:
        dataset = dataset.drop(['LDA_00','global_sentiment_polarity','global_rate_negative_words','avg_negative_polarity','n_unique_tokens','n_non_stop_unique_tokens'],axis=1)
    
    
  
   
    
    
    scaler = Scaler(dataset)
    dataset = scaler.__call__()  
    dataset[' intercept'] = 1      #incercept column
    
    
    
    return np.array(dataset)
    
    '''
    Arguments:
    csv_path: path to csv file
    is_train: True if using training data (optional)
    scaler: a class object for doing feature scaling (optional)
    '''

    '''
    help:
    useful links: 
        * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
        * https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/
    '''

    raise NotImplementedError
    
def get_targets(csv_path):
    '''
    Description:
    read target outputs from the csv file
    return a numpy array of shape m x 1
    m is number of examples
    '''
    data = pd.read_csv(csv_path)
    return np.array(data[[' shares']])
    raise NotImplementedError
    
def analytical_solution(feature_matrix, targets, C=0.0):
    '''
    Description:
    implement analytical solution to obtain weights
    as described in lecture 5d
    return value: numpy array
    '''
    n= feature_matrix.shape[0]
    x = feature_matrix
    y = targets
    w = x.T@x 
    wshape,qshape = w.shape
    w = w +C*np.eye(wshape)*n
    w = np.linalg.inv(w)@x.T
    return w@y
    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    '''

    raise NotImplementedError

def get_predictions(feature_matrix, weights):
    '''
    description
    return predictions given feature matrix and weights
    return value: numpy array
    '''
    return feature_matrix@weights
    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    '''

    raise NotImplementedError
    

def mse_loss(feature_matrix, weights, targets):
    '''
    Description:
    Implement mean squared error loss function
    return value: float (scalar)
    '''
    n= feature_matrix.shape[0]
    predictions = get_predictions(feature_matrix, weights)
    loss = predictions - targets
    sq_loss = np.square(loss)
    return np.sum(sq_loss,axis = 0)/n
    
    
    
    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    '''
    raise NotImplementedError

def l2_regularizer(weights):
    '''
    Description:
    Implement l2 regularizer
    return value: float (scalar)
    '''
    sq_weights = np.square(weights)

    return np.sum(sq_weights,axis = 0)

    '''
    Arguments
    weights: numpy array of shape n x 1
    '''
    raise NotImplementedError
    
def loss_fn(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute the loss function: mse_loss + C * l2_regularizer
    
    '''
    print(mse_loss(feature_matrix, weights, targets) + C*l2_regularizer(weights))
    return mse_loss(feature_matrix, weights, targets) + C*l2_regularizer(weights)
    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: float (scalar)
    '''

    raise NotImplementedError

    
def compute_gradients(feature_matrix, weights, targets, C=0.0):
    '''
    Description:
    compute gradient of weights w.r.t. the loss_fn function implemented above

    '''
    n= feature_matrix.shape[0]
    l = targets - get_predictions(feature_matrix, weights)
    grad = -2*(feature_matrix.T@l)/n + 2*C*weights
    
    return grad
    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    weights: numpy array of shape n x 1
    targets: numpy array of shape m x 1
    C: weight for regularization penalty
    return value: numpy array
    '''
    raise NotImplementedError
    
def sample_random_batch(feature_matrix, targets, batch_size):
    '''
    Description
    Batching -- Randomly sample batch_size number of elements from feature_matrix and targets
    return a tuple: (sampled_feature_matrix, sampled_targets)
    sampled_feature_matrix: numpy array of shape batch_size x n
    sampled_targets: numpy array of shape batch_size x 1
    '''
    index = list(np.random.randint(23786, size=(1, batch_size)))
    
    
    return (train_features[index,:][0],train_targets[index,:][0])
    

    
    
    '''
    Arguments:
    feature_matrix: numpy array of shape m x n
    targets: numpy array of shape m x 1
    batch_size: int
    '''    
    raise NotImplementedError
    
    
    
def initialize_weights(n):
    '''
    Description:
    initialize weights to some initial values
    return value: numpy array of shape n x 1
    '''
    
    
    return np.zeros((n,1))
    '''
    Arguments
    n: int
    '''
    raise NotImplementedError

def update_weights(weights, gradients, lr):
    '''
    Description:
    update weights using gradient descent
    retuen value: numpy matrix of shape nx1
    
    '''

    return weights-(lr*gradients)
    '''
    Arguments:
    # weights: numpy matrix of shape nx1
    # gradients: numpy matrix of shape nx1
    # lr: learning rate
    '''    

    raise NotImplementedError


    # allowed to modify argument list as per your need
    # return True or False
    
    
def do_gradient_descent(train_feature_matrix,  
                        train_targets, 
                        dev_feature_matrix,
                        dev_targets,
                        lr=1.0,
                        C=0.0,
                        batch_size=32,
                        max_steps=10000,
                        eval_steps=10):
    '''
    feel free to significantly modify the body of this function as per your needs.
    ** However **, you ought to make use of compute_gradients and update_weights function defined above
    return your best possible estimate of LR weights
    a sample code is as follows -- 
    '''
    weights = initialize_weights(train_feature_matrix.shape[1])
    
    dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
    train_loss = mse_loss(train_feature_matrix, weights, train_targets)
    prev_dev_loss = 0
    prev_train_loss = 0
    print("step {} \t dev loss: {} \t train loss: {}".format(0,dev_loss,train_loss))
    for step in range(1,max_steps+1):
        
        #sample a batch of features and gradients
        features,targets = sample_random_batch(train_feature_matrix,train_targets,batch_size)
        
        
        #compute gradients
        gradients = compute_gradients(features, weights, targets, C)
        
        #update weights
        weights = update_weights(weights, gradients, lr)
        
        dev_loss = mse_loss(dev_feature_matrix, weights, dev_targets)
        train_loss = mse_loss(train_feature_matrix, weights, train_targets)

        
        if step%eval_steps == 0:
            
            print("step {} \t dev loss: {} \t train loss: {}".format(step,dev_loss,train_loss))

        '''
        implement early stopping etc. to improve performance.
        
        '''

        if abs(prev_train_loss - train_loss) < 1e-8:
            break
        prev_train_loss = train_loss
    return weights

def do_evaluation(feature_matrix, targets, weights):
    # your predictions will be evaluated based on mean squared error 
    predictions = get_predictions(feature_matrix, weights)
    loss =  mse_loss(feature_matrix, weights, targets)
    return loss

if __name__ == '__main__':
    scaler = Scaler('/kaggle/input/programming-assignment-1/train.csv')
    train_features, train_targets = get_features('/kaggle/input/programming-assignment-1/train.csv',True,scaler), get_targets('/kaggle/input/programming-assignment-1/train.csv')
    dev_features, dev_targets = get_features('/kaggle/input/programming-assignment-1/dev.csv',True,scaler), get_targets('/kaggle/input/programming-assignment-1/dev.csv')
    test_features              = get_features('/kaggle/input/programming-assignment-1/test.csv',False,scaler)

    a_solution = analytical_solution(train_features, train_targets, C=1e-6)
    print('evaluating analytical_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, a_solution)
    train_loss=do_evaluation(train_features, train_targets, a_solution)
    print('analytical_solution \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
    print('training LR using gradient descent...')
    gradient_descent_soln = do_gradient_descent(train_features,train_targets,dev_features,dev_targets,lr=0.1,C=1e-6,batch_size=23000,max_steps=1000000000,eval_steps=10)

    print('evaluating iterative_solution...')
    dev_loss=do_evaluation(dev_features, dev_targets, gradient_descent_soln)
    train_loss=do_evaluation(train_features, train_targets, gradient_descent_soln)
    print('gradient_descent_soln \t train loss: {}, dev_loss: {} '.format(train_loss, dev_loss))
    
    test_pred1 = get_predictions(test_features,a_solution)
    test_pred2 = get_predictions(test_features,gradient_descent_soln)
    print(' Square error betwwen analytical and gradient descent = ',format(np.sum(np.square(test_pred2-test_pred1))))




