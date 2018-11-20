import numpy as np
import os 
import math
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

rnd_seed=1

def conv_concat(X, y, y_dim):

    yb = tf.reshape(y, [tf.shape(X)[0], 1, 1, y_dim])
    yb = tf.tile(yb, [1, tf.shape(X)[1], tf.shape(X)[2] ,1])
    output = tf.concat([X, yb], 3)
    return output

def lin_concat(X, y, y_dim):

    yb = tf.reshape(y, [tf.shape(X)[0], y_dim])
    output = tf.concat([X, yb], 1)
    
    return output

def lrelu(x, alpha=0.1):

    """
    Implements the leakyRELU function:

    inputs X, returns X if X>0, returns alpha*X if X<0
    """


    return tf.maximum(alpha*x,x)

def evaluation(Y_pred, Y):

    """
    Returns the accuracy by comparing the convoluted output Y_hat
    with the labels of the samples Y

    """
    
    correct = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    
    return accuracy

def supervised_random_mini_batches(X, Y, mini_batch_size, seed):

    """
    Creates a list of random mini_batches from (X, Y)
    
    Arguments:
    X -- input data, of shape (number of examples, input size)
    Y -- true "label" one hot matrix of shape (number of examples, n_classes)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    
    m = X.shape[0]        #number of examples in set
    n_classes = Y.shape[1]
    mini_batches=[]
    
    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    #print('Zeroth element of batch permutation:', permutation[0])
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:]
    #partition of (shuffled_X, shuffled_Y) except the last mini_batch

    num_complete_mini_batches = math.floor(m/mini_batch_size)
    for k in range(num_complete_mini_batches):
        mini_batch_X = shuffled_X[k*mini_batch_size:(k+1)*mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k*mini_batch_size:(k+1)*mini_batch_size,:]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    # handling the case of last mini_batch < mini_batch_size    
    if m % mini_batch_size !=0:
        
        mini_batch_X = shuffled_X[mini_batch_size*num_complete_mini_batches:m,:]
        mini_batch_Y = shuffled_Y[mini_batch_size*num_complete_mini_batches:m,:]
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def unsupervised_random_mini_batches(X, mini_batch_size, seed):

    """
    Creates a list of random mini_batches from (X, Y)
    
    Arguments:
    X -- input data, of shape (number of examples, input size)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of mini_batch_X
    """
    
    m = X.shape[0]        #number of examples in set
    mini_batches=[]
    
    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    #print('Zeroth element of batch permutation:', permutation[0])
    shuffled_X = X[permutation,:]
    
    #partition of shuffled_X except the last mini_batch
    
    num_complete_mini_batches = math.floor(m/mini_batch_size)
    for k in range(num_complete_mini_batches):
        mini_batch_X = shuffled_X[k*mini_batch_size:(k+1)*mini_batch_size,:]
        mini_batches.append(mini_batch_X)
        
    # handling the case of last mini_batch < mini_batch_size    
    if m % mini_batch_size !=0:
        
        mini_batch_X = shuffled_X[mini_batch_size*num_complete_mini_batches:m,:]
        mini_batches.append(mini_batch_X)
    
    return mini_batches

def unsupervised_random_mini_batches_labels(X, mini_batch_size, seed):

    """
    Creates a list of random mini_batches from (X, Y)
    
    Arguments:
    X -- input data, of shape (number of examples, input size)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of mini_batch_X
    """
    
    m = X.shape[0]        #number of examples in set
    mini_batches=[]
    
    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    #print('Zeroth element of batch permutation:', permutation[0])
    shuffled_X = X[permutation]
    
    #partition of shuffled_X except the last mini_batch
    
    num_complete_mini_batches = math.floor(m/mini_batch_size)
    for k in range(num_complete_mini_batches):
        mini_batch_X = shuffled_X[k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batches.append(mini_batch_X)
        
    # handling the case of last mini_batch < mini_batch_size    
    if m % mini_batch_size !=0:
        
        mini_batch_X = shuffled_X[mini_batch_size*num_complete_mini_batches:m]
        mini_batches.append(mini_batch_X)
    
    return mini_batches

# def preprocess_true(true):

# 	mean_true=true[true!=0].mean()
# 	std_true=np.std(true[np.where(true!=0)],axis=0)

# 	true[true!=0]-=mean_true
# 	true=np.where(true==0,0,true/std_true)

# 	return true, mean_true, std_true
# def preprocess_reco(reco):

#   mean_reco=np.mean(reco,axis=0)
#   std_reco=np.std(reco,axis=0)

#   reco-=mean_reco
#   reco=np.where(reco==0,0,reco/std_reco)

#   return reco, mean_reco, std_reco

# def reconstruct(sample, mean, std):
#   return np.where(sample!=0,sample*std+mean,0)

def four_cells(img):
    img = img.flatten()
    return img[img.argsort()[-4:][::-1]]

def normalise(X):

    X=np.where(X>0,X,0)
    #temp = X.reshape(X.shape[0],X.shape[1]*X.shape[2]*X.shape[3])
    #temp = temp.sum(axis=1)
    max_X = np.max(X)
    #max_X = np.max(temp.sum(axis=1)) 
    
    X=X/max_X
    min_X=0
    
    return X, min_X, max_X

def denormalise(X, min_X, max_X):
    #mask = X!=0
    #return np.where(X!=0, np.exp(X*max_X), 0)
    return np.where(X!=0, X*max_X, 0)


# def normalise(X):

#     X=np.where(X>12 ,X,0)
#     #X=np.where(X>12,np.log(X),0)

#     # E_max = X.max()
#     # E_min = np.min(X[X>0])
#     # X = np.where(X>0, X-(E_max+E_min)/2,0)
#     # X/=X.max()

#     E_min=np.min(X[X>0])
#     X=np.where(X>0,X-E_min,0)
#     E_max=np.max(X)
#     X=np.where(X!=0,X/E_max,0)
    
#     return X, E_max, E_min

# def denormalise(X, E_max, E_min):
#     # X=X*E_max-E_min)/2
#     # X=np.where(X!=0, X+(E_max+E_min)/2, 0)

#     X=np.where(X!=0,X*E_max,0)
#     X=np.where(X!=0, X+E_min, 0)
#     #X=np.where(X!=0, np.exp(X), 0)

#     return X

def delete_undetected_events(true, reco):

    pos_rejected=[]
    
    for i in range(len(reco)):
        if np.array_equal(reco[i],np.zeros_like(reco[i])) or np.array_equal(true[i],np.zeros_like(true[i])) :
            pos_rejected.append(i)

    reco_filtered=np.delete(reco,pos_rejected,axis=0)
    true_filtered=np.delete(true, pos_rejected, axis=0)

    assert len(true_filtered)==len(reco_filtered)

    return true_filtered, reco_filtered

def selection(true, reco, n_cells, energy_fraction):

    pos_selected=[]
    pos_rejected=[]
    
    for i in range(len(reco)):
        tot_E=reco[i].sum()
        reshaped=reco[i].flatten()
        if (reshaped[reshaped.argsort()[-n_cells:][::-1]].sum())/tot_E<energy_fraction:
            pos_rejected.append(i)
        else:
            pos_selected.append(i)

    reco_filtered=np.delete(reco,pos_rejected,axis=0)
    true_filtered=np.delete(true, pos_rejected, axis=0)

    reco_rejected=np.delete(reco, pos_selected, axis=0)
    true_rejected=np.delete(true, pos_selected, axis=0)

    assert len(true_filtered)==len(reco_filtered)

    return true_filtered, reco_filtered, true_rejected, reco_rejected

def crop_conditional(true, reco, dim):

    ETs=[]
    assert len(reco)==len(true)
    cropped_reco=np.zeros(shape=(reco.shape[0],2*dim+1,2*dim+1,1))
    max_x = reco.shape[2]
    max_y = reco.shape[1]
    
    for i in range(len(reco)):
        y , x , _= np.where(true[i]>0)
        ETs.append(true[i][np.where(true[i]>0)][0])
        reco_y, reco_x, _ = np.where(reco[i]==reco[i].max())
        
        #CENTER OF IMAGE
        if 2*dim<reco_y[0]<=max_y-2*dim and 2*dim<reco_x[0]<=max_x-2*dim:
            cropped_reco[i]=reco[i, reco_y[0]-dim:reco_y[0]+dim+1, reco_x[0]-dim:reco_x[0]+dim+1, :]

    ETs=np.array(ETs)    
    return ETs, cropped_reco

def crop_function(true, reco, dim):
    
    assert len(reco)==len(true)
    j=0
    cropped_reco=np.zeros(shape=(reco.shape[0],2*dim,2*dim,1))
    cropped_true=np.zeros(shape=(true.shape[0],2*dim,2*dim,1))
    max_x = reco.shape[2]
    max_y = reco.shape[1]
    
    for i in range(len(reco)):
        y , x , _= np.where(true[i]>0)

        #CORNERS
        
        #top left corner
        # if y[0]<=2*dim  and x[0]<=2*dim:
        #     #print(i, x, y)
        #     cropped_reco[i]=reco[i,0:2*dim, 0:2*dim, :]
        #     cropped_true[i]=true[i,0:2*dim, 0:2*dim, :]
        #     j+=1
            
        # #top right corner
        # elif y[0]<=2*dim  and max_x-2*dim<x[0]<=max_x:
        #     #print(i, x, y)
        #     cropped_reco[i]=reco[i,0:2*dim,  max_x-2*dim:max_x, :]
        #     cropped_true[i]=true[i,0:2*dim,  max_x-2*dim:max_x, :]
        #     j+=1
            
        # #bottom right corner
        # elif max_y-2*dim<y[0]<=max_y and max_x-2*dim<x[0]<=max_x:
        #     #print(i, x, y)
        #     cropped_reco[i]=reco[i,max_y-2*dim:max_y, max_x-2*dim:max_x, :]
        #     cropped_true[i]=true[i,max_y-2*dim:max_y, max_x-2*dim:max_x, :]
        #     j+=1
            
        # #bottom left corner
        # elif max_y-2*dim<y[0]<=max_y and x[0] <=2*dim:
        #     #print(i, x, y)
        #     cropped_reco[i]=reco[i,max_y-2*dim:max_y, 0:2*dim, :]
        #     cropped_true[i]=true[i,max_y-2*dim:max_y, 0:2*dim, :]
        #     j+=1
            
        # #BORDERS
        
        # #bottom border without corners
        # elif max_y-2*dim<=y[0]<max_y and 2*dim<x[0]<=max_x-2*dim:
        #     #print(i, x, y)
        #     cropped_reco[i]=reco[i, max_y-2*dim:max_y, x[0]-dim:x[0]+dim,  :]
        #     cropped_true[i]=true[i, max_y-2*dim:max_y, x[0]-dim:x[0]+dim,  :]
        #     j+=1
            
        # #top border without corners
        # elif y[0]-2*dim<=0 and 2*dim<x[0]<=max_x-2*dim:
        #     #print(i, x, y)
        #     cropped_reco[i]=reco[i, 0:2*dim, x[0]-dim:x[0]+dim,  :]
        #     cropped_true[i]=true[i, 0:2*dim, x[0]-dim:x[0]+dim,  :]
        #     j+=1
            
        # #left border without corners
        # elif 2*dim<y[0]<=max_y-2*dim and x[0]<=2*dim:
        #     #print(i, x, y)
        #     cropped_reco[i]=reco[i,y[0]-dim:y[0]+dim,0:2*dim,  :]
        #     cropped_true[i]=true[i,y[0]-dim:y[0]+dim,0:2*dim,  :]
        #     j+=1
            
        # #right border without corners
        # elif 2*dim<y[0]<=max_y-2*dim and max_x-2*dim<x[0]<=max_x:
        #     #print(i, x, y)
        #     cropped_reco[i]=reco[i, y[0]-dim:y[0]+dim, max_x-2*dim:max_x,  :]
        #     cropped_true[i]=true[i, y[0]-dim:y[0]+dim, max_x-2*dim:max_x,  :]
        #     j+=1
 
        
        #CENTER OF IMAGE
        if 2*dim<y[0]<=max_y-2*dim and 2*dim<x[0]<=max_x-2*dim:
            #print(i, x, y)
            cropped_reco[i]=reco[i, y[0]-dim:y[0]+dim, x[0]-dim:x[0]+dim, :]
            cropped_true[i]=true[i, y[0]-dim:y[0]+dim, x[0]-dim:x[0]+dim, :]
            j+=1
        #assert i==j-1
    return cropped_true, cropped_reco

def load_batch(true_path, reco_path, i):
    
    with open(reco_path+'sample{0}.pickle'.format(i), 'rb') as f:
        reco=pickle.load(f)

    with open(true_path+'sample{0}.pickle'.format(i), 'rb') as f:
        true=pickle.load(f)
    
    #cut that extra produced pixel
    true=true[:,1:true.shape[1]-1,1:true.shape[2]-1,:]
    
    return true, reco

def load_data(true_path, reco_path, n_batches, select=False, n_cells=None, energy_fraction=0.0, crop=False, dim=None, preprocess=None, test_size=None):

    if n_batches == 1:

        #delete undetected particles
        true1, reco1 = load_batch(true_path, reco_path, 0)
        true, reco = delete_undetected_events(true1, reco1)
        
        #delete too noisy events
        if select:
            true_output, reco_output, _, _,  = selection(true, reco, n_cells, energy_fraction)
            if crop:
                true_output, reco_output = crop_function(true_output, reco_output, dim)

            true=true_output
            reco=reco_output

        if crop:
            true_output, reco_output = crop_function(true, reco, dim)
            

            true=true_output
            reco=reco_output

    elif n_batches > 1:

        true1, reco1 = load_batch(true_path, reco_path, 0)
        true, reco = delete_undetected_events(true1, reco1)
        
        #delete too noisy events
        if select:
            true_output, reco_output, _, _,  = selection(true, reco, n_cells, energy_fraction)
            if crop:
                true_output, reco_output = crop_function(true_output, reco_output, dim)
        
        if crop:
            true_output, reco_output = crop_function(true, reco, dim)

        for i in range(1, n_batches):

            true1, reco1 = load_batch(true_path, reco_path, i)
            true_temp, reco_temp = delete_undetected_events(true1, reco1)
            
            #delete too noisy events
            if select:
                true_temp, reco_temp, _, _,  = selection(true_temp, reco_temp, n_cells, energy_fraction)
                
                if crop:
                    true_temp, reco_temp = crop_function(true_temp, reco_temp, dim)
            
            if crop and not select:
                true_temp, reco_temp = crop_function(true_temp, reco_temp, dim)

            if crop or select:
                true = np.concatenate((true_output, true_temp), axis=0)
                reco = np.concatenate((reco_output, reco_temp), axis=0)
            else:
                true = np.concatenate((true, true_temp), axis=0)
                reco = np.concatenate((reco, reco_temp), axis=0)


    if preprocess =='normalise':

        reco, min_reco, max_reco = normalise(reco)
        true, min_true, max_true = normalise(true)
        m = reco.shape[0]
        train_size = m - test_size

        train_true = true[0:train_size]
        test_true = true[train_size:m]

        train_reco = reco[0:train_size]
        test_reco = reco[train_size:m]

        return train_true, test_true, min_true, max_true, train_reco, test_reco, min_reco, max_reco

    else:
        m = reco.shape[0]
        train_size = m - test_size

        train_true = true[0:train_size]
        test_true = true[train_size:m]

        train_reco = reco[0:train_size]
        test_reco = reco[train_size:m]

        return train_true, test_true, train_reco, test_reco

def load_data_conditional(true_path, reco_path, n_batches, dim=None, preprocess=None, test_size=None):

    if n_batches == 1:

        #delete undetected particles
        true1, reco1 = load_batch(true_path, reco_path, 0)
        true2, reco2 = delete_undetected_events(true1, reco1)
        ETs, reco_output = crop_conditional(true2, reco2, dim)
        #delete too noisy events

    elif n_batches > 1:

        true1, reco1 = load_batch(true_path, reco_path, 0)
        true2, reco2 = delete_undetected_events(true1, reco1)
        ETs, reco_output = crop_conditional(true2, reco2, dim)
        

        for i in range(1, n_batches):

            true1, reco1 = load_batch(true_path, reco_path, i)
            true_temp, reco_temp = delete_undetected_events(true1, reco1)
            ETs_temp, reco_output_temp = crop_conditional(true_temp, reco_temp, dim)
            
            #delete too noisy events
            ETs = np.concatenate((ETs, ETs_temp), axis=0)
            reco_output = np.concatenate((reco_output, reco_output_temp), axis=0)

    true = ETs
    reco = reco_output

    if preprocess =='normalise':

        reco, min_reco, max_reco = normalise(reco)
        true, min_true, max_true = normalise(true)
        m = reco.shape[0]
        train_size = m - test_size

        train_true = true[0:train_size]
        test_true = true[train_size:m]

        train_reco = reco[0:train_size]
        test_reco = reco[train_size:m]

        return train_true, test_true, min_true, max_true, train_reco, test_reco, min_reco, max_reco

    else:
        m = reco.shape[0]
        train_size = m - test_size

        train_true = true[0:train_size]
        test_true = true[train_size:m]

        train_reco = reco[0:train_size]
        test_reco = reco[train_size:m]

        return train_true, test_true, train_reco, test_reco

def draw_one_sample(train_true, train_reco, preprocess=None,
    min_true=None, max_true=None, min_reco=None, max_reco=None, 
    save=False, PATH=None):

    j = np.random.randint(len(train_true))

    X_batch_A = train_true[j]
    X_batch_B = train_reco[j]

    if preprocess=='normalise':
        X_batch_A=denormalise(X_batch_A, min_true, max_true)
        X_batch_B=denormalise(X_batch_B, min_reco, max_reco)

    n_H_A, n_W_A ,n_C = X_batch_A.shape
    n_H_B, n_W_B ,n_C = X_batch_B.shape

    plt.subplot(2,2,1)
    plt.imshow(X_batch_A.reshape(n_H_A,n_W_A))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('True E_T: {:.6g} MeV'.format(X_batch_A.sum()))
    plt.subplots_adjust(wspace=0.2,hspace=0.2)

    plt.subplot(2,2,2)
    plt.imshow(X_batch_B.reshape(n_H_B,n_W_B))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Reco E_T: {:.6g} MeV'.format(X_batch_B.sum()))
    plt.subplots_adjust(wspace=0.2,hspace=0.2)

    plt.suptitle('HCAL MC simulation\n ')
    fig = plt.gcf()
    fig.set_size_inches(11,4)
    if not save:
        plt.show()
    else:
        plt.savefig(PATH+'/HCAL_reconstruction_example_{0}.png'.format(j),dpi=80)

def draw_one_sample_conditional(train_true, train_reco, preprocess=None,
    min_true=None, max_true=None, min_reco=None, max_reco=None, 
    save=False, PATH=None):

    j = np.random.randint(len(train_true))

    X_batch_A = train_true[j]
    X_batch_B = train_reco[j]

    if preprocess=='normalise':
        X_batch_B=denormalise(X_batch_B, min_reco, max_reco)
        X_batch_A=denormalise(X_batch_A, min_true, max_true)

    n_H_B, n_W_B ,n_C = X_batch_B.shape

    plt.imshow(X_batch_B.reshape(n_H_B,n_W_B))
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('True E_T: {:.6g} MeV, Reco MC E_T: {:.6g}'.format(X_batch_A, X_batch_B.sum()))

    plt.suptitle('HCAL MC simulation\n ')
    fig = plt.gcf()
    fig.set_size_inches(11,4)
    if not save:
        plt.show()
    else:
        plt.savefig(PATH+'/HCAL_reconstruction_example_{0}.png'.format(j),dpi=80)

def draw_nn_sample(X_A, X_B, i, preprocess=False,
    min_true=None, max_true=None, min_reco=None, max_reco=None, f=None, 
    save=True, is_training=False, total_iters=None, PATH=None):

    j = np.random.randint(len(X_A))

    _, n_H_A, n_W_A, n_C = X_A.shape 
    _, n_H_B, n_W_B, _ = X_B.shape 

    #draw the response for one particle
    if i ==1 :
        X_A = X_A[j]
        X_B = X_B[j]
        sample_nn = f(X_A.reshape(1, n_H_A, n_W_A, n_C))
    #draw the response for i particles
    if i>1:
        X_A = X_A[j:j+i]
        X_B = X_B[j:j+i]
        X_A = X_A.sum(axis=0)
        X_B = X_B.sum(axis=0)

        sample_nn = f(X_A.reshape(1, n_H_A, n_W_A, n_C))

    if preprocess=='normalise':

        X_A=denormalise(X_A, min_true, max_true)
        X_B=denormalise(X_B, min_reco, max_reco)
        sample_nn=denormalise(sample_nn, min_reco, max_reco)

    plt.subplot(1,3,1)
    plt.gca().set_title('True ET {0:.6g}'.format(X_A.sum()))
    plt.imshow(X_A.reshape(n_H_A,n_W_A))
    plt.axis('off')
    plt.subplots_adjust(wspace=0.2,hspace=0.2)

    plt.subplot(1,3,2)
    plt.gca().set_title('MC Reco ET {0:.6g}'.format(X_B.sum()))
    plt.imshow(X_B.reshape(n_H_B,n_W_B))
    plt.axis('off')
    plt.subplots_adjust(wspace=0.2,hspace=0.2)

    plt.subplot(1,3,3)
    plt.gca().set_title('NN Reco ET {0:.6g}'.format(sample_nn.sum()))
    plt.imshow(sample_nn.reshape(n_H_B,n_W_B))
    plt.axis('off')
    plt.subplots_adjust(wspace=0.2,hspace=0.2)
    if is_training:
        plt.suptitle('At iter {0}'.format(total_iters))
    fig = plt.gcf()
    fig.set_size_inches(10,8)

    if save:
        if is_training:
            plt.savefig(PATH+'/sample_at_iter_{0}.png'.format(total_iters),dpi=80)
        else:
            plt.savefig(PATH+'/nn_reco_sample_{0}.png'.format(j),dpi=80)
    else:
        plt.show()








