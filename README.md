from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import h5py
import matplotlib.pyplot as plt 
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


N_DIGITS = 2
X_FEATURE = 'x'  # Name of the input feature.



RATE=4096
BATCH = 1
gloss=[]
gacc=[]

####
####   Network architecture
####
def conv_model(features, labels, mode):
  feature = tf.reshape(features[X_FEATURE], [-1, RATE,1])
  with tf.variable_scope('conv_layer1'):
    h_conv1 = tf.layers.conv1d(feature, filters=16, kernel_size=[16], padding='valid', activation=tf.nn.relu)
    h_pool1 = tf.layers.max_pooling1d(h_conv1, pool_size=4, strides=4, padding='valid', name='p1')
  with tf.variable_scope('conv_layer2'):
    h_conv2 = tf.layers.conv1d(h_pool1, filters=32, kernel_size=[8], padding='valid',dilation_rate=4, activation=tf.nn.relu)
    h_pool2 = tf.layers.max_pooling1d(h_conv2, pool_size=4, strides=4, padding='valid', name='p2')
  with tf.variable_scope('conv_layer3'):
    h_conv3 = tf.layers.conv1d(h_pool2, filters=64, kernel_size=[8], padding='valid',dilation_rate=4, activation=tf.nn.relu)
    h_pool3 = tf.layers.max_pooling1d(h_conv3, pool_size=4, strides=4, padding='valid', name='p3')
    
    dim = h_pool3.get_shape().as_list()
    fcnn = dim[1]*dim[2]
    h_pool3_flat = tf.reshape(h_pool3, [-1, fcnn])   ## linearize the matrix into 1D vector, =64*119
    #print ("================ P3F: ", mode, h_pool3_flat.get_shape())

  # Densely connected layer
  h_fc1 = tf.layers.dense(h_pool3_flat, 64, activation=tf.nn.relu, name='d1')
  ### if mode == tf.estimator.ModeKeys.TRAIN: h_fc1 = tf.layers.dropout(h_fc1, rate=0.5)   ## dropout not used by Huerta

  ### 2-output  
  logits = tf.layers.dense(h_fc1, N_DIGITS, activation=None, name='d2')   ## No activation

  # Compute predictions
  predict_op =  tf.argmax(input=logits, axis=1)   ## return largest index

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'class': predict_op,
        'prob': tf.nn.softmax(logits, name="softmax_tensor")
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Create training op.
  loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
  print ("================ loss: ", mode, loss.get_shape())

  summary_hook = tf.train.SummarySaverHook(2, output_dir='5.24normalization', 
                                           summary_op=tf.summary.merge_all())

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op  = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[summary_hook])

  # Compute evaluation metrics.
  eval_metric_ops = {
      #'accuracy': tf.metrics.accuracy(labels=labels, predictions=predict_op )
      'sensitivity':tf.metrics.sensitivity_at_specificity(specificity=0.05,num_thresholds=200, predictions=predict_op,labels=labels)
  }
  return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops, training_hooks=[summary_hook])

def prepare_data(SNR):
    file_name="bbh.h5"
    f=h5py.File(file_name,'r')   

    
    #for i in [0,16,39,61,82,102,121,139,156,172,187,201,214,226,237,247,256,264,271,277,282,286,289,291]:
    for i in (0,100):
        key = 'waveform/%d'%i
        k1  = 'waveform/%d/t2m'%i
        k2  = 'waveform/%d/hp'%i
        m1  = f[key].attrs['m'][0]
        m2  = f[key].attrs['m'][1]
        t2m = f[k1][:]
        hp  = f[k2][:]
        
        c=[]
        ac=[]
        k=0
        hp1 =[]
        ag=[]
        
        a=t2m.tolist()
        for j in a:
            if abs(j)<0.01:
                b=a.index(j)
                c.append(b)
                k=k+1
        k=int(k/2)
        #print(c[k])
    
    
        d=int(2457+1638*np.random.rand())
        e=c[k]-d
        g=4096-d
        h=c[k]+g
        t2m1 =f[k1][e:h]
        
        n   =t2m1.tolist()
        for _ in range (4096-len(n)):
            n.append(int(0))
            
        #print(len(n))
        
        t2m1=n
    
        hp1  =f[k2][e:h]
        m    =hp1.tolist()
        for _ in range(4096-len(m)):
            m.append(int(0))
        
        ad=max(m)
        
        for ab in m:
            ab=ab/ad*SNR*0.5
            ac.append(ab)
            
        hp1=ac
        
    N=200   #signal legth
    N0=N//2
    X = (np.random.rand(N,RATE))-0.5
    X[:N0] = X[:N0]+hp1
    X      = X.astype(np.float32)
    y      = np.zeros(N)
    y[:N0] = np.ones(N0)
    y      = y.astype(np.int32)
    
    #print(X)
    #print("this is y",y)
    return train_test_split(X, y, test_size=0.1,random_state=2)
    
       
    f.close()    
    
     


def CNN(SNR):
    tf.logging.set_verbosity(tf.logging.INFO)
    #tf.logging.set_verbosity(tf.logging.WARN)
    
    X_train, X_test, y_train, y_test = prepare_data(SNR)
    ### Download and load MNIST dataset.##mnist = tf.contrib.learn.datasets.DATASETS['mnist']('/tmp/mnist')
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = { X_FEATURE: X_train },  y = y_train,
        batch_size = BATCH,### step = N * epoches / Batch 
        num_epochs = 50,  shuffle= True)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x= { X_FEATURE: X_test },   y= y_test,
        #batch_size = None,
        num_epochs=1,   shuffle= False)

    ########## CNN
    #try: os.remove('5.24normalization')  
    #except OSError: pass

    config = tf.contrib.learn.RunConfig(
        log_device_placement=True,
        save_checkpoints_steps=100,
        save_checkpoints_secs=None,
        save_summary_steps=10,
        gpu_memory_fraction=1,
        model_dir='5.24normalization'
    )
    classifier = tf.estimator.Estimator(model_fn=conv_model, config=config)
    scores =classifier.train(input_fn=train_input_fn, steps=None)
    scores = classifier.evaluate(input_fn=test_input_fn, steps=None)
    
    ###===========================
    aj=[]
    prediction = classifier.predict(input_fn=test_input_fn)
    for p in prediction:  
        print (p)
        ah=p["class"]
        aj.append(ah)
    print("class is:",aj)
    print("this is y_test",y_test)
        
    print(scores)  
    return scores['loss'],scores['sensitivity']
    

    if 0:  
        ######### Linear classifier.
        feature_columns = [ tf.feature_column.numeric_column(X_FEATURE, shape=RATE) ]

        classifier = tf.estimator.LinearClassifier(feature_columns = feature_columns, n_classes=N_DIGITS)
        classifier.train(input_fn=train_input_fn, steps=None)
        scores = classifier.evaluate(input_fn=test_input_fn, steps=None)
        print("+++++ LC: ", scores)


def main(unused):
    #snr = np.linspace(1,0.1,10)
    snr = [0.01]
    for x in snr:
        l, a = CNN(x)
        gloss.append(l)
        gacc.append(a)
    print(gloss)
    print(gacc)
    
    plt.plot(snr,gloss)
    plt.plot(snr,gacc)
    plt.title("snr vs sensitivity")
    plt.xlabel("snr")
    plt.ylabel("sensitivity")
if __name__ == '__main__':
  tf.app.run()
  print("End-----------------")


