
def extractFeatures(filename):
    import numpy as np
    labels = []
    labels = np.array([labels])
    features = []
    features = np.array([features])
    for line in file(filename):
        row = line.split(',')
        labels = np.append(labels, np.array(row[5]))
        features = np.append(features, np.array(row[0:5]))
    return np.array(features, dtype=int), np.array(labels, dtype=int)

def getFinalRating():
    import tensorflow as tf
    try:

        filename1 = "/home/tharindra/PycharmProjects/WorkBench/DataMiningAssignment/MovieTrain.csv"
        filename2 = "/home/tharindra/PycharmProjects/WorkBench/DataMiningAssignment/MovieTest.csv"
        tr_features, tr_labels = extractFeatures(filename1)
        ts_features, ts_labels = extractFeatures(filename2)

        tr_features = tr_features.reshape(2903, 5)
        ts_features = ts_features.reshape(1246, 5)

        tr_encode = oneHot(tr_labels)
        ts_encode = oneHot(ts_labels)

        print ("tr_features", tr_features)
        print ("ts_features", ts_features)
        print ("tr_encode", tr_encode)
        print ("ts_encode", ts_encode)

        # setting hyper parameters & other variables
        training_epochs = 2000
        n_features = 5
        n_classes = 3
        n_neurons_in_h1 = 100
        n_neurons_in_h2 = 50
        learning_rate = 0.01

        # placeholdr tensors built to store features(in X) and labels(in Y)
        X = tf.placeholder(tf.float32, [None, n_features], name='features')
        Y = tf.placeholder(tf.float32, [None, n_classes], name='labels')

        # network parameters(weights and biases) a set and initialized(Layer1)
        W1 = tf.Variable(tf.random_normal([n_features, n_neurons_in_h1]), name='weights1')
        b1 = tf.Variable(tf.random_normal([n_neurons_in_h1]), name='biases1')
        # activation function(sigmoid)
        y1 = tf.nn.sigmoid((tf.matmul(X, W1) + b1), name='activationLayer1')

        # network parameters(weights and biases) a set and initialized(Layer2)
        W2 = tf.Variable(tf.random_normal([n_neurons_in_h1, n_neurons_in_h2]), name='weights2')
        b2 = tf.Variable(tf.random_normal([n_neurons_in_h2]), name='biases2')
        # activation function(sigmoid)
        y2 = tf.nn.sigmoid((tf.matmul(y1, W2) + b2), name='activationLayer2')

        # output layer
        Wo = tf.Variable(tf.random_normal([n_neurons_in_h2, n_classes]), name='weightsOut')
        bo = tf.Variable(tf.random_normal([n_classes]), name='biasesOut')
        # activation function(softmax)
        a = tf.nn.softmax((tf.matmul(y2, Wo) + bo), name='activationOutputLayer')

        # tensorboard histograms on summary operations
        tf.summary.histogram("weights1", W1)
        tf.summary.histogram("biases1", b1)
        tf.summary.histogram("weights2", W2)
        tf.summary.histogram("biases2", b2)
        tf.summary.histogram("weightsOut", Wo)
        tf.summary.histogram("biasesOut", bo)

        # name scope for the cost function for more clarity on tensorboard
        with tf.name_scope('Cost'):
            # cost function
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(a), reduction_indices=[1]), name='CostFunction')
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
            # scalar summary for plotting cost variation againt epoches
            tf.summary.scalar('Cost', cross_entropy)

        # name scope for the accuracy for more clarity on tensorboard
        with tf.name_scope('Accuracy'):
            # compare predicted value from network with the expected value/target
            correct_prediction = tf.equal(tf.argmax(a, 1), tf.argmax(Y, 1))
            # accuracy determination
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")
            tf.summary.scalar('Accuracy', accuracy)

        # creation of tensorflow session to executethe computational graph
        with tf.Session() as sess:
            # log file for path for saving summary details for tensorborad visualizations
            writer = tf.summary.FileWriter("/home/tharindra/PycharmProjects/WorkBench/Movies/Train3-AfterChanging#ofClasses")
            writer.add_graph(sess.graph)
            merged_summary = tf.summary.merge_all()

        initial = tf.global_variables_initializer()

   
            # training loop over the number of epoches

            for epoch in range(training_epochs):
                # feeding training data/examples
                sess.run(train_step, feed_dict={X: tr_features, Y: tr_encode})
                # feeding testing data to determine model accuracy
                summary, acc = sess.run([merged_summary, accuracy], feed_dict={X: ts_features, Y: ts_encode})
                # write results to summary file
                writer.add_summary(summary, epoch)
                # print accuracy for each epoch
                print(epoch, acc)
                
             # initialization of all variables    
        with tf.Session() as sess:
        sess.run(initial)
        predict, actual = (tf.argmax(a, 1), tf.argmax(Y, 1))
        value =  str(sess.run((predict, actual), feed_dict={X: [[14, 0, 946, 10443, 660]], Y: [[1, 0, 0]]}))
        print value
        with open("/home/tharindra/PycharmProjects/WorkBench/DataMiningAssignment/hello.txt", "w") as f:
            f.write(value)
            
        saver = tf.train.Saver()
        saver.save(sess, "/home/tharindra/PycharmProjects/WorkBench/save.ckpt")
        print("Model saved")
        
            return predict,actual
    except Exception as e:
        print e



def oneHot(labels):
    import numpy as np
    p = len(labels)
    up = len(np.unique(labels))
    b = np.zeros((p, up))
    b[np.arange(p), labels] = 1
    return b


