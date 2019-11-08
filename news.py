import os
import argparse
from sklearn.datasets import fetch_20newsgroups
import tensorflow as tf
import numpy as np
from datetime import datetime

def network_one(learning_rate, epochs, batches, seed=12345):
    """ A fully connected feed forward network with one hidden layer and softmax output"""
    # seed = 12345  # seeds
    dict_size = 12500  # size of dictionary for tfidf tokenization
    min_epochs = 2  # for early stoppage, so we don't exit before adequate training
    # tf.set_random_seed(seed)
    # np.random.seed(seed)
    checkpoint_path = "news-{}-{}-{}-{}-{}.ckpt".format(int(combination), learning_rate, int(epochs), int(batches),
                                                        int(seed))

    news = fetch_20newsgroups(subset='all', random_state=seed)  # fetching the full dataset
    # news = fetch_20newsgroups(subset='all')  # fetching the full dataset

    categories = range(0, 20)  # so we can switch between full and subsets

    N = round(len(news.data) * 0.7)  # the index of the training set's end
    N_V = round(len(news.data) * 0.8)  # the index of the validation set end

    # splitting the data into the train,val, test
    x_train, y_train, x_val, y_val, x_test, y_test = news.data[0:N], news.target[0:N], \
                                                     news.data[N:N_V], news.target[N:N_V], \
                                                     news.data[N_V:], news.target[N_V:]

    y_train = tf.keras.utils.to_categorical(y_train, len(categories))  # reshaping for soft max
    y_test = tf.keras.utils.to_categorical(y_test, len(categories))
    y_val = tf.keras.utils.to_categorical(y_val, len(categories))

    from sklearn.feature_extraction.text import TfidfVectorizer  # Transforming to TFIDF
    vec = TfidfVectorizer(binary=False, use_idf=True, strip_accents=None,
                          max_features=dict_size, stop_words='english')
    x_train_tfidf = vec.fit_transform(x_train)
    x_test_tfidf = vec.transform(x_test)
    x_val_tfidf = vec.transform(x_val)

    from sklearn.preprocessing import normalize  # Normalising the x train,test and val with l2
    x_train_tfidf = normalize(x_train_tfidf, norm='l2', axis=1)
    x_test_tfidf = normalize(x_test_tfidf, norm='l2', axis=1)
    x_val_tfidf = normalize(x_val_tfidf, norm='l2', axis=1)

    # hyper-parameters
    # learning_rate = 0.001
    num_epochs = int(epochs)
    batch_size = int(batches)
    keep_prob = 0.5

    # Network sizes
    n_hidden_1 = 256  # hidden 1 neurons
    # n_hidden_2 = 256  # hidden 2 neurons
    # n_hidden_3 = 256
    num_input = x_train_tfidf.shape[1]  # for each word in our dictionary
    num_classes = len(categories)  # each category (20 in full model)

    # tf inputs x and labels y
    X = tf.placeholder("float", [None, num_input])  # x params stored here; each x is a tfidf for a news instance
    Y = tf.placeholder("float", [None, num_classes])  # labels here

    initialiser = tf.contrib.layers.xavier_initializer()  # initialising with xavier init
    weights = {
        'h1': tf.Variable(initialiser([num_input, n_hidden_1])),
        # 'h2': tf.Variable(initialiser([n_hidden_1, n_hidden_2])),
        # 'h3': tf.Variable(initialiser([n_hidden_2, n_hidden_3])),
        'out': tf.Variable(initialiser([n_hidden_1, num_classes]))
    }

    zero_init = tf.zeros_initializer()  # initialise biases to zero
    biases = {
        'b1': tf.Variable(zero_init([n_hidden_1])),
        # 'b2': tf.Variable(zero_init([n_hidden_2])),
        # 'b3': tf.Variable(zero_init([n_hidden_3])),
        'out': tf.Variable(zero_init([num_classes]))
    }

    def network(x):
        """ this is the network itself -- note that the softmax activation is outside this fn"""

        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])  # manually operating on weights and biases
        layer_1 = tf.nn.relu(layer_1)  # activating
        # layer_1 = tf.nn.sigmoid(layer_1)
        layer_1 = tf.layers.batch_normalization(layer_1)  # batch normalising
        # layer_1 = tf.layers.dropout(layer_1, keep_prob)  # dropout

        # layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # layer_2 = tf.nn.relu(layer_2)
        # layer_2 = tf.layers.batch_normalization(layer_2)
        # layer_2 = tf.nn.dropout(layer_2, keep_prob)

        # layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
        # layer_3 = tf.nn.relu(layer_3)
        # layer_3 = tf.layers.batch_normalization(layer_3)
        # layer_3 = tf.nn.dropout(layer_3, keep_prob)
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer

    logits = network(X)  # logits, building model
    prediction = tf.nn.softmax(logits)  # softmax activation

    # Loss is softmax crossentropy, optimiser adam
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # get predictions and compare accuracy
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # initialise and assign weight/bias initialisations
    init = tf.global_variables_initializer()

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_dir = "logs/run-{}".format(now)

    l = tf.summary.scalar('loss', loss_op)  # loss for tensorboard
    a = tf.summary.scalar('accuracy', accuracy)  # accuracy for tensorboard

    with tf.Session() as sess:
        sess.run(init)  # initialising

        train_acc = list()  # creating some lists to score los/acc histories for early stoppage or matplot plotting
        val_accs = list()

        train_writer = tf.summary.FileWriter(log_dir + '/train', tf.get_default_graph())  # for summaries to tensorboard
        val_writer = tf.summary.FileWriter(log_dir + '/validation', tf.get_default_graph())
        saver = tf.train.Saver()  # to save iterations of the model and the final model

        for epoch in range(1, num_epochs + 1):  # looping over epochs

            for index, offset in enumerate(range(0, x_train_tfidf.shape[0], batch_size)):
                # selects a window of the data of size batch_size and iterates through to complete a step
                batch_x = x_train_tfidf[offset: offset + batch_size, ]
                batch_x = batch_x.todense()  # turn the sparse subset matrix to dense for operations
                batch_x = np.asarray(batch_x)

                batch_y = y_train[offset: offset + batch_size, ]
                n_batches = round(x_train_tfidf.shape[0] / batch_size)  # so we can see progress

                # backpropagate -- the weights optimise in train op
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Epoch " + str(epoch) + ", Minibatch " + str(index) + " / " + str(n_batches) + " Loss= " +
                      "{:.4f}".format(loss) + ", Training Accuracy= " +
                      "{:.3f}".format(acc))
                train_acc.append(acc)

                if index % 10 == 0:
                    acc_summary = a.eval(feed_dict={X: batch_x, Y: batch_y})
                    loss_summary = l.eval(feed_dict={X: batch_x, Y: batch_y})
                    n = (epoch - 1) * n_batches + index  # so that tensorboard knows to plot chronologically by batch (note: not epoch)

                    train_writer.add_summary(loss_summary, n)
                    train_writer.add_summary(acc_summary, n)

                # if index % 50 == 0:
                #     # saving iterations of the model
                #     save_path = saver.save(sess, "tmp/news_final.ckpt")

                if index % 50 == 0:
                    # saving iterations of the model
                    save_path = saver.save(sess, checkpoint_path)

            print("Step Completed..." + "Mean Accuracy of Last Epoch:" +
                  str(np.mean(train_acc[-batch_size:])))

            #  Now for validation
            n_vals = round(x_val_tfidf.shape[0] / batch_size)

            for index, offset in enumerate(range(0, x_val_tfidf.shape[0], batch_size)):
                batch_x = x_val_tfidf[offset: offset + batch_size, ]
                batch_x = batch_x.todense()
                batch_x = np.asarray(batch_x)

                batch_y = y_val[offset: offset + batch_size, ]

                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})

                acc_summary = a.eval(feed_dict={X: batch_x, Y: batch_y})
                loss_summary = l.eval(feed_dict={X: batch_x, Y: batch_y})

                n = (epoch - 1) * n_vals + index  # sending index to tensorboard

                val_writer.add_summary(loss_summary, n)
                val_writer.add_summary(acc_summary, n)

                print("Step " + str(epoch) + ", Minibatch " + str(index) + " / " + str(n_vals) + " Loss= " +
                      "{:.4f}".format(loss) + ", Validation Accuracy= " +
                      "{:.3f}".format(acc))
                val_accs.append(acc)
            print("Step Completed..." + "Mean Accuracy of Last Epoch:" +
                  str(np.mean(train_acc[-batch_size:])) + " Validation Accuracy: " + str(np.mean(val_accs[-3:])))

            if epoch >= min_epochs and np.mean(val_accs[-n_vals:]) < (np.mean(val_accs[-2 * n_vals:-n_vals])):
                # early stopping if the validation accuracy decreases for a given epoch
                print("No Validation Improvement")
                break

            # print("Would you like to continue training?....")
            # answer = input('y/n')
            # if answer == 'n':
            #     break

        print("weight optimisation over...")

        # NOW THE TEST SET NEEDS TO BE BATCHED UP

        test_accs = list()
        n_tests = round(x_test_tfidf.shape[0] / batch_size)

        for index, offset in enumerate(range(0, x_test_tfidf.shape[0], batch_size)):
            batch_x = x_test_tfidf[offset: offset + batch_size, ]
            batch_x = batch_x.todense()
            batch_x = np.asarray(batch_x)

            batch_y = y_test[offset: offset + batch_size, ]

            #  loss and accuracy -- no longer running the training operation, not changing weights
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})

            print("Minibatch " + str(index) + " \ " + str(n_tests) + " Loss= " +
                  "{:.4f}".format(loss) + ", Testing Accuracy= " +
                  "{:.3f}".format(acc))

            test_accs.append(acc)
        print("Testing Accuracy:", np.mean(test_accs))

        # save_path = saver.save(sess, "/tmp/news_final.ckpt")

        save_path = saver.save(sess, checkpoint_path)
        print("SAVED MODEL TO: %s" % save_path)

        train_writer.close()
        val_writer.close()


def network_two(learning_rate, epochs, batches):
    import tensorflow as tf
    from datetime import datetime
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.model_selection import GridSearchCV
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import Sequential
    from keras.layers import Dense, Conv1D, Embedding, SpatialDropout1D, GlobalAveragePooling1D
    from keras.optimizers import Adam
    from keras.utils import to_categorical
    from keras.regularizers import l2

    tf.keras.backend.clear_session()
    tf.logging.set_verbosity(tf.logging.INFO)

    MAX_LENGTH = 200
    LEARNING_RATE = learning_rate # 0.01
    NO_EPOCHS = int(epochs) # 3
    BATCH_SIZE = int(batches) # 128

    # Fetch the dataset and split into training, validation and testing sets.
    news = fetch_20newsgroups(subset='all')
    train_index_end = round(0.7 * len(news.data))
    val_index_end = round(0.8 * len(news.data))
    news_train, news_val, news_test = news.data[:train_index_end], \
                                      news.data[train_index_end:val_index_end], \
                                      news.data[val_index_end:]

    x_train, y_train = news.data[:train_index_end], news.target[:train_index_end]
    x_val, y_val = news.data[train_index_end:val_index_end], news.target[train_index_end:val_index_end]
    x_test, y_test = news.data[val_index_end:], news.target[val_index_end:]

    categories = range(0, 20)

    # Tokenize the input to be sequences of words, then pad to a max length
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(news_train)
    x_train = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=MAX_LENGTH, padding='post')
    x_val = pad_sequences(tokenizer.texts_to_sequences(x_val), maxlen=MAX_LENGTH, padding='post')
    x_test = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=MAX_LENGTH, padding='post')
    y_train = to_categorical(y_train, len(categories))
    y_val = to_categorical(y_val, len(categories))
    y_test = to_categorical(y_test, len(categories))
    vocab_size = len(tokenizer.word_index) + 1

    # Define the Convolutional Neural Network model
    def create_model(learning_rate=0.01, feature_maps=32, kernel_size=6, weight_decay_rate=0.0001,
                     init_mode='glorot_normal'):
        model = Sequential()
        model.add(Embedding(vocab_size, 128))
        model.add(SpatialDropout1D(0.2))
        model.add(Conv1D(feature_maps,
                         activation='relu',
                         kernel_size=kernel_size,
                         kernel_regularizer=l2(weight_decay_rate),
                         kernel_initializer=init_mode))
        # model.add(Conv1D(64,
        #                  activation='relu',
        #                  kernel_size=12,
        #                  kernel_regularizer=l2(weight_decay_rate),
        #                  kernel_initializer=init_mode))
        model.add(GlobalAveragePooling1D())
        # model.add(Dense(32, activation='relu', kernel_initializer=init_mode))
        # model.add(Dense(32, activation='relu', kernel_initializer=init_mode))
        model.add(Dense(20, name='output', activation='softmax', kernel_initializer=init_mode))

        opt = Adam(lr=learning_rate)

        # Compile and return the model
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['acc'],
        )
        return model

    # Grid search for the best combinations of hyperparameters
    # learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    # epochs = range(12)
    # batches = [32, 64, 128, 256]
    # init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    # model = KerasClassifier(build_fn=create_model, epochs=6, batch_size=256, verbose=1)
    # param_grid = dict(init_mode=init_mode)
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    #
    # x_search = pad_sequences(tokenizer.texts_to_sequences(news.data), maxlen=MAX_LENGTH, padding='post')
    # y_search = to_categorical(news.target, len(categories))
    # grid_result = grid.fit(x_search, y_search)
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # Define callback functions to save a checkpoint after training as well as tensorboard logs
    checkpoint_path = "news-{}-{}-{}-{}-{}.ckpt".format(2, LEARNING_RATE, NO_EPOCHS, BATCH_SIZE, seed)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=0, period=1)
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="./logs/run-{}".format(now), write_graph=True,
                                                 update_freq="batch")

    # Create the model, then fit to the data and evaluate
    model = create_model()
    model.fit(x_train,
              y_train,
              validation_data=(x_val, y_val),
              epochs=NO_EPOCHS,
              batch_size=BATCH_SIZE,
              callbacks=[cp_callback, tensorboard])
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    print("Test Loss: " + str(loss))
    print("Test Accuracy: " + str(accuracy))

    print("Combination Two with learning rate: {} epochs: {} and batch size: {}".format(learning_rate, epochs, batches))


def main(combination, learning_rate, epochs, batches, seed):

    # Set Seed
    seed = np.int(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    print("Seed: {}".format(seed))


    if int(combination)==1:
        network_one(learning_rate, epochs, batches)
    if int(combination)==2:
        network_two(learning_rate, epochs, batches)

    print("Done!")

def check_param_is_numeric(param, value):

    try:
        value = float(value)
    except:
        print("{} must be numeric".format(param))
        quit(1)
    return value


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description="Assignment Program")
    arg_parser.add_argument("combination", help="Flag to indicate which network to run")
    arg_parser.add_argument("learning_rate", help="Learning Rate parameter")
    arg_parser.add_argument("iterations", help="Number of iterations to perform")
    arg_parser.add_argument("batches", help="Number of batches to use")
    arg_parser.add_argument("seed", help="Seed to initialize the network")

    args = arg_parser.parse_args()

    combination = check_param_is_numeric("combination", args.combination)
    learning_rate = check_param_is_numeric("learning_rate", args.learning_rate)
    epochs = check_param_is_numeric("epochs", args.iterations)
    batches = check_param_is_numeric("batches", args.batches)
    seed = check_param_is_numeric("seed", args.seed)

    main(combination, learning_rate, epochs, batches, seed)
