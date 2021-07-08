def buildModel_LSTM_64_16(inputshape, outputs, options, softmax=True):
    import tensorflow as tf

    tf_recall=tf.keras.metrics.Recall()

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
        64,
        name="BiDiIn",
        return_sequences=True,
        input_shape=inputshape), input_shape=inputshape))
    model.add(tf.keras.layers.Dropout(options.dropout))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        16,
        return_sequences=False,
        input_shape=inputshape)))
    model.add(tf.keras.layers.Dropout(options.dropout))
    model.add(tf.keras.layers.Dense(
        name="ExpressionClass",
        units=outputs,
        activation='softmax' if softmax else None))
    
    model.compile(
        optimizer='adam',
        loss=options.loss,
        metrics=['accuracy',
                 'AUC',
                 tf_recall])

    return model

