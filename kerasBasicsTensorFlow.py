import tensorflow as tf
print("TensorFlow version:", tf.__version__)

#Loading a dataset

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#Building a ML model
learningModel = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = learningModel(x_train[:1]).numpy()
predictions


#Converting logits to probabilities for each class
tf.nn.softmax(predictions).numpy()

#Defining a loss function for training
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #the loss function takes a vector and returns a scalar loss for each example
                                                                          # The loss is zero if the model is sure of the correct class

loss_fn(y_train[:1], predictions).numpy()


#Configure and compiling the model
learningModel.compile(optimizer = 'adam',
              loss = loss_fn,
              metrics = ['accuracy'])

#Adjust model parameters and minimize the loss
learningModel.fit(x_train, y_train, epochs = 5)

#Check the model's performance
learningModel.evaluate(x_test, y_test, verbose = 2)

#Returning a probability
probability_model = tf.keras.Sequential([
    learningModel,
    tf.keras.layers.Softmax()
])
probability_model(x_test[:5])