#Beginner's quickstart tutorial TensorFlow

#importing TensorFlow
import tensorflow as tf
print("TensorFlow version:", tf.__version__)

#Loading a dataset
#pixel values of images ranges from 0 - 255 (to scale these values to a range of 0 - 1, divide by 255.0)
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#Build a ML model
#Sequential is useful for stacking layers
#Where each layer has 1 input tensor and 1 output tensor
#Layers are functions with a known mathematical structure that can be reused and has trainable variables
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions

#tf.nn.softmax converts logits to probabilities
tf.nn.softmax(predictions).numpy()

#Defining a loss function
#takes a vector of ground truth values and a vector of logits and returns a scalar loss for each example
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy() 
loss_fn(y_train[:1], predictions).numpy()

#Configure and compile with Keras
#optimizer = adam
model.compile(optimizer='adam',
loss=loss_fn,
metrics=['accuracy'])

#Train and evaluate with Model.fit and model.evaluate
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)

#Make the model return a probability
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
probability_model(x_test[:5])