#Load CSV data tutorial
import pandas as pd
import numpy as np
np.set_printoptions(precision=3,suppress=True)

import tensorflow as tf
from tensorflow.keras import layers
#downloading the data into a DataFrame
abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
        "Viscera weight", "Shell weight", "Age"])

print(abalone_train.head()) #measurements of abalone, a type of sea snail
#Task for this dataset: Predict age from other measurements

abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')

#packing features into a single numpy array
abalone_features = np.array(abalone_features)
abalone_features

#Making a regression model to predict age
abalone_model = tf.keras.Sequential([
  layers.Dense(64, activation='relu'),
  layers.Dense(1)
])

abalone_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam())

#To train the model, we have to pass the features and labels to Model.fit
abalone_model.fit(abalone_features, abalone_labels, epochs= 10) #epoch = 1 full pass through entire training data set

#Basic preprocessing
# creating the first layer
normalize = layers.Normalization()

#adapting the normalization layer to the data
normalize.adapt(abalone_features)

#incorporate the normalization layer into the model
norm_abalone_model = tf.keras.Sequential([
  normalize,
  layers.Dense(64, activation='relu'),
  layers.Dense(1)
])

norm_abalone_model.compile(loss = tf.keras.losses.MeanSquaredError(),
                           optimizer = tf.keras.optimizers.Adam())

norm_abalone_model.fit(abalone_features, abalone_labels, epochs=10)

#Mixed Data Types
#Not all datasets are limited to the same data type (i.e just floating points, just strings, etc)
#(Titanic dataset contains different data types)

titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
titanic.head()

titanic_features = titanic.copy()
titanic_labels = titanic_features.pop('survived')

#NOTE - Since there are different data types, you cannot just stack the features into a numpy array 
        # Option 1: preprocess the data offline to convert categorical columns to numerical columns t hen pass the processed output to the tensorflow model
        # Functional API oprates on symbolic tensors - they do not have a value
#Creating a symbolic input
input = tf.keras.Input(shape=(), dtype=tf.float32)

# Performing a calculation using the input
result = 2*input + 1

#the result doesn't have a value
result

calc = tf.keras.Model(inputs=input, outputs=result)
print(calc(np.array([1])).numpy())
print(calc(np.array([2])).numpy())

#Building the preprocessing model steps:
#Step 1: build a set of symbolic tf.keras.Input objects (match names and data-types of the CSV columns)
inputs = {}

for name, column in titanic_features.items():
  dtype = column.dtype
  if dtype == object:
    dtype = tf.string
  else:
    dtype = tf.float32

  inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

inputs

numeric_inputs = {name:input for name,input in inputs.items()
                  if input.dtype==tf.float32}

x = layers.Concatenate()(list(numeric_inputs.values()))
norm = layers.Normalization()
norm.adapt(np.array(titanic[numeric_inputs.keys()]))
all_numeric_inputs = norm(x)

all_numeric_inputs

#Step 2: Collect all the symbolic preprocessing results, to concatenate later
preprocessed_inputs = [all_numeric_inputs]
for name, input in inputs.items():
  if input.dtype == tf.float32:
    continue

  lookup = layers.StringLookup(vocabulary=np.unique(titanic_features[name]))
  one_hot = layers.CategoryEncoding(num_tokens=lookup.vocabulary_size())

  x = lookup(input)
  x = one_hot(x)
  preprocessed_inputs.append(x)

#Step 3: Concatenate all the preprocessed inputs together and build a model that handles preprocessing
preprocessed_inputs_cat = layers.Concatenate()(preprocessed_inputs)

titanic_preprocessing = tf.keras.Model(inputs, preprocessed_inputs_cat)

tf.keras.utils.plot_model(model = titanic_preprocessing , rankdir="LR", dpi=72, show_shapes=True)

#Step 4: convert to a dictionary of tensors (because Keras models do not automatically convert pandas DataFrames)
titanic_features_dict = {name: np.array(value) 
for name, value in titanic_features.items()}

#Slice out the first training example and pass it to the preprocessing model that was just created
features_dict = {name:values[:1] for name, values in titanic_features_dict.items()}
titanic_preprocessing(features_dict)

#build a model on top of this
def titanic_model(preprocessing_head, inputs):
  body = tf.keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  preprocessed_inputs = preprocessing_head(inputs)
  result = body(preprocessed_inputs)
  model = tf.keras.Model(inputs, result)

  model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=tf.keras.optimizers.Adam())
  return model

titanic_model = titanic_model(titanic_preprocessing, inputs)

#NOTE-when training the model, pass dictionary of features as 'x' and the label as 'y'
titanic_model.fit(x=titanic_features_dict, y=titanic_labels, epochs=10)

##This implementation includes everything from and after "Using tf.data" in the Load CSV data TensorFlow tutorial
#If more control is needed over input data pipelines or you need to use data that does not easily fit into memory, use tf.data
import itertools
def slices(features):
    for i in itertools.count():
        example = {name:values[i] for name, values in features.items()}
        yield example

##Example 1:for example in slices(titanic_features_dict):
for example in slices(titanic_features_dict):
  for name, value in example.items():
    print(f"{name:19s}: {value}")
  break

#The most basic tf.data.Dataset returns a tf.data.Dataset that implements a general version of the above slices function
features_ds = tf.data.Dataset.from_tensor_slices(titanic_features_dict)

#Example 2 - iterating over a tf.data.Dataset (i.e it is possible to iterate):
for example in features_ds:
  for name, value in example.items():
    print(f"{name:19s}: {value}")
  break

#from-tensor-slices can handle any structure of nested dictionaries or tuble
#The following makes a dataset of (features_dict, labels) pairs:
#Then shuffle and batch the data:
titanic_ds = tf.data.Dataset.from_tensor_slices((titanic_features_dict, titanic_labels))
titanic_batches = titanic_ds.shuffle(len(titanic_labels)).batch(32)

#pass the dataset (not features or labels)
titanic_model.fit(titanic_batches, epochs=5)

#EXAMPLES WITHOUT tf.data
#Example: fonts
fonts_zip = tf.keras.utils.get_file(
    'fonts.zip',  "https://archive.ics.uci.edu/ml/machine-learning-databases/00417/fonts.zip",
    cache_dir='.', cache_subdir='fonts',
    extract=True)

import pathlib
font_csvs =  sorted(str(p) for p in pathlib.Path('fonts').glob("*.csv"))

font_csvs[:10]

len(font_csvs)

fonts_ds = tf.data.experimental.make_csv_dataset(
    file_pattern = "fonts/*.csv",
    batch_size=10, num_epochs=1,
    num_parallel_reads=20,
    shuffle_buffer_size=10000)

#first batch
for features in fonts_ds.take(1):
  for i, (name, value) in enumerate(features.items()):
    if i>15:
      break
    print(f"{name:20s}: {value}")
print('...')
print(f"[total: {len(features)} features]")

#functions that decode a list of strings into a list of columns: tf.io.decode_csv
#Example: decoding titanic data as strings
text = pathlib.Path(titanic_file_path).read_text()
lines = text.split('\n')[1:-1]

all_strings = [str()]*10
all_strings

features = tf.io.decode_csv(lines, record_defaults=all_strings) 

for f in features:
  print(f"type: {f.dtype.name}, shape: {f.shape}")


#To parse the titanic data with their actual types create a list of record_defaults of the corresponding types

#tf.data.experimental.CsvDataset
#provides minimal CSV Dataset interface w/out the convenience features of other functions
simple_titanic = tf.data.experimental.CsvDataset(titanic_file_path, record_defaults=titanic_types, header=True)

for example in simple_titanic.take(1):
  print([e.numpy() for e in example])


#Multiple files
#To parse the fonts dataset using tf.data.experimental.CsvDataset, first determine the column types for record_defaults
font_line = pathlib.Path(font_csvs[0]).read_text().splitlines()[1]
print(font_line)
#total number of features can be found by counting commas
num_font_features = font_line.count(',')+1
font_column_types = [str(), str()] + [float()]*(num_font_features-2)

#reading input files sequentially
font_csvs[0]
simple_font_ds = tf.data.experimental.CsvDataset(
    font_csvs, 
    record_defaults=font_column_types, 
    header=True)

#to interleave multiple files, use Dataset.interleave
#The interleave method takes a map_func that creates a child Dataset for each element of the parent Dataset
#Dataset returned by the interleave returns elements by cycling over the child Datasets
def make_font_csv_ds(path):
  return tf.data.experimental.CsvDataset(
    path, 
    record_defaults=font_column_types, 
    header=True)

font_rows = font_files.interleave(make_font_csv_ds,
cycle_length=3)


fonts_dict = {'font_name':[], 'character':[]}

for row in font_rows.take(10):
  fonts_dict['font_name'].append(row[0].numpy().decode())
  fonts_dict['character'].append(chr(int(row[2].numpy())))

pd.DataFrame(fonts_dict)


#NOTE- Batches improve performance by breaking data into smaller chunks