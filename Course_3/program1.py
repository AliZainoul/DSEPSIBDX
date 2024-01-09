# Importing necessary libraries
import tensorflow as tf

# Creating a simple neural network model for illustration
model = tf.keras.Sequential([
    # First Layer (Dense Layer):
    # - 128 neurons: Each neuron receives input from all 784 features and contributes to the learning process.
    # - activation='relu': Rectified Linear Unit (ReLU) activation function introduces non-linearity.
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    
    # Second Layer (Dropout Layer):
    # - 0.2: Represents the dropout rate. It means 20% of randomly selected neurons will be deactivated during training to prevent overfitting.
    tf.keras.layers.Dropout(0.2),
    
    # Third Layer (Dense Layer - Output Layer):
    # - 10 neurons: Each neuron represents a class in a multi-class classification problem (e.g., recognizing digits from 0 to 9).
    # - activation='softmax': Softmax activation function converts the output into a probability distribution over classes.
    tf.keras.layers.Dense(10, activation='softmax')
])

# Input Shape (Parameter of the First Layer):
# - input_shape=(784,): Specifies the shape of the input data. Each input sample has 784 features.
# - In image classification tasks, each feature might represent a pixel value in a flattened image (28x28 pixels = 784 features).
# - So, the 784 in input_shape=(784,) represents the number of features or input dimensions in each data sample.

# Displaying the summary of the model architecture
print(model.summary())


# Details:
# 1. Importing libraries:
#    - import tensorflow as tf : Imports the TensorFlow library.

# 2. Creating the model:
#    - tf.keras.Sequential : Creates a sequential model.
#      - Dense Layer (128 neurons, ReLU activation, input_shape=(784,))
#        - This layer represents the first hidden layer of the network.
#      - Dropout Layer (20% rate)
#        - This layer prevents overfitting by randomly deactivating 20% of neurons during training.
#      - Output Dense Layer (10 neurons, softmax activation)
#        - This layer represents the output layer of the network.

# 3. Displaying the architecture:
#    - print(model.summary()) : Displays a summary of the model architecture.
#      - The summary includes the layer type, output shape, and the number of parameters to train for each layer.
#      - Information also includes the total number of parameters, the number of trainable parameters, and the number of non-trainable parameters.

# 4. Results obtained:
#    - The model consists of three layers.
#    - The first dense layer has 100,480 parameters.
#    - The second dropout layer has no parameters.
#    - The third dense layer has 1,290 parameters.
#    - The model has a total of 101,770 parameters.
#    - The output also shows information about the size of tensors at each layer.

# 5. Note:
#    - The information message (I tensorflow/core/platform/cpu_feature_guard.cc:182) indicates that TensorFlow is optimized to use available CPU instructions.
#    - Suggests rebuilding TensorFlow with appropriate compilation flags to enable certain instructions.

# Documentation:
#   - Sequential Model: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
#   - Dense Layer: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
#   - Dropout Layer: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout
#   - Activation Functions: https://www.tensorflow.org/api_docs/python/tf/keras/activations
#   - Model Summary: https://www.tensorflow.org/api_docs/python/tf/keras/Model#summary