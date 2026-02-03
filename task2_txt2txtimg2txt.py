#%% md
# # Introduction to Deep Learning, Assignment 2, Task 2
# 
# # Introduction
# 
# 
# The goal of this assignment is to learn how to use encoder-decoder recurrent neural networks (RNNs). Specifically we will be dealing with a sequence to sequence problem and try to build recurrent models that can learn the principles behind simple arithmetic operations (**integer addition, subtraction and multiplication.**).
# 
# <img src="https://i.ibb.co/5Ky5pbk/Screenshot-2023-11-10-at-07-51-21.png" alt="Screenshot-2023-11-10-at-07-51-21" border="0" width="500"></a>
# 
# In this assignment you will be working with three different kinds of models, based on input/output data modalities:
# 1. **Text-to-text**: given a text query containing two integers and an operand between them (+ or -) the model's output should be a sequence of integers that match the actual arithmetic result of this operation
# 2. **Image-to-text**: same as above, except the query is specified as a sequence of images containing individual digits and an operand.
# 3. **Text-to-image**: the query is specified in text format as in the text-to-text model, however the model's output should be a sequence of images corresponding to the correct result.
# 
# 
# ### Description
# Let us suppose that we want to develop a neural network that learns how to add or subtract
# two integers that are at most two digits long. For example, given input strings of 5 characters: ‘81+24’ or
# ’41-89’ that consist of 2 two-digit long integers and an operand between them, the network should return a
# sequence of 3 characters: ‘105 ’ or ’-48 ’ that represent the result of their respective queries. Additionally,
# we want to build a model that generalizes well - if the network can extract the underlying principles behind
# the ’+’ and ’-’ operands and associated operations, it should not need too many training examples to generate
# valid answers to unseen queries. To represent such queries we need 13 unique characters: 10 for digits (0-9),
# 2 for the ’+’ and ’-’ operands and one for whitespaces ’ ’ used as padding.
# The example above describes a text-to-text sequence mapping scenario. However, we can also use different
# modalities of data to represent our queries or answers. For that purpose, the MNIST handwritten digit
# dataset is going to be used again, however in a slightly different format. The functions below will be used to create our datasets.
# 
# ---
# 
# *To work on this notebook you should create a copy of it.*
# 
# When using the Lab Computers, download the Jupyter Notebook to one of the machines first.
# 
# If you want to use Google Colab, you should first copy this notebook and enable GPU runtime in 'Runtime -> Change runtime type -> Hardware acceleration -> GPU **OR** TPU'.
# 
#%% md
# # Function definitions for creating the datasets
# 
# First we need to create our datasets that are going to be used for training our models.
# 
# In order to create image queries of simple arithmetic operations such as '15+13' or '42-10' we need to create images of '+' and '-' signs using ***open-cv*** library. We will use these operand signs together with the MNIST dataset to represent the digits.
#%%
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
import random
from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Dense, RNN, LSTM, Flatten, TimeDistributed, LSTMCell
from tensorflow.keras.layers import RepeatVector, Conv2D, SimpleRNN, GRU, Reshape, ConvLSTM2D, Conv2DTranspose
tf.config.list_physical_devices('GPU')
#%%
from scipy.ndimage import rotate


# Create plus/minus operand signs
def generate_images(number_of_images=50, sign='-'):
    blank_images = np.zeros([number_of_images, 28, 28])  # Dimensionality matches the size of MNIST images (28x28)
    x = np.random.randint(12, 16, (number_of_images, 2)) # Randomized x coordinates
    y1 = np.random.randint(6, 10, number_of_images)       # Randomized y coordinates
    y2 = np.random.randint(18, 22, number_of_images)     # -||-

    for i in range(number_of_images): # Generate n different images
        cv2.line(blank_images[i], (y1[i], x[i,0]), (y2[i], x[i, 1]), (255,0,0), 2, cv2.LINE_AA)     # Draw lines with randomized coordinates
        if sign == '+':
            cv2.line(blank_images[i], (x[i,0], y1[i]), (x[i, 1], y2[i]), (255,0,0), 2, cv2.LINE_AA) # Draw lines with randomized coordinates

    return blank_images

def show_generated(images, n=5):
    plt.figure(figsize=(2, 2))
    for i in range(n**2):
        plt.subplot(n, n, i+1)
        plt.axis('off')
        plt.imshow(images[i])
    plt.show()

show_generated(generate_images())
show_generated(generate_images(sign='+'))
#%%
def create_data(highest_integer, num_addends=2, operands=['+', '-']):
    """
    Creates the following data for all pairs of integers up to [1:highest integer][+/-][1:highest_integer]:

    @return:
    X_text: '51+21' -> text query of an arithmetic operation (5)
    X_img : Stack of MNIST images corresponding to the query (5 x 28 x 28) -> sequence of 5 images of size 28x28
    y_text: '72' -> answer of the arithmetic text query
    y_img :  Stack of MNIST images corresponding to the answer (3 x 28 x 28)

    Images for digits are picked randomly from the whole MNIST dataset.
    """

    num_indices = [np.where(MNIST_labels==x) for x in range(10)]
    num_data = [MNIST_data[inds] for inds in num_indices]
    image_mapping = dict(zip(unique_characters[:10], num_data))
    image_mapping['-'] = generate_images()
    image_mapping['+'] = generate_images(sign='+')
    image_mapping['*'] = generate_images(sign='*')
    image_mapping[' '] = np.zeros([1, 28, 28])

    X_text, X_img, y_text, y_img = [], [], [], []

    for i in range(highest_integer + 1):      # First addend
        for j in range(highest_integer + 1):  # Second addend
            for sign in operands: # Create all possible combinations of operands
                query_string = to_padded_chars(str(i) + sign + str(j), max_len=max_query_length, pad_right=True)
                query_image = []
                for n, char in enumerate(query_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    query_image.append(image_set[index].squeeze())

                result = eval(query_string)
                result_string = to_padded_chars(result, max_len=max_answer_length, pad_right=True)
                result_image = []
                for n, char in enumerate(result_string):
                    image_set = image_mapping[char]
                    index = np.random.randint(0, len(image_set), 1)
                    result_image.append(image_set[index].squeeze())

                X_text.append(query_string)
                X_img.append(np.stack(query_image))
                y_text.append(result_string)
                y_img.append(np.stack(result_image))

    return np.stack(X_text), np.stack(X_img)/255., np.stack(y_text), np.stack(y_img)/255.

def to_padded_chars(integer, max_len=3, pad_right=False):
    """
    Returns a string of len()=max_len, containing the integer padded with ' ' on either right or left side
    """
    length = len(str(integer))
    padding = (max_len - length) * ' '
    if pad_right:
        return str(integer) + padding
    else:
        return padding + str(integer)

#%% md
# # Creating our data
# 
# The dataset consists of 20000 samples that (additions and subtractions between all 2-digit integers) and they have two kinds of inputs and label modalities:
# 
#   **X_text**: strings containing queries of length 5: ['  1+1  ', '11-18', ...]
# 
#   **X_image**: a stack of images representing a single query, dimensions: [5, 28, 28]
# 
#   **y_text**: strings containing answers of length 3: ['  2', '156']
# 
#   **y_image**: a stack of images that represents the answer to a query, dimensions: [3, 28, 28]
#%%
# Illustrate the generated query/answer pairs

unique_characters = '0123456789+- '       # All unique characters that are used in the queries (13 in total: digits 0-9, 2 operands [+, -], and a space character ' '.)
highest_integer = 99                      # Highest value of integers contained in the queries

max_int_length = len(str(highest_integer))# Maximum number of characters in an integer
max_query_length = max_int_length * 2 + 1 # Maximum length of the query string (consists of two integers and an operand [e.g. '22+10'])
max_answer_length = 3    # Maximum length of the answer string (the longest resulting query string is ' 1-99'='-98')

# Create the data (might take around a minute)
(MNIST_data, MNIST_labels), _ = tf.keras.datasets.mnist.load_data()
X_text, X_img, y_text, y_img = create_data(highest_integer)
print(X_text.shape, X_img.shape, y_text.shape, y_img.shape)


## Display the samples that were created
def display_sample(n):
    labels = ['X_img:', 'y_img:']
    for i, data in enumerate([X_img, y_img]):
        plt.subplot(1,2,i+1)
        # plt.set_figheight(15)
        plt.axis('off')
        plt.title(labels[i])
        plt.imshow(np.hstack(data[n]), cmap='gray')
    print('='*50, f'\nQuery #{n}\n\nX_text: "{X_text[n]}" = y_text: "{y_text[n]}"')
    plt.show()

for _ in range(10):
    display_sample(np.random.randint(0, 10000, 1)[0])
#%% md
# ## Helper functions
# 
# The functions below will help with input/output of the data.
#%%
# One-hot encoding/decoding the text queries/answers so that they can be processed using RNNs
# You should use these functions to convert your strings and read out the output of your networks

def encode_labels(labels, max_len=3):
  n = len(labels)
  length = len(labels[0])
  char_map = dict(zip(unique_characters, range(len(unique_characters))))
  one_hot = np.zeros([n, length, len(unique_characters)])
  for i, label in enumerate(labels):
      m = np.zeros([length, len(unique_characters)])
      for j, char in enumerate(label):
          m[j, char_map[char]] = 1
      one_hot[i] = m

  return one_hot


def decode_labels(labels):
    pred = np.argmax(labels, axis=2)
    predicted = [''.join([unique_characters[i] for i in j]) for j in pred]

    return predicted

X_text_onehot = encode_labels(X_text)
y_text_onehot = encode_labels(y_text)

print(X_text_onehot.shape, y_text_onehot.shape)
#%% md
# ---
# ---
# 
# ## I. Text-to-text RNN model
# 
# The following code showcases how Recurrent Neural Networks (RNNs) are built using Keras. Several new layers are going to be used:
# 
# 1. LSTM
# 2. TimeDistributed
# 3. RepeatVector
# 
# The code cell below explains each of these new components.
# 
# <img src="https://i.ibb.co/NY7FFTc/Screenshot-2023-11-10-at-09-27-25.png" alt="Screenshot-2023-11-10-at-09-27-25" border="0" width="500"></a>
# 
#%%
def build_text2text_model():

    # We start by initializing a sequential model
    text2text = tf.keras.Sequential()

    # "Encode" the input sequence using an RNN, producing an output of size 256.
    # In this case the size of our input vectors is [5, 13] as we have queries of length 5 and 13 unique characters. Each of these 5 elements in the query will be fed to the network one by one,
    # as shown in the image above (except with 5 elements).
    # Hint: In other applications, where your input sequences have a variable length (e.g. sentences), you would use input_shape=(None, unique_characters).
    text2text.add(LSTM(256, input_shape=(None, len(unique_characters))))

    # As the decoder RNN's input, repeatedly provide with the last output of RNN for each time step. Repeat 3 times as that's the maximum length of the output (e.g. '  1-99' = '-98')
    # when using 2-digit integers in queries. In other words, the RNN will always produce 3 characters as its output.
    text2text.add(RepeatVector(max_answer_length))

    # By setting return_sequences to True, return not only the last output but all the outputs so far in the form of (num_samples, timesteps, output_dim). This is necessary as TimeDistributed in the below expects
    # the first dimension to be the timesteps.
    text2text.add(LSTM(256, return_sequences=True))

    # Apply a dense layer to the every temporal slice of an input. For each of step of the output sequence, decide which character should be chosen.
    text2text.add(TimeDistributed(Dense(len(unique_characters), activation='softmax')))

    # Next we compile the model using categorical crossentropy as our loss function.
    text2text.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    text2text.summary()

    return text2text
#%%
## Your code (look at the assignment description for your tasks for text-to-text model):
## (Your first task is to fit the text2text model using X_text and y_text)
def analyze_errors(X_test_text, y_true_text, y_pred_text, max_errors_to_show=10):
    """
    Compares true and predicted labels to find and analyze errors, 
    printing a sample of misclassified results.
    """
    error_samples = []
    
    X_test_text = np.array(X_test_text)
    y_true_text = np.array(y_true_text)
    y_pred_text = np.array(y_pred_text)
    
    # Find indices where the prediction does NOT match the true label (padded strings)
    mismatched_indices = np.where(y_true_text != y_pred_text)[0]
    
    print(f"\n--- Sample Error Analysis ({len(mismatched_indices)} total errors) ---")
    
    for i in mismatched_indices[:max_errors_to_show]:
        query = X_test_text[i]
        true_ans = y_true_text[i]
        pred_ans = y_pred_text[i]
        
        error_samples.append((query, true_ans, pred_ans))
        print(f"Query: '{query}' | True: '{true_ans}' | Predicted: '{pred_ans}'")
        
    return error_samples

def run_text2text_experiment(X_text_all, y_text_all, splits, epochs=50):
    """
    Runs the experiment for different train/test splits using the provided text data.
    """
    
    # Use the provided encode_labels function for both input and target data
    X_all = encode_labels(X_text_all, max_len=5)
    y_all = encode_labels(y_text_all, max_len=3)
    
    results = {}
    
    for train_ratio in splits:
        split_name = f"{int(train_ratio * 100)}% Train / {int((1 - train_ratio) * 100)}% Test"
        print(f"\n--- starting Experiment for: {split_name} (Epochs: {epochs}) ---")
        
        # Split the data, preserving text versions for error analysis
        X_train, X_test, y_train, y_test, X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(
            X_all, y_all, X_text_all, y_text_all, 
            train_size=train_ratio, 
            random_state=42 
        )
        
        print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
        
        # Build and train the model
        model = build_text2text_model()
        history = model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=epochs,
            validation_split=0.1, 
            verbose=1
        )
        
        # Evaluate on the test set
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"Test Accuracy for {split_name}: {accuracy:.4f}")
        
        # Store results
        results[split_name] = {
            'accuracy': accuracy, 
            'history': history.history,
            'error_analysis': None
        }
        
        # --- error analysis ---
        y_pred_one_hot = model.predict(X_test, verbose=0)
        
        # Use the provided decode_labels function for decoding predictions
        y_pred_text = decode_labels(y_pred_one_hot) 
        
        error_samples = analyze_errors(X_text_test, y_text_test, y_pred_text, max_errors_to_show=10)
        results[split_name]['error_analysis'] = error_samples
        
        tf.keras.backend.clear_session()

    return results

#%%
TRAIN_SPLITS = [0.95, 0.90, 0.75, 0.50, 0.25, 0.10, 0.05] 
EPOCHS_TO_RUN = 100
experiment_results = run_text2text_experiment(X_text, y_text, TRAIN_SPLITS, epochs=EPOCHS_TO_RUN)
#%%
def visualize_results(experiment_results, epochs):
    """Plots accuracy and prints error interpretation."""
    
    print("\n\n####################################")
    print("#### Final Experiment Summary ####")
    print("####################################\n")

    splits = list(experiment_results.keys())
    accuracies = [results['accuracy'] for results in experiment_results.values()]

    # Plotting the generalization accuracy 
    plt.figure(figsize=(20, 8))
    plt.bar(splits, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'purple', 'green'])
    plt.xlabel("Train/Test Split Configuration")
    plt.ylabel("Test Accuracy (Generalization)")
    # plt.title(f"Generalization Capability of Text-to-Text Model ({epochs} Epochs)")
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center', fontweight='bold')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--')
    plt.show()

    # --- Interpretation of Mistakes ---
    for split, res in experiment_results.items():
        print(f"### Results for {split}")
        errors = res['error_analysis']
        
        if not errors:
            print("* Model achieved 100% accuracy on the sampled test set.")
            continue

        off_by_one_count = 0
        sign_error_count = 0
        
        for query, true_ans_padded, pred_ans_padded in errors:
            try:
                # Strip spaces for numerical comparison
                true_val = int(true_ans_padded.strip())
                pred_val = int(pred_ans_padded.strip())
                
                # Check for off-by-one (indicative of carry/borrow failure)
                if abs(true_val - pred_val) == 1:
                    off_by_one_count += 1
                    
                # Check for sign errors (e.g., -10 predicted as 10)
                if true_val * pred_val < 0 and abs(true_val) == abs(pred_val):
                     sign_error_count += 1
                     
            except ValueError:
                # Ignore errors that aren't cleanly convertible to numbers
                pass
                
        
        print(f"* **Total Errors (on sample):** {len(errors)}")
        print("---")

visualize_results(experiment_results, EPOCHS_TO_RUN)
#%%
def plot_training_history(experiment_results):
    """
    Plots the training and validation loss for all model splits, 
    allowing for comparison of learning over time.
    """
    num_splits = len(experiment_results)
    
    # Determine grid size for subplots
    cols = 3
    rows = int(np.ceil(num_splits / cols))
    
    plt.figure(figsize=(cols * 5, rows * 4), dpi=100)
    plt.suptitle("Training and Validation Loss for Different Train/Test Splits", fontsize=16, y=1.02)

    for i, (split_name, res) in enumerate(experiment_results.items()):            
        history = res['history']
        
        plt.subplot(rows, cols, i + 1)
        
        # Plot Loss
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        
        # Add Title with Final Accuracy
        acc = res.get('accuracy', np.nan) 
        title_text = f"Split: {split_name}"
        if not np.isnan(acc):
            title_text += f"\n(Final Acc: {acc:.4f})"
            
        plt.title(title_text, fontsize=10)
        plt.ylabel('Categorical Crossentropy Loss')
        plt.xlabel('Epoch')
        
        # Dynamic Y-axis limit based on the maximum loss in the subplot
        max_loss = 0
        if history['loss']: max_loss = max(max_loss, max(history['loss']))
        if history['val_loss']: max_loss = max(max_loss, max(history['val_loss']))
        
        plt.ylim(0, max_loss + 0.1) 
        plt.legend()
        plt.grid(axis='y', linestyle='--')
        
    plt.tight_layout()
    plt.savefig('training_history_comparison.png')
    print("training_history_comparison.png")


plot_training_history(experiment_results)
#%%
def plot_accuracy_history(experiment_results):
    """
    Plots the training and validation accuracy for all model splits, 
    allowing for comparison of learning over time.
    """
    num_splits = len(experiment_results)
    
    # Determine grid size for subplots
    cols = 3
    rows = int(np.ceil(num_splits / cols))
    
    plt.figure(figsize=(cols * 5, rows * 4), dpi=100)
    plt.suptitle("Training and Validation Accuracy for Different Train/Test Splits", fontsize=16, y=1.02)

    for i, (split_name, res) in enumerate(experiment_results.items()):
        
        # Ensure the history object is present and contains accuracy data
        if 'history' not in res or 'accuracy' not in res['history']:
            # Skip splits where data wasn't recorded correctly
            print(f"Skipping {split_name}: Accuracy history not found.")
            continue
            
        history = res['history']
        
        plt.subplot(rows, cols, i + 1)
        
        # Plot Accuracy
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Val Accuracy')
        
        # Add Title with Final Test Accuracy
        test_acc = res.get('accuracy', np.nan) 
        title_text = f"Split: {split_name}"
        if not np.isnan(test_acc):
            title_text += f"\n(Final Test Acc: {test_acc:.4f})"
            
        plt.title(title_text, fontsize=10)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        
        # Y-axis limit for accuracy plots
        plt.ylim(0, 1.05) 
        plt.legend()
        plt.grid(axis='y', linestyle='--')
        
    plt.tight_layout()
    plt.show()
plot_accuracy_history(experiment_results)
#%%
def plot_comparison_histories(experiment_results):
    """
    Generates two consolidated plots: one for loss history and one for accuracy history,
    with separate colors and styles for each train/test split.
    """
    
    # Ensure splits are sorted from largest to smallest for logical color progression
    sorted_results = dict(sorted(experiment_results.items(), key=lambda item: float(item[0].split('%')[0]), reverse=True))

    cmap = plt.colormaps.get_cmap('Spectral')
    num_splits = len(sorted_results)
    colors = [cmap(j / (num_splits - 1)) for j in range(num_splits)]
    line_styles = ['-', '--'] # Solid for Train, Dashed for Val

    
    fig_loss, ax_loss = plt.subplots(figsize=(14, 8))
    ax_loss.set_title("Training and Validation Loss Comparison by Split", fontsize=16)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Categorical Crossentropy Loss")
    ax_loss.grid(True, linestyle='--')
    
    fig_acc, ax_acc = plt.subplots(figsize=(14, 8))
    ax_acc.set_title("Training and Validation Accuracy Comparison by Split", fontsize=16)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.grid(True, linestyle='--')
    
    max_loss_limit = 0
    
    for i, (split_name, res) in enumerate(sorted_results.items()):
        
        if 'history' not in res:
            continue
            
        history = res['history']
        color = colors[i]
        
        # --- Loss Plotting ---
        # Train Loss (Solid line)
        ax_loss.plot(history['loss'], 
                     label=f'{split_name} - Train Loss', 
                     color=color, 
                     linestyle=line_styles[0], 
                     alpha=0.8)
        
        # Validation Loss (Dashed line)
        ax_loss.plot(history['val_loss'], 
                     label=f'{split_name} - Val Loss', 
                     color=color, 
                     linestyle=line_styles[1], 
                     alpha=0.8)
        
        max_loss_limit = max(max_loss_limit, max(history['loss']), max(history['val_loss']))
        
        # --- Accuracy Plotting ---
        # Train Accuracy (Solid line)
        ax_acc.plot(history['accuracy'], 
                    label=f'{split_name} - Train Accuracy', 
                    color=color, 
                    linestyle=line_styles[0], 
                    alpha=0.8)
        
        # Validation Accuracy (Dashed line)
        ax_acc.plot(history['val_accuracy'], 
                    label=f'{split_name} - Val Accuracy', 
                    color=color, 
                    linestyle=line_styles[1], 
                    alpha=0.8)

    # Finalize Loss Plot
    ax_loss.set_ylim(0, max_loss_limit * 1.05)
    # Move legend outside the plot area
    ax_loss.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8) 
    fig_loss.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()
    plt.close(fig_loss) # Close figure to avoid double display

    # Finalize Accuracy Plot
    ax_acc.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig_acc.tight_layout(rect=[0, 0, 0.85, 1])
    plt.close(fig_acc) # Close figure to avoid double display

    # Save and display both plots
    fig_loss.savefig('comparison_loss_history.png')
    fig_acc.savefig('comparison_accuracy_history.png')
    
    
plot_comparison_histories(experiment_results)
#%% md
# 
# ---
# ---
# 
# ## II. Image to text RNN Model
# 
# Hint: There are two ways of building the encoder for such a model - again by using the regular LSTM cells (with flattened images as input vectors) or recurrect convolutional layers [ConvLSTM2D](https://keras.io/api/layers/recurrent_layers/conv_lstm2d/).
# 
# The goal here is to use **X_img** as inputs and **y_text** as outputs.
#%%
# data pipeline, to prevent overfitting
y_text_onehot = encode_labels(y_text, max_len=3)
X_train, X_test, y_train, y_test = train_test_split(
    X_img,
    y_text_onehot,
    test_size=0.2,
    random_state=69,
    shuffle=True
)
batch_size = 64
# Reshape to (Batch, Sequence, Height, Width, Channels)

X_train = np.expand_dims(X_train, -1)
X_test  = np.expand_dims(X_test, -1)

print("New shape:", X_train.shape)
#%%
# y_text_onehot = encode_labels(y_text, max_len=3)
# X_train, X_test, y_train, y_test = train_test_split(
#     X_img,
#     y_text_onehot,
#     test_size=0.75,
#     random_state=69,
#     shuffle=True
# )
# batch_size = 64

#%%
import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm_img2text_model_augmented():
    # Input shape: (Sequence length, Height, Width)
    inputs = tf.keras.Input(shape=(5, 28, 28))

    x = layers.Reshape((5, 28, 28, 1))(inputs)
    
    x = layers.TimeDistributed(layers.Rescaling(1./255))(x)
    x = layers.TimeDistributed(layers.RandomRotation(0.05, fill_mode='constant', fill_value=0))(x)
    x = layers.TimeDistributed(layers.RandomZoom(0.1, fill_mode='constant', fill_value=0))(x)
    x = layers.TimeDistributed(layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='constant', fill_value=0))(x)
    

    x = layers.Reshape((5, 28, 28))(x)
    
    # Flatten each image vector of size 784
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Flatten()
    )(x)

    # Encoder LSTM
    x = tf.keras.layers.LSTM(256)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Repeat for decoder timesteps
    x = tf.keras.layers.RepeatVector(max_answer_length)(x)

    # Decoder LSTM
    x = tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.4, recurrent_dropout=0.3)(x)


    # Output per timestep
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(len(unique_characters), activation="softmax")
    )(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    model.summary()
    return model
#%%
import tensorflow as tf
from tensorflow.keras import layers, models

def build_conv_img2text_model_augmented():
    # Input shape: (Sequence length, Height, Width, Channels)
    inputs = tf.keras.Input(shape=(5, 28, 28, 1))

    # Apply augmentation to each of the 5 images independently
    x = layers.TimeDistributed(layers.Rescaling(1./255))(inputs)
    
    x = layers.TimeDistributed(layers.RandomRotation(0.05, fill_mode='constant', fill_value=0))(x)
    x = layers.TimeDistributed(layers.RandomZoom(0.1, fill_mode='constant', fill_value=0))(x)
    x = layers.TimeDistributed(layers.RandomTranslation(height_factor=0.1, width_factor=0.1, fill_mode='constant', fill_value=0))(x)
    
    x = tf.keras.layers.ConvLSTM2D(
        filters=32, 
        kernel_size=(3,3),
        padding="same",
        return_sequences=False,
        dropout=0.3,
        recurrent_dropout=0.3
    )(x)

    x = tf.keras.layers.Flatten()(x)
    
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Repeat encoded vector for decoder timesteps
    x = tf.keras.layers.RepeatVector(max_answer_length)(x)

    # Decoder LSTM
    x = tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.4, recurrent_dropout=0.3)(x)

    # Output per timestep
    outputs = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(len(unique_characters), activation="softmax")
    )(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    model.summary()
    return model
#%%
img2txtModel = build_lstm_img2text_model_augmented()
history_lstm = img2txtModel.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=64
)
#%%
conv2dLSTM = build_conv_img2text_model_augmented()
conv2dLSTMHistory = conv2dLSTM.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=64
)
#%%
colors = {'LSTM': 'blue', 'ConvLSTM': 'orange'}

# ----------------------------
# Plot Accuracy
# ----------------------------
plt.figure(figsize=(10,5))
# LSTM
plt.plot(history_lstm.history['accuracy'], label='LSTM Train', color=colors['LSTM'], linestyle='-')
plt.plot(history_lstm.history['val_accuracy'], label='LSTM Val', color=colors['LSTM'], linestyle='--')
# ConvLSTM
plt.plot(conv2dLSTMHistory.history['accuracy'], label='ConvLSTM Train', color=colors['ConvLSTM'], linestyle='-')
plt.plot(conv2dLSTMHistory.history['val_accuracy'], label='ConvLSTM Val', color=colors['ConvLSTM'], linestyle='--')

plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# Plot Loss
# ----------------------------
plt.figure(figsize=(10,5))
# LSTM
plt.plot(history_lstm.history['loss'], label='LSTM Train', color=colors['LSTM'], linestyle='-')
plt.plot(history_lstm.history['val_loss'], label='LSTM Val', color=colors['LSTM'], linestyle='--')
# ConvLSTM
plt.plot(conv2dLSTMHistory.history['loss'], label='ConvLSTM Train', color=colors['ConvLSTM'], linestyle='-')
plt.plot(conv2dLSTMHistory.history['val_loss'], label='ConvLSTM Val', color=colors['ConvLSTM'], linestyle='--')

plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

#%%
continuation_epochs = 250

conv2dLSTMHistory_continued = conv2dLSTM.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=continuation_epochs,
    initial_epoch=conv2dLSTMHistory.epoch[-1] + 1,  # Start the epoch count from where you left off (100 + 1)
    batch_size=64
)
#%%
colors = {'LSTM': 'blue', 'ConvLSTM': 'orange'}

convlstm_total_acc = conv2dLSTMHistory.history['accuracy'] + conv2dLSTMHistory_continued.history['accuracy']
convlstm_total_val_acc = conv2dLSTMHistory.history['val_accuracy'] + conv2dLSTMHistory_continued.history['val_accuracy']

convlstm_total_loss = conv2dLSTMHistory.history['loss'] + conv2dLSTMHistory_continued.history['loss']
convlstm_total_val_loss = conv2dLSTMHistory.history['val_loss'] + conv2dLSTMHistory_continued.history['val_loss']



# ----------------------------
# Plot Accuracy
# ----------------------------
plt.figure(figsize=(10,5))
# LSTM (Original, unchanged)
plt.plot(history_lstm.history['accuracy'], label='LSTM Train', color=colors['LSTM'], linestyle='-')
plt.plot(history_lstm.history['val_accuracy'], label='LSTM Val', color=colors['LSTM'], linestyle='--')
# ConvLSTM (Merged data)
plt.plot(convlstm_total_acc, label='ConvLSTM Train', color=colors['ConvLSTM'], linestyle='-')
plt.plot(convlstm_total_val_acc, label='ConvLSTM Val', color=colors['ConvLSTM'], linestyle='--')

plt.title('Training and Validation Accuracy (Extended)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# Plot Loss
# ----------------------------
plt.figure(figsize=(10,5))
# LSTM (Original, unchanged)
plt.plot(history_lstm.history['loss'], label='LSTM Train', color=colors['LSTM'], linestyle='-')
plt.plot(history_lstm.history['val_loss'], label='LSTM Val', color=colors['LSTM'], linestyle='--')
# ConvLSTM (Merged data)
plt.plot(convlstm_total_loss, label='ConvLSTM Train', color=colors['ConvLSTM'], linestyle='-')
plt.plot(convlstm_total_val_loss, label='ConvLSTM Val', color=colors['ConvLSTM'], linestyle='--')

plt.title('Training and Validation Loss (Extended)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
#%%
continuation_epochs = 1000

conv2dLSTMHistory_continued_further = conv2dLSTM.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=continuation_epochs,
    initial_epoch=conv2dLSTMHistory.epoch[-1] + 1,  # Start the epoch count from where you left off (100 + 1)
    batch_size=64
)
#%% md
# ---
# ---
# 
# ## III. Text to image RNN Model
# 
# Hint: to make this model work really well you could use deconvolutional layers in your decoder (you might need to look up ***Conv2DTranspose*** layer). However, regular vector-based decoder will work as well.
# 
# The goal here is to use **X_text** as inputs and **y_img** as outputs.
#%%
# Your code




