import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt

import scipy.stats as ss
from sklearn.model_selection import RandomizedSearchCV

from time import time


#######################
#  TASK 1
#######################
print("TASK 1 PART IS INITATED.")

print("\n")


# Loading train and test datasets
with np.load('train_data_label.npz') as data:
    train_data = data['train_data']
    train_label = data['train_label']
    
with np.load('test_data_label.npz') as data:
    test_data = data['test_data']
    test_label = data['test_label']

# Checking whether test and train data and labels have nan values.
print("NaN Values for train_data", np.isnan(train_data).any())
print("NaN Values for train_label",np.isnan(train_label).any())
print("NaN Values for test_data",np.isnan(test_data).any())
print("NaN Values for test_label",np.isnan(test_label).any())

print("\n")


train_data_unsplit = train_data.copy()
train_label_unsplit = train_label.copy()

# Splitting training data into training and validation datasets
train_data, val_data, train_label, val_label = train_test_split(
    train_data, train_label, test_size=0.20, random_state=54020)

## Random Forest Classifier
print("Random Forest Classifier Part Initiated.")
print("\n")


# Training model


start = time()
clf = RandomForestClassifier(max_depth=10, random_state=54020)
clf.fit(train_data, train_label)

print(f'Time taken to run: {time() - start} seconds')


# Running confusion matrix on validation dataset
print("Confusion Matrix on Validation Data")
print("\n")
print(confusion_matrix(val_label, clf.predict(val_data)))
print("\n")


# Testing accuracy using f1 score

print("Testing accuracy using f1 score")
print("\n")
print(f1_score(val_label, clf.predict(val_data), average='weighted'))
print("\n")


# Hyperparameter tuning using randomised search cross validation
print("Hyperparameter tuning using randomised search cross validation")
print("\n")

start = time()
features_dict = {
        "max_depth": [3, 5, 7, 10],
        "max_features": ss.randint(1, 20),
        "min_samples_split": ss.randint(2, 20),
    }

print("for HP, use both original training and validation together, for efficiency of the data.")
# for HP, use both original training and validation together, for efficiency of the data.

rs_p = RandomizedSearchCV(clf, features_dict, random_state=54020,verbose=2)
randomized_searched_model = rs_p.fit(train_data_unsplit, train_label_unsplit)
print(randomized_searched_model.best_params_)
print(f'Time taken to run: {time() - start} seconds')
print("\n")



# Optimal hyperparameters = {'max_depth': 10, 'max_features': 4, 'min_samples_split': 7}
# Fit model using new optimal hyperparameters

print("Optimal hyperparameters = {'max_depth': 10, 'max_features': 4, 'min_samples_split': 7}")
print("Fit model using new optimal hyperparameters")

start = time()
hyperparameter_optimized_clf = RandomForestClassifier(max_depth=10,max_features=4 ,min_samples_split=7,random_state=54020)
hyperparameter_optimized_clf.fit(train_data, train_label)

print("F1 Score for Validation Data: ",f1_score(val_label, hyperparameter_optimized_clf.predict(val_data), average='weighted'))

print(f'Time taken to run: {time() - start} seconds')

print("\n")


# Testing the accuracy of the updated model on unseen data

print("Testing the accuracy of the updated model on unseen data")
print("F1 Score of Updated Model for Test Data: ",f1_score(test_label, hyperparameter_optimized_clf.predict(test_data), average='weighted'))

print("\n")

# Testing the accuracy of the base model on unseen data
print("Testing the accuracy of the base model on unseen data")
print("F1 Score of Base Model for Test Data: ",f1_score(test_label, clf.predict(test_data), average='weighted'))

print("\n")


# Generating the confusion matrix (two of them for different uses, one array, and one dataframe).

print("Generating the confusion matrix (two of them for different uses, one array, and one dataframe).")
cm = confusion_matrix(test_label, clf.predict(test_data), labels=np.unique(test_label))
cm_dataframe = pd.DataFrame(cm, index=np.unique(test_label), columns=np.unique(test_label))
cm_dataframe.set_index(cm_dataframe.index.map(lambda x: chr(int(x+97))), inplace=True)
cm_dataframe.rename(columns=lambda x: chr(x+97), inplace=True)
print(cm_dataframe)

print("\n")


# Generating pretty graph for the confusion matrix.

#print("Generating pretty graph for the confusion matrix.")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_dataframe.index)
plt.rcParams["figure.figsize"] = (10,10)
disp.plot()
#plt.show()


# Metrics.

class_report = classification_report(test_label, clf.predict(test_data), target_names=cm_dataframe.index)
print(class_report)
print("\n")


# Easiest letters to classify (by F1 score).
print("Easiest letters to classify (by F1 score).")
precision, recall, fscore, support = precision_recall_fscore_support(test_label, clf.predict(test_data))
per_class_metrics = pd.DataFrame({'Letter':cm_dataframe.index, 'Precision':precision.round(2), 'Recall':recall.round(2), 'F1 score':fscore.round(2)})
per_class_metrics.set_index('Letter', inplace=True)
per_class_metrics.sort_values(by='F1 score', ascending=False, inplace=True)
print(per_class_metrics.head())

print("\n")


# Hardest letters to classify (by F1 score).
print("Hardest letters to classify (by F1 score).")
per_class_metrics.sort_values(by='F1 score', ascending=True, inplace=True)
print(per_class_metrics.head())

print("\n")



## Convolutional Neural Network
print("Convolutional Neural Network Part Initiated.")
print("\n")

import tensorflow as tf
np.random.seed(1)
tf.random.set_seed(2)

from tensorflow.keras import datasets, layers, models
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt


# Reshaping data for CNN

reshaped_train_data = np.array([np.reshape(i, (28, 28)) for i in train_data])
reshaped_test_data = np.array([np.reshape(i, (28, 28)) for i in test_data])
reshaped_val_data = np.array([np.reshape(i, (28, 28)) for i in val_data])

# Building CNN model

model = models.Sequential()
model.add(layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(Dropout(rate=0.3))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(Dropout(rate=0.3))
model.add(layers.Dense(25, activation = 'softmax'))

print(model.summary())

# Hyperparameter tuning

print("Hyperparameter Tuning")
print("\n")

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001
)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

start = time()
history = model.fit(reshaped_train_data, train_label, epochs=15, batch_size = 100,   
                    validation_data=(reshaped_val_data, val_label),callbacks=[
                                               tf.keras.callbacks.EarlyStopping(
                                                   monitor='val_loss',
                                                   patience=3,
                                                   restore_best_weights=True
                                                   ),learning_rate_reduction])

print(f'Time taken to run: {time() - start} seconds')


# Plotting validation accuracy

#print("Plotting validation accuracy")

plt.rcParams["figure.figsize"] = (5,5)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.05,1.05])
plt.legend(loc='lower right')
#plt.show()

print("Test Loss and Test Accuracy: ")
test_loss, test_acc = model.evaluate(reshaped_test_data,  test_label, verbose=2)
print(test_loss,test_acc)

print("\n")

predicted_test_labels = [np.argmax(i) for i in model.predict(reshaped_test_data)]


# Accuracy of model on unseen data
print("Accuracy of model on unseen data")
print(f1_score(test_label, predicted_test_labels, average='weighted'))

import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt


# Generating the confusion matrix (two of them for different uses, one array, and one dataframe).

print("Generating the confusion matrix (two of them for different uses, one array, and one dataframe).")

cm = confusion_matrix(test_label, predicted_test_labels, labels=np.unique(test_label))
cm_dataframe = pd.DataFrame(cm, index=np.unique(test_label), columns=np.unique(test_label))
cm_dataframe.set_index(cm_dataframe.index.map(lambda x: chr(int(x+97))), inplace=True)
cm_dataframe.rename(columns=lambda x: chr(x+97), inplace=True)
print(cm_dataframe)

print("\n")

# Generating pretty graph for the confusion matrix. Teacher from practicals said it was pretty and good idea to include in report.

#print("Generating pretty graph for the confusion matrix. Teacher from practicals said it was pretty and good idea to include in report.")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_dataframe.index)
plt.rcParams["figure.figsize"] = (10,10)
disp.plot()
#plt.show()
print("\n")

# Metrics.
class_report = classification_report(test_label, predicted_test_labels, target_names=cm_dataframe.index)
print(class_report)

print("\n")


# Easiest letters to classify (by bests F1 score).
print("Easiest letters to classify (by bests F1 score).")
precision, recall, fscore, support = precision_recall_fscore_support(test_label, predicted_test_labels)
per_class_metrics = pd.DataFrame({'Letter':cm_dataframe.index, 'Precision':precision.round(2), 'Recall':recall.round(2), 'F1 score':fscore.round(2)})
per_class_metrics.set_index('Letter', inplace=True)
per_class_metrics.sort_values(by='F1 score', ascending=False, inplace=True)
print(per_class_metrics.head())

print("\n")

# Hardest letters to classify (by worsts F1 score).
print("Hardest letters to classify (by worsts F1 score).")
per_class_metrics.sort_values(by='F1 score', ascending=True, inplace=True)
print(per_class_metrics.head())

print("\n")


#######################
#  TASK 2
#######################

print("TASK 2 PART IS INITATED.")
print("\n")

import numpy as np
import matplotlib.pyplot as plt


# Making dictionaries to create labels for images

task2_data_test = np.load('test_images_task2.npy')


task2_letter_labels = {0:"A",1:"B",2:"C",
                3:"D",4:"E",5:"F",
               6:"G",7:"H",8:"I",
                10:"K",11:"L",12:"M",
                13:"N",14:"O",15:"P",
                16:"Q",17:"R",18:"S",
               19:"T",20:"U",21:"V",
                22:"W",23:"X",24:"Y"}
task2_number_labels = {0:"00",1:"01",2:"02",
                3:"03",4:"04",5:"05",
               6:"06",7:"07",8:"08",
                10:"10",11:"11",12:"12",
                13:"13",14:"14",15:"15",
                16:"16",17:"17",18:"18",
               19:"19",20:"20",21:"21",
                22:"22",23:"23",24:"24"}


# Removing noise from images

from scipy import stats

def image_cropper(image):
    cropped_images = []
    for image_line in image.T:
        if list(stats.mode(image_line))[0] == 200:
            continue
        else:
            cropped_images.append(image_line)
    cropped_images = np.asarray(cropped_images).T
    return cropped_images



def manuel_slicer(data):
    pre_sliced_list = [data[:28][:,0:28],
                  data[:28][:,28:56],
                  data[:28][:,56:84],
                  data[:28][:,84:112],
                  data[:28][:,112:140] 
                  ]
    sliced_list = []
    for i in pre_sliced_list:
        if i.shape[1] == 0:
            continue
        else:
            if i.shape[1] != 28:
                gap = 28 - i.shape[1]
                i = np.hstack((i[:28] ,np.ones((28,gap), dtype=int)))
            sliced_list.append(i)
    return np.asarray(sliced_list)


def predict_top_5_images(images,model):
    indexed = model.predict(images).argsort()
    top_5_each = [i[-5:][::-1] for i in indexed]
    return list(zip(*top_5_each))

def convert_label_letter(predicted_labels):
    converted_top_5_labels = []
    for i in predicted_labels:
        str_label = "".join([task2_number_labels[j] for j in i])
        converted_top_5_labels.append(str_label)
    return converted_top_5_labels


# Classifying images in test set

ten_k_final_predictions = []
from tqdm import tqdm

print("Classifying images in test set")

for _,input_image in tqdm(enumerate(task2_data_test)):
    cropped_images = image_cropper(input_image)
    sliced_images = manuel_slicer(cropped_images)
    predicted_top_5_images = predict_top_5_images(sliced_images,model)
    ten_k_final_predictions.append(convert_label_letter(predicted_top_5_images))

result = pd.DataFrame(np.asarray(ten_k_final_predictions))

result.to_csv('result.csv')

print("First Five Rows of Result")
print(result.head())