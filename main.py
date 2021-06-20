import pandas as pd
import nltk
from tensorflow import keras
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
import numpy as np
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow import keras
from matplotlib import pyplot as plt
import string
from sklearn.feature_extraction.text import TfidfVectorizer
'''
# these are needed once for the code to run successfully
nltk.download('stopwords')
nltk.download('punkt')
'''


# simple identity tokenizer to use in TfidfVectorizer, as we have already pre-processed the input on our own
def identity_tokenizer(text):
    return text


# f1 score
def f1_score(precision, recall):
    precision_ = np.array(precision)
    recall_ = np.array(recall)
    # f1_score = 2 * (precision * recall) / (precision + recall), add epsilon to avoid 0/0 division just in case
    return 2 * (precision_ * recall_) / (precision_ + recall_ + K.epsilon())


# 4 splits means 75%-25% dataset split
kfold_splits = 4
epochs = 40
batchsize = 64
# bias regularizer
r1 = 1e-3
# kernel regularizer
r2 = 1e-1
# activity regularizer
r3 = 1e-3

csv_file = 'spam_or_not_spam.csv'
dataframe = pd.read_csv(csv_file, header=[0])

# preprocessing
dataframe = pd.DataFrame(dataframe)
# remove rows where email is nan and fix the indexing
dataframe.dropna(subset=['email'], inplace=True)
# and fix the indices
dataframe.reset_index(inplace=True)

# get the raw text and begin preprocessing
rawtext = dataframe['email']

dataset = []
temp_data = []
# the stop words, used in filtering out the stop words in the loop
stop_words = set(nltk.corpus.stopwords.words("english"))
# porter stemmer used for normalization of words later on
stemmer = nltk.stem.PorterStemmer()
# punctuation set for the preprocessing
punctuation = set(string.punctuation)
# add the newline to the punctuation set
punctuation.add('\n')
print("Starting preprocessing . . .")
for i in range(len(rawtext)):
    temp_data = []
    # tokenize the line
    temp_data = nltk.tokenize.word_tokenize(rawtext[i])
    # then remove the stopwords
    temp_data = [w for w in temp_data if w not in stop_words]
    # use  porter stemmer now to streamline/normalize the words
    temp_data = [stemmer.stem(word) for word in temp_data]
    # remove punctuation and newlines
    temp_data = [w for w in temp_data if w not in punctuation]
    # convert to lowercase
    temp_data = [t.lower() for t in temp_data]
    # finally append the processed data
    dataset.append(temp_data)
print('Finished pre-processing, proceeding to transform the input dataset . . .')
# transform the dataset
print(type(dataset))
dataset = [list(dataset[i]) for i in range(len(dataset))]
print(type(dataset))
# after appending everything convert to tf-idf matrix
tfidf_vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
input_ = tfidf_vectorizer.fit_transform(dataset).toarray()
# generate the finaldataset
# this matrix is 1499x17574
output_ = np.array(dataframe['label'])
print("Pre-processing finished.")

# Split the data to training and testing data 4-Fold aka 75-25
kfold = KFold(n_splits=kfold_splits, shuffle=True)

# for use on the plots:
history_list = []
# for model evaluation
scores_list = []
# f1 history list
f1_history_list = []
f1_score_list = []

input_shape = len(input_[0])
for i, (train, test) in enumerate(kfold.split(input_)):
    # setup the neural network layers
    model = keras.Sequential([
        # hidden layers, use relu
        keras.layers.Dense(units=15, activation='relu', input_shape=(input_shape,)),
        keras.layers.Dense(units=20, activation='relu', bias_regularizer=l2(r1), kernel_regularizer=l2(r2), activity_regularizer=l2(r3)),
        keras.layers.Dense(units=20, activation='relu', bias_regularizer=l2(r1), kernel_regularizer=l2(r2), activity_regularizer=l2(r3)),
        # output layer, binary classification hence we use sigmoid
        keras.layers.Dense(units=1, activation='sigmoid')
    ])

    # compile and fit
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    history = model.fit(input_[train], output_[train], epochs=epochs, batch_size=batchsize, verbose=0, validation_data=(input_[test], output_[test]))

    # f1 score calculation workaround
    if i == 0:
        f1_history = f1_score(history.history['precision'], history.history['recall'])
    else:
        f1_history = f1_score(history.history[f'precision_{i}'], history.history[f'recall_{i}'])

    if i == 0:
        f1_eval = f1_score(history.history['val_precision'], history.history['val_recall'])
    else:
        f1_eval = f1_score(history.history[f'val_precision_{i}'], history.history[f'val_recall_{i}'])

    # append the data for plots
    f1_history_list.append(f1_history)
    f1_history_list.append(f1_eval)
    # history for plots
    history_list.append(history)
    # score for verbose info
    scores = model.evaluate(input_[test], output_[test], verbose=0)
    scores_list.append(scores)

    # print verbose info
    for i in range(1, len(model.metrics_names)):
        print(f'{model.metrics_names[i]} {scores[i] * 100}')
    # calculate the evaluation f1 score
    f1_score_ = 2 * (scores[1] * scores[2] / (scores[1] + scores[2] + K.epsilon()))
    print(f'f1 score {f1_score_*100}')
    f1_score_list.append(f1_score_)

# after everything finishes plot the average of the folds as well as the best model
precision_history = []
val_precision_history = []
recall_history = []
val_recall_history = []
f1_score_history = []
val_f1_score_history = []

# store the data more neatly
for i in range(kfold_splits):
    if i == 0:
        precision_history.append(history_list[i].history['precision'])
        val_precision_history.append(history_list[i].history['val_precision'])
        recall_history.append(history_list[i].history['recall'])
        val_recall_history.append(history_list[i].history['val_recall'])
        f1_score_history.append(f1_history_list[2 * i])
        val_f1_score_history.append(f1_history_list[2 * i + 1])
    else:
        precision_history.append(history_list[i].history[f'precision_{i}'])
        val_precision_history.append(history_list[i].history[f'val_precision_{i}'])
        recall_history.append(history_list[i].history[f'recall_{i}'])
        val_recall_history.append(history_list[i].history[f'val_recall_{i}'])
        f1_score_history.append(f1_history_list[2 * i])
        val_f1_score_history.append(f1_history_list[2 * i + 1])

# create the averages
avg_precision = [sum(col)/len(col) for col in zip(*precision_history)]
avg_val_precision = [sum(col)/len(col) for col in zip(*val_precision_history)]
avg_recall = [sum(col)/len(col) for col in zip(*recall_history)]
avg_val_recall = [sum(col)/len(col) for col in zip(*val_recall_history)]
avg_f1_score = [sum(col)/len(col) for col in zip(*f1_score_history)]
avg_val_f1_score = [sum(col)/len(col) for col in zip(*val_f1_score_history)]

# plot the averages
plt.figure(1)
# plot the data for the the precision
plt.plot(avg_precision, label='training precision')
plt.plot(avg_val_precision, label='validation precision')
plt.title(f'average precision plots, total folds: {kfold_splits}')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()

plt.figure(2)
# now recall
plt.plot(avg_recall, label='training recall')
plt.plot(avg_val_recall, label='validation recall')
plt.title(f'average recall plots, total folds: {kfold_splits}')
plt.xlabel('Epoch')
plt.ylabel('Recall')

plt.figure(3)
# finally f1_score
plt.plot(avg_f1_score, label='training f1_score')
plt.plot(avg_val_f1_score, label='validation f1_score')
plt.title(f'average F1_Score plots, total folds: {kfold_splits}')
plt.xlabel('Epoch')
plt.ylabel('F1_Score')

# now find the best model, retrain it and finally store it
# technically we want the best precision for spam detection, but since we get 99% of the time 100% precision,
# we decided to use the best f1 score to select the best trained model, as that guarantees also the best recall score
# since precisions is almost always 100%
best_model_index = 0
for i in range(1, kfold_splits):
    if f1_score_list[i] > f1_score_list[best_model_index]:
        best_model_index = i

print('----------------------')
print(f'Best model was index {best_model_index}')
print('Proceeding to plot the specific model data')

# plot the best fold
plt.figure(4)
# plot the data for the the precision
if best_model_index == 0:
    plt.plot(history_list[best_model_index].history['precision'], label='training precision')
    plt.plot(history_list[best_model_index].history['val_precision'], label='training precision')
else:
    plt.plot(history_list[best_model_index].history[f'precision_{best_model_index}'], label='training precision')
    plt.plot(history_list[best_model_index].history[f'val_precision_{best_model_index}'], label='training precision')
plt.title(f'Best fold precision plot')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()

plt.figure(5)
# plot the data for the the precision
if best_model_index == 0:
    plt.plot(history_list[best_model_index].history['recall'], label='training Precision')
    plt.plot(history_list[best_model_index].history['val_recall'], label='validation Precision')
else:
    plt.plot(history_list[best_model_index].history[f'recall_{best_model_index}'], label='training Recall')
    plt.plot(history_list[best_model_index].history[f'val_recall_{best_model_index}'], label='validation Recall')
plt.title(f'Best fold recall plot')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

plt.figure(6)
# plot the data for the the precision
plt.plot(f1_history_list[2 * best_model_index], label='training F1_Score')
plt.plot(f1_history_list[2 * best_model_index + 1], label='validation F1_Score')
plt.title(f'Best fold F1_Score plot')
plt.xlabel('Epoch')
plt.ylabel('F1_Score')
plt.legend()

plt.show()
