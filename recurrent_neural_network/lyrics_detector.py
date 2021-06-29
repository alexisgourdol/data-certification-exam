# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Lyrics detector Challenge
# 
# The goal for this challenge is to leverage your knowledge of Deep Learning to design and train a lyrics classifier. For a given verse $X$, our model should learn to predict the artist $y$. The dataset consists of lyrics scrapped from the Genius website.
# 
# ### Objectives:
# - Text preprocessing
# - Text embedding
# - Train a RNN to detect the artist behind a set of lyrics

# <codecell>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

# <codecell>

# !pip install Unidecode
# !pip install gensim
# !pip install python-Levenshtein

# <codecell>

from unidecode import unidecode
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

# <markdowncell>

# ## 1. Data Cleaning
# 
# Our dataset contains around 4,000 verses of lyrics from different artists: Drake, Ed Sheeran and Kanye West (the verses are given in this order).

# <codecell>

raw_data = pd.read_csv("https://wagon-public-datasets.s3.amazonaws.com/certification_france_2021_q2/verses.csv")
data = raw_data.copy() # From now on, update `data` as you see fit and don't touch raw_data
data

# <markdowncell>

# ❓ **Have a look at the verse index 18th**. 
# - What do you observe?
# - Clean verses from non standard characters using [`unidecode.unidecode()`](https://pypi.org/project/Unidecode/)

# <codecell>

data.iloc[18].verse # the strings has some unicode weird characters e.G. \u2005was\u2005

# <codecell>

print(data.shape)
data.verse = data.verse.apply(unidecode)
print(data.shape)
print(data.iloc[18].verse)  # should be cleaned

# <markdowncell>

# ❓ **Check if some verses are duplicated.** 
# - It can be frequent in music lyrics.
# - If so, remove them to avoid data leaks between train and test sets

# <codecell>

data.verse.duplicated().sum() # 946 duplicated lyrics

# <codecell>

data.drop_duplicates(subset='verse', inplace=True)

# <codecell>

#3031 rows × 2 columns without subset vs 3029 rows × 2 columns with subset

# <codecell>

from nbresult import ChallengeResult
result = ChallengeResult(
    'data_loading',
    shape=data.shape,
    verses=data.verse[:50]
)

result.write()

# <markdowncell>

# ## 2. Data Analysis (given to you)

# <markdowncell>

# 👉 **We check the number of unique artist and the number of verses per artist**

# <codecell>

data.artist.value_counts()

# <markdowncell>

# 👉 **For each artist, let's have a look at the top-10 most used words to see if they look similar?**
# 
# We'll use Tensorflow's [`Tokenizer`](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer)'s index_word

# <codecell>

drake = data[data.artist =='Drake'].verse
ed = data[data.artist =='Ed Sheeran'].verse
kanye = data[data.artist =='Kanye West'].verse

# <codecell>

tokenizer_drake = tf.keras.preprocessing.text.Tokenizer()
tokenizer_ed = tf.keras.preprocessing.text.Tokenizer()
tokenizer_kanye = tf.keras.preprocessing.text.Tokenizer()

tokenizer_drake.fit_on_texts(drake)
tokenizer_ed.fit_on_texts(ed)
tokenizer_kanye.fit_on_texts(kanye)

# <codecell>

pd.DataFrame(data={
    "Drake": pd.Series(tokenizer_drake.index_word)[:10],
    "Ed Sheeran": pd.Series(tokenizer_ed.index_word)[:10],
    "Kanye West": pd.Series(tokenizer_kanye.index_word)[:10],
})

# <codecell>

tokenizer_drake.index_word

# <codecell>

tokenizer_drake.__dict__.keys()

# <markdowncell>

# 👉 **Let's quantify how much vocabulary do they have in common**
# 
# - An artist **vocabulary** is the **set** of all unique used words
# - We compute the `ratio` of (i) the length of vocabulary they **share**, over (ii) the length of the **total** vocabulary of the dataset
# 
# <details>
#     <summary>Hints</summary>
# 
# We'll use Python [`set.intersection()`](https://www.programiz.com/python-programming/methods/set/intersection) and [`set.union()`](https://www.programiz.com/python-programming/methods/set/union)
# </details>

# <codecell>

drake_vocabulary = set(tokenizer_drake.index_word.values())
ed_vocabulary = set(tokenizer_ed.index_word.values())
kanye_vocabulary = set(tokenizer_kanye.index_word.values())

# <codecell>

common_vocabulary = drake_vocabulary.intersection(ed_vocabulary).intersection(kanye_vocabulary)
global_vocabulary = drake_vocabulary.union(ed_vocabulary).union(kanye_vocabulary)

ratio = len(common_vocabulary)/len(global_vocabulary)
print(f"{ratio*100:.2f}% of the artists' vocabulary is common")

# <markdowncell>

# ## 3. Data Preprocessing

# <markdowncell>

# ### 3.1 Word Embedding
# We now need to think about embedding our sentences into numbers. We will be using [`gensim.models.Word2Vec`](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec) to embed each word of the sentence and concatenate the embeddings of the words forming the sentence.

# <markdowncell>

# ❓ **Transform the list of strings (verses) into a list of word sequences (a word sequence is a list of words contained in a string)**
# - Store these sequences of words in a new column `data["seq"]` in your dataframe
# - You can use `tensorflow.keras.preprocessing.text.text_to_word_sequence` 

# <codecell>

data["seq"] = [tf.keras.preprocessing.text.text_to_word_sequence(_) for _ in data.verse]

# <codecell>

data.head()

# <markdowncell>

# ❓ **Let's check if we can cap the length of each sequences without losing too much information**
# - Plot the distribution of sequences lengths using the [`seaborn.kdeplot`](https://seaborn.pydata.org/generated/seaborn.displot.html#seaborn-displot) function
# - Does it seem reasonable to limit ourself to 300 words per verse later on? 

# <codecell>

sns.kdeplot(data.seq.apply(len))
plt.axvline(x=300, c='red'); # yes, 300 is very reasonable

# <markdowncell>

# ❓ **Keep only the first `300` words of each sequences to reduce the useless long tail of long verses**

# <codecell>

data.seq.apply(keep_300)

# <codecell>

def keep_300(x: list) -> list:
    return x[:300]

sns.kdeplot(data.seq.apply(keep_300).apply(len));

# <codecell>

data.seq = data.seq.apply(keep_300)

#check of max value
print(data.seq.apply(keep_300).apply(len).max())

data.head()

# <markdowncell>

# ❓ **Train a `gensim.models.Word2Vec` model on your dataset** 
# - You want to embed each word into vectors of dimension `100`
# - No words should be excluded
# - Give Word2Vec at least 50 epochs to be sure it converges
# - Store these lists of vectors in a new column `data["embed"]`

# <codecell>

from gensim.models import Word2Vec
word2vec = Word2Vec(sentences=data.seq, vector_size=100, epochs=50)

# <codecell>

def embed_sentence(word2vec, sentence):
    embedded_sentence = []
    for word in sentence:
        if word in word2vec.wv:
            embedded_sentence.append(word2vec.wv[word])
        
    return np.array(embedded_sentence)

# <codecell>

def embedding(word2vec, sentences):
    embed = []
    
    for sentence in sentences:
        embedded_sentence = embed_sentence(word2vec, sentence)
        embed.append(embedded_sentence)
        
    return embed

# <codecell>

data["embed"] = embedding(word2vec, list(data.seq))

# <codecell>

print(len(data.seq.loc[0]))
print(data.embed.loc[0].shape)

print(len(data.seq.loc[100]))
print(data.embed.loc[100].shape)

print(len(data.seq.loc[2500]))
print(data.embed.loc[2500].shape)

print(len(data.seq.loc[700]))
print(data.embed.loc[700].shape)

# <codecell>

# Check 
assert len(data['embed']) == len(data)

# <markdowncell>

# ### 3.2 Create (X,y)

# <markdowncell>

# ❓ **Create your numpy array `X` of shape (number_of_verses, 300, 100)**
# 
# - 300 words per verse (pad verses shorter than 300 with zeros at the end) 
# - each words being a vector of size 100
# 
# <img src="https://raw.githubusercontent.com/lewagon/data-images/master/DL/padding.png" width=400>

# <codecell>

data.embed.head().apply(len).to_frame()

# <codecell>

X = pad_sequences(data.embed, dtype='float32', padding='post', value=0)
X.shape

# <markdowncell>

# ❓ **Create the numpy array `y` of shape `(n_verses, 3)` that contains the one-hot-encoded list of labels, for the RNN**

# <codecell>

artists_order = pd.get_dummies(data.artist).columns
artists_order

# <codecell>

y = pd.get_dummies(data.artist).to_numpy()
print(y.shape)
y

# <markdowncell>

# 👉 We train/test split the dataset below for you

# <codecell>

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# <codecell>

from nbresult import ChallengeResult
result = ChallengeResult(
    'data_preprocessing',
    n_zeros = np.sum(X == 0),
    X_shape = X.shape,
    y_shape = y.shape,
)

result.write()

# <markdowncell>

# ## 4. Recurrent Neural Network

# <markdowncell>

# 👉 Run this code below if you haven't managed to build your own (X,Y) training sets. This will load them as solution
# 
# ```python
# ! wget \
# 'https://wagon-public-datasets.s3.amazonaws.com/certification_france_2021_q2/data_lyrics_solution.pickle'
# 
# import pickle
# with open("data_lyrics_solution.pickle", "rb") as file:
#     (X_train, y_train, X_test, y_test) = pickle.load(file)
#     
# ! rm data_lyrics_solution.pickle
# ```

# <markdowncell>

# ❓ **First, store your baseline accuracy to beat as `score_baseline`**
# - Consider predicting always the most frequent artist

# <codecell>

print(artists_order)
y.sum(axis=0)

# <codecell>

# Let's predict always Drake

# <codecell>

y_baseline = np.tile([1,0,0], reps=(3029,1))
print(y_baseline.shape)
print(y_baseline[:10])
print('...')

# <codecell>

score_baseline = y.sum(axis=0)[0] / y.sum(axis=0).sum()

# <markdowncell>

# ❓ **Create a RNN architecture to predict the artists `y`  given verses `X`** :
# 
# - Keep it simple: use only one LSTM layer and one *hidden* dense layer between the input and output layers
# - Don't forget to take care of fake "zeros" added during preprocessing
# - Store it into the `model` variable.

# <codecell>

X_train.shape

# <codecell>

vocab_size = X_train.shape[1]
vocab_size

# <codecell>

embedding_dimension = X_train.shape[2]
embedding_dimension

# <codecell>

model = Sequential()
model.add(layers.Masking(mask_value=0))
model.add(layers.LSTM(units=10, activation='tanh')) # tanh
model.add(layers.Dense(1))
model.add(layers.Dense(3, activation='softmax'))

# <markdowncell>

# ❓ **Train your `model` on the `(X_train, y_train)` training set**
# - Use an appropriate loss
# - Adapt the learning rate of your optimizer if convergence is too slow/fast
# - Make sure your model does not overfit with appropriate control techniques
# 
# 💡 You will not be judged by the computing power of your computer, you can reach decent performance in less than 3 minutes of training without GPUs.

# <codecell>

optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=0.01, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
    name='RMSprop')

model.compile(loss='binary_crossentropy', optimizer=optimizer , metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=16, validation_split=0.2, epochs=10, verbose=0);

# <markdowncell>

# ❓ **Plot the training and validation losses through training**

# <codecell>

model.history.history

# <codecell>

# Plot below your train/val loss history ====> learning_rate=0.01 
plt.plot(model.history.history['loss'], label='loss')
plt.plot(model.history.history['val_loss'], label='val_loss')
plt.legend();

# Run also this code to save figure as jpg in path below (it's your job to ensure it works)
fig = plt.gcf()
plt.savefig("tests/history.png")

# <markdowncell>

# ❓ **Save your accuracy on test set as `score_test`**

# <codecell>

model.evaluate(X_test, y_test, verbose=1)

# <codecell>

score_test = 0.6320

# <markdowncell>

# 🧪 **Send your results below**

# <codecell>

from nbresult import ChallengeResult

result = ChallengeResult(
    "network",
    loss = model.loss,
    input_shape = list(model.input.shape),
    layer_names = [layer.name for layer in model.layers],
    final_activation = model.layers[-1].activation.__wrapped__._keras_api_names[0],
    score_baseline = score_baseline,
    score_test = score_test,
)
result.write()
