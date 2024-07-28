#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


import nltk


# In[4]:


import nltk
nltk.download('punkt')
nltk.download('stopwords')

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# In[5]:


df = pd.read_csv('spam.csv',encoding='ISO-8859-1')


# In[6]:


df.sample(5)


# In[7]:


df.info()


# In[8]:


df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)


# In[9]:


df.sample(2)


# In[10]:


df.rename(columns={'v1':'target', 'v2': 'text'}, inplace=True)


# In[11]:


df.sample(2)


# In[12]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

df['target'] = encoder.fit_transform(df['target'])


# In[13]:


df.head()


# In[14]:


df.isnull().sum()


# In[15]:


df.duplicated().sum()


# In[16]:


df.drop_duplicates(keep='first', inplace=True)


# In[17]:


df.shape


# In[18]:


df['target'].value_counts()


# In[19]:


plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")
plt.show()


# In[20]:


df['num_characters'] = df['text'].apply(len)

# Calculate the number of words in each SMS using NLTK tokenizer and add as a new column
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))

# Calculate the number of sentences in each SMS using NLTK tokenizer and add as a new column
df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))


# In[21]:


df.head()


# In[22]:


df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[23]:


df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[25]:


plt.figure(figsize=(12, 8))
sns.histplot(df[df['target'] == 0]['num_characters'])
sns.histplot(df[df['target'] == 1]['num_characters'], color='pink')
plt.tight_layout()
plt.show()


# In[26]:


plt.figure(figsize=(12, 8))
sns.histplot(df[df['target'] == 0]['num_words'])
sns.histplot(df[df['target'] == 1]['num_words'], color='pink')
plt.tight_layout()
plt.show()


# In[27]:


sns.pairplot(df, hue='target')


# In[28]:


df[['target','num_characters','num_words','num_sentences']].corr()


# In[29]:


sns.heatmap(df[['target','num_characters','num_words','num_sentences']].corr(),annot=True)


# In[30]:


import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps =PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)


# In[31]:


transform_text("""Say Every Body
            Our Satguru
            The Love Charger
            True Love Charger
            You are The Love Charger
            You are The Love Charger
            I am so Lucky Because
            You are my Love Charger
            You are The Love Charger
            Billions battery when goes down
            You charged up with Love
            So strong Your Power Love""")


# In[32]:


df['transformed_text'] = df['text'].apply(transform_text)


# In[33]:


df.head()


# In[35]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')

spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep = " "))
plt.figure(figsize=(12,8))
plt.imshow(spam_wc)
plt.show()


# In[36]:


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))
plt.figure(figsize=(12,8))
plt.imshow(ham_wc)
plt.show()


# In[37]:


spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
len(spam_corpus)


# In[38]:


from collections import Counter
Counter(spam_corpus).most_common(30)


# In[39]:


sns.barplot(x = pd.DataFrame(Counter(spam_corpus).most_common(30))[0],y = pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.show()


# In[40]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[41]:


len(ham_corpus)


# In[42]:


sns.barplot(x = pd.DataFrame(Counter(ham_corpus).most_common(30))[0],y = pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.tight_layout()
plt.show()


# In[44]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features= 3000)

X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[45]:


X.shape


# In[46]:


y = df['target'].values


# In[47]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=2)


# In[48]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[49]:


gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))


# In[50]:


mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))


# In[51]:


bnb.fit(X_train, y_train)
y_pred = bnb.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))


# In[52]:


from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression(solver='liblinear', penalty='l1')


# In[53]:


clfs = {'LR': lrc}


# In[54]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[ ]:




