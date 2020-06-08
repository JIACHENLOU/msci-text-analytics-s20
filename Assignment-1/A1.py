#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm
import random
import string


# In[2]:


# open data files and convert into lowercase
with open('/Users/Janice/Desktop/MSCI641/neg.txt') as n:
    neg = n.readlines()
with open('/Users/Janice/Desktop/MSCI641/pos.txt') as p:
    pos = p.readlines()
all_txt = neg + pos

lower = []
for i in all_txt:
  lower.append(i.lower())


# In[3]:


# download stopwords online
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


# In[4]:


# remove special characters from data
no_special_characters = []

for i in lower:
  no_special_characters.append(i.translate(str.maketrans('','', string.punctuation)))
no_special_characters[:5]


# In[5]:


# define a function to remove stopwords
def remove_words(file, stopwords):
    no_stopwords = []
    
    for line in file:
        split_word = line.split(" ")
        for i in range(len(split_word)):
            if split_word[i] in stopwords:
                split_word[i] = ""
                
        filtered_line = " ".join(split_word)
        filtered_line = filtered_line.replace("  ", " ")
        
        no_stopwords.append(filtered_line)
        
    return no_stopwords

# split each sentence into words
# if a word is in the stopwords lists, replace it with ""
# join the list of words, form filtered sentences 


# In[6]:


total_sentence = len(no_special_characters)
total_sentence


# In[7]:


# with stopwords 
# split data into 3 sets
csv_data = ''
train = ''
val = ''
test = ''

train_size = int(0.8*total_sentence)
val_size = int(0.1*total_sentence)

random.shuffle(no_special_characters)


# tokenized corpus, include stopwords, create csv files
for idx, sentence in tqdm(enumerate(no_special_characters)):
    sentence = sentence.strip().split()
    csv_sentence = '{}\n'.format(','.join(sentence))
    csv_data += csv_sentence
    
    if idx < train_size:
        train += csv_sentence
    elif idx >= train_size and idx < (train_size + val_size):
        val += csv_sentence 
    else:
        test += csv_sentence 
        

# without stopwords 
# remove stopwords
no_stopwords_file = remove_words(no_special_characters, stop_words)
total_no_stopwords = len(no_stopwords_file)

# split data into 3 sets
csv_data2 = ''
train2 = ''
val2 = ''
test2 = ''

train_size2 = int(0.8*total_no_stopwords)
val_size2 = int(0.1*total_no_stopwords)

# tokenized corpus, without stopwords, create csv files
for idx, sentence in tqdm(enumerate(no_stopwords_file)):
    sentence = sentence.strip().split()
    csv_sentence2 = '{}\n'.format(','.join(sentence))
    csv_data2 += csv_sentence2
    
    if idx < train_size2:
        train2 += csv_sentence2
    elif idx >= train_size2 and idx < (train_size2 + val_size2):
        val2 += csv_sentence2 
    else:
        test2 += csv_sentence2


# In[10]:


with open('out_w.csv', 'w') as f:
    f.write(csv_data)
with open('train_w.csv', 'w') as f:
    f.write(train)
with open('val_w.csv', 'w') as f:
    f.write(val)
with open('test_w.csv', 'w') as f:
    f.write(test)
with open('out_without.csv', 'w') as f:
    f.write(csv_data2)
with open('train_without.csv', 'w') as f:
    f.write(train2)
with open('val_without.csv', 'w') as f:
    f.write(val2)
with open('test_without.csv', 'w') as f:
    f.write(test2)


# In[ ]:




