from tqdm import tqdm
import random
import string
import os

def read_file(data):
    with open(os.path.join(data, 'pos.txt')) as f:
        pos = f.readlines()
    with open(os.path.join(data, 'neg.txt')) as f:
        neg = f.readlines()
    all_txt = pos + neg
    return list(zip(all_txt, [1]*len(pos) + [0]*len(neg)))
    
# download stopwords online
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]


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

all_txt = read_file('/Users/Janice/Desktop/MSCI641/A1')
len(all_txt)

csv_data = ''
train = ''
val = ''
test = ''
labels_data = ''

train_size = int(0.8*len(all_txt))
val_size = int(0.1*len(all_txt))

random.shuffle(all_txt)

for idx, line in tqdm(enumerate(all_txt)):
    sentence = line[0].strip().split()
    label = line[1]
    
    lower = []
    for i in sentence:
      lower.append(i.lower())
    
    no_special_characters = []
    for i in lower:
      no_special_characters.append(i.translate(str.maketrans('','', string.punctuation)))
    
    csv_sentence = '{}\n'.format(','.join(no_special_characters))
    csv_data += csv_sentence
    
    labels_data += '{}\n'.format(label)
    
    if idx < train_size:
        train += csv_sentence
    elif idx >= train_size and idx < (train_size + val_size):
        val += csv_sentence
    else:
        test += csv_sentence
        
 with open('out.csv', 'w') as f:
    f.write(csv_data)
with open('train.csv', 'w') as f:
    f.write(train)
with open('val.csv', 'w') as f:
    f.write(val)
with open('test.csv', 'w') as f:
    f.write(test)
with open('labels.csv', 'w') as f:
    f.write(labels_data)
  
def read_csv(data):
    with open(data) as f:
        data = f.readlines()
    return [' '.join(line.strip().split(',')) for line in data]

out = read_csv('/Users/Janice/Desktop/MSCI641/A2/NEW_out_w.csv')
remove_out = remove_words(out, stop_words)
total_no_stopwords = len(remove_out)

csv_data2 = ''
train2 = ''
val2 = ''
test2 = ''

train_size2 = int(0.8*total_no_stopwords)
val_size2 = int(0.1*total_no_stopwords)

# tokenized corpus, without stopwords, create csv files
for idx, sentence in tqdm(enumerate(remove_out)):
    sentence = sentence.strip().split()
    csv_sentence2 = '{}\n'.format(','.join(sentence))
    csv_data2 += csv_sentence2
    
    if idx < train_size2:
        train2 += csv_sentence2
    elif idx >= train_size2 and idx < (train_size2 + val_size2):
        val2 += csv_sentence2 
    else:
        test2 += csv_sentence2

with open('out_ns.csv', 'w') as f:
    f.write(csv_data2)
with open('train_ns.csv', 'w') as f:
    f.write(train2)
with open('val_ns.csv', 'w') as f:
    f.write(val2)
with open('test_ns.csv', 'w') as f:
    f.write(test2)


        
