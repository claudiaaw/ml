# format: [[word_list], [tag_list]]
# word_list = [[tweet1],[tweet2], ...]
# tag_list = [[tags for tweet1],[tags for tweet2], ...]
# tweet<n> = [word1, word2, ...]
# tags for tweet<n> = [tag1, tag2, ...]
def read_train(file_name):
    in_file = open(file_name,'r',encoding='utf8')
    l = []
    words = []
    tags = []
    word_list = []
    tag_list = []
    for line in in_file:
        x = line.strip().split()
        if x != []:
            words.append(x[0].lower())
            tags.append(x[1].rstrip('\n'))
        else:
            word_list.append(words)
            tag_list.append(tags)
            words = []
            tags = []

    l.append(word_list)
    l.append(tag_list)
    in_file.close()
    return l
    
# reading and writing to files 
# format:[[tweet1],[tweet2], ...]
# tweet<n> = [word1, word2, ...]
def read_dev_in(file_name):
    in_file = open(file_name,'r',encoding='utf8')
    l = []
    tweet = []
    for line in in_file:
        tweet.append(line.strip().lower())
        if line.strip()=="":
            tweet.remove("")
            l.append(tweet)
            tweet=[]
                
        
    in_file.close()
    return l

def write_devp5(language,word_list,tag_list):
    file_name = language+"/"+"test.p5.out"
    if os.path.isfile(file_name):
        print('file exist')
        try:
            os.remove(file_name)
            print("deleted file")
        except OSError:
            print("ERROR")
            
    out_file = open(file_name,'a',encoding='utf8')

    for i in range(len(word_list)):
        for j in range(len(word_list[i])):
            out_file.write(word_list[i][j]+" "+tag_list[i][j]+"\n")
        out_file.write(" \n")
    
    out_file.close()

#==========================================================================
#==========================================================================

#check if string is made up of numbers (will not affect sentiment)
def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
#list of stop words to disregard in tweet
stop_file = open("stop_list.txt",'r',encoding='utf8')
stop_list = []
for line in stop_file:
    if line[0] != "\ufeff":
        stop_list.append(line.strip())                
        
stop_file.close()


#==========================================================================
#==========================================================================



import time, itertools, re, math, operator, os
from collections import Counter

class part5:
    tags = ["B-negative","B-neutral","B-positive","I-negative","I-neutral","I-positive","O"]
    
    def __init__(self,train_words,train_tags,stop_list):
        self.train_words = train_words
        self.train_tags = train_tags
        self.long_train_words =  [j for i in self.train_words for j in i]
        self.stop_list = stop_list
        self.word_dict = {}
        self.preprocessing(train_words,train_tags)
        processed_words = [j for i in self.p_words_list for j in i]
        
        self.word_count = Counter(processed_words)

        
    def preprocessing(self,train_words,train_tags):
        self.p_words_list = []
        self.p_tags_list = []

        for tweet_index in range(len(train_words)):
            tweet_word = train_words[tweet_index]
            tweet_tag = train_tags[tweet_index]
            processed_word = []
            processed_tag = []
            
            for word_index in range(len(tweet_word)):
                word = tweet_word[word_index]
                tag = tweet_tag[word_index]
                
                #removing unnecessary words eg stop words, urls
                if word[0] == "@" or word[0] == "#":
                    word = word[1:]
                elif word not in self.stop_list and word[0:7]!="http://" and word.isalnum() and not(is_number(word)):
                    if len(word)>=5:
                        word = self.word_stem(word)
                    elif len(word)>=4:
                        word = self.remove_repeat(word)
                    if word not in self.word_dict:
                        self.word_dict[word]= {"B-negative":0,"B-neutral":0,"B-positive":0,"I-negative":0,"I-neutral":0,"I-positive":0,"O":0}
                    self.word_dict[word][tag]+=1

                    #print(self.word_stem(word))
                    processed_word.append(word)
                    processed_tag.append(tag)
                tag_group = []
                word_group = []
                
                    
                        
                                        
            self.p_words_list.append(processed_word)
            self.p_tags_list.append(processed_tag)
        
    #removes repeated characters in a word if character repeats more than 2 times for example haaapppyyy -> haappyy
    def remove_repeat(self,word):
        return re.sub(r'(.)\1{2,}', r'\1\1', word)
    
    #attempt to reduce words to their root words for example going -> go
    def word_stem(self,word):
        
        n = len(word)
        
        if word[n-3:] == "ing" and word[:n-3] in self.long_train_words:
            new_word = word[:n-3]          
            return new_word
        
        elif word[n-2:] == "ed" and word[:n-2] in self.long_train_words:
            new_word = word[:n-2]
            return new_word
        
        elif word[n-1] == "s" and word[:n-1] in self.long_train_words:
            new_word = word[:n-1]
            return new_word
        
        else:
            return word
    
    #to reduce data that causes skewed prediction
    def reduce_dict(self):
        list_sum = []
        for word in self.word_dict:
            total = sum(self.word_dict[word].values())
            list_sum.append((total,word))
        for i in range(len(list_sum)):
            if list_sum[i][0]<3:
                self.word_dict.pop(word,None)
        return list_sum
    
    #train naive bayes
    def nb_training(self):
        count_label = {}
        for tag in self.tags:
            count_label[tag] = 0
            for word in self.word_dict:
                count_label[tag] += self.word_dict[word][tag]
        total = sum(count_label.values())
        count_label["prob"]={}
        for tag in self.tags:
            count_label["prob"][tag] = count_label[tag]/total
        return count_label
            
    #run naive bayes algorithm    
    def naive_bayes(self,test_data):
        train_result = self.nb_training()
        predicted_result= []
        
        for tweet in test_data:
            tweet_sentiment = []
            predicted_tag = []
            prob = 0
            for tag in self.tags:
                prob = math.log(train_result["prob"][tag])-math.log(len(tweet)*train_result[tag])
                
                for word in tweet:
                                            
                    if word in self.word_dict:
                        occurence = self.word_dict[word][tag]
                        if occurence > 0:
                            prob+=math.log(occurence)
                        else:
                            prob+=math.log(1)
                    else:
                        prob+=math.log(1)
                tweet_sentiment.append((prob,tag))
            tweet_sentiment.sort()
            most_probable_sentiment = tweet_sentiment[-1][1]
            
            for word in tweet:
                if word[0] == "#" or word[0] == "@":
                    if len(word)>=5:
                        new_word = self.word_stem(word[1:])
                    else:
                        new_word = word[1:]
                    if new_word in self.word_dict:
                        
                        new_tag = max(self.word_dict[new_word].items(), key=operator.itemgetter(1))[0]
                        predicted_tag.append(new_tag)
                    else:
                        predicted_tag.append(most_probable_sentiment)
                elif word in self.stop_list or word[0:7]=="http://" or not word.isalnum() or is_number(word):
                    predicted_tag.append("O")
                else:
                    new_word = self.word_stem(word)
                    if new_word in self.word_dict:
                        
                        new_tag = max(self.word_dict[new_word].items(), key=operator.itemgetter(1))[0]
                        
                        predicted_tag.append(new_tag)
                    else:
                        predicted_tag.append(most_probable_sentiment)
            predicted_result.append(predicted_tag)
        return predicted_result

train_data = read_train("EN/train")
test_data = read_dev_in("EN/test.in")

starttime = time.time()
part_5 = part5(train_data[0],train_data[1],stop_list)
print("predicting...")
predicted_tags = part_5.naive_bayes(test_data)

write_devp5("EN",test_data,predicted_tags)

elapsed = time.time()-starttime

print ("time taken = "+str(elapsed)+"s")