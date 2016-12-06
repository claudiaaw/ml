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
            words.append(x[0])
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
        tweet.append(line.strip())
        if line.strip()=="":
            tweet.remove("")
            l.append(tweet)
            tweet=[]
                
        
    in_file.close()
    return l

def write_devp2(language,word_list,tag_list):
    file_name = language+"/"+"dev.p2.out"
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
        
def write_devp3(language,word_list,tag_list):
    file_name = language+"/"+"dev.p3.out"
    if os.path.isfile(file_name):
        print('file exist')
        try:
            os.remove(file_name)
            print("deleted file")
        except OSError:
            print ("error")
            #pass
    out_file = open(file_name,'a',encoding='utf8')

    for i in range(len(word_list)): 
        lw = len(word_list[i])
        lt = len(tag_list[i])
        #print("length of words = "+str(lw))
        #print("length of tags = "+str(lt))
        if lt!=lw:
            print(word_list[i])
            print(tag_list[i])
            print (i)
        
        for j in range(len(word_list[i])):
            out_file.write(word_list[i][j]+" "+tag_list[i][j]+"\n")
        out_file.write(" \n")
    
    out_file.close()   
    
def write_devp4(language,word_list,tag_list):
    file_name = language+"/"+"dev.p4.out"
    if os.path.isfile(file_name):
        print('file exist')
        try:
            os.remove(file_name)
            print("deleted file")
        except OSError:
            print ("error")
            #pass
    out_file = open(file_name,'a',encoding='utf8')

    for i in range(len(word_list)): 
        lw = len(word_list[i])
        lt = len(tag_list[i])
        #print("length of words = "+str(lw))
        #print("length of tags = "+str(lt))
        if lt!=lw:
            print(word_list[i])
            print(tag_list[i])
            print (i)
        
        for j in range(len(word_list[i])):
            out_file.write(word_list[i][j]+" "+tag_list[i][j]+"\n")
        out_file.write(" \n")
    
    out_file.close()   

#======================================================================================
#======================================================================================

import time, os, math
from collections import Counter
from itertools import count

class HMM:

    tags = ["B-negative","B-neutral","B-positive","I-negative","I-neutral","I-positive","O"]
    word_tag_em = {} # {(word,tag): emission probability}
    transition_prob = {} # {(prev_tag,current_tag): transition probability}
    def __init__(self,train_words,train_tags):
        self.train_words = train_words 
        self.train_tags = train_tags 
        
        self.join_train_words = [j for i in train_words for j in i] #convert nested lists into one big list for zipping
        self.join_train_tags = [j for i in train_tags for j in i]
        
        self.tag_count = Counter(self.join_train_tags) # total count for each tag found in training data
        self.word_tag_count = Counter(list(zip(self.join_train_words,self.join_train_tags))) # total count for each word-tag pair found in training data
        
        self.train_emission()
        self.new_word = self.new_word_emission() # (tag,emission)
        
        
        # find count for each (prev_tag,current_tag) pair found in training data      
        self.transition_count = self.get_trans_count() 
        self.train_transition()
        



#======================================================================        
#============================= PART 2 =================================
#======================================================================

    #counting number of occurence of y_i
    def count_tag(self,tag):                
        return self.tag_count[tag]
    
    
    #2a count(y pair x)/ count(y)
    def emission(self,word,tag):
        return self.word_tag_count[(word,tag)]
    
    
    #2b emission count(y pair x)/ (count(y)+1)
    def get_emission_prob(self,word,tag):
        if word not in self.join_train_words:
            return 1/(self.count_tag(tag)+1)
        else:
            return self.word_tag_count[(word,tag)]/(self.count_tag(tag)+1)
    
    #2b finding tag for new word using estimation 1/ (count(y)+1)   
    def new_word_emission(self):
        em=[]
        for tag in self.tags:  
            em.append(self.word_tag_em["new word"][tag])
        #return label using the index of maximum emission found
        new_word_em = max(em)
        new_word_tag = self.tags[em.index(max(em))]
        return (new_word_tag,new_word_em)
    
    def train_emission(self):
        #for each (word,tag) pair, calculate emission score and put in dictionary
        for wt_pair in self.word_tag_count:
            self.word_tag_em[wt_pair]=self.get_emission_prob(wt_pair[0],wt_pair[1])
        self.word_tag_em["new word"]={}
        for tag in self.tags:
            
            self.word_tag_em["new word"][tag]=self.get_emission_prob("new word",tag)
        return self.word_tag_em
    
    def sentiment_analysis(self,test_data):
        
        predicted_tags = []
        for tweet in test_data:
            max_tag = []
            for word in tweet:
                temp = []
                if word in self.join_train_words:
                    #print("\nI AM INSIDE\n")
                    for tag in self.tags:
                        if (word,tag) in self.word_tag_em:
                            temp.append((self.word_tag_em[(word,tag)]))
                        else:
                            temp.append(0)
                    #print(temp)
                    max_em = max(temp)
                    p_tag = self.tags[temp.index(max(temp))]

                else:
                    #print("\nI AM OUTSIDE\n")
                    p_tag = self.new_word[0]
                    

                        
                max_tag.append(p_tag)
                
            predicted_tags.append(max_tag)
        return predicted_tags




#======================================================================    
#=========================== PART 3 ===================================
#======================================================================


    def get_trans_count(self):
        tags = [["START"]+i+["STOP"] for i in self.train_tags]
        tags = [j for i in tags for j in i]
        start_tags = tags[:len(tags)-1]
        tags_stop = tags[1:]
        return Counter(list(zip(start_tags,tags_stop))) 
        
        
    
    def get_transition(self,prev_tag,current_tag):
        if prev_tag == "STOP" and current_tag == "START":
            return 0
        elif (prev_tag,current_tag) not in self.transition_count:
            return 0
        else:
            if prev_tag=="START" or current_tag=="STOP":
                return self.transition_count[prev_tag,current_tag]/len(self.train_tags)
            else:
                return self.transition_count[prev_tag,current_tag]/self.count_tag(prev_tag)
    
    def train_transition(self):
        for pair in self.transition_count:
            self.transition_prob[pair] = self.get_transition(pair[0],pair[1])
        for tag in self.tags:
            self.transition_prob[("START",tag)] = self.get_transition("START",tag)
            self.transition_prob[(tag,"STOP")] = self.get_transition(tag,"STOP")
        return self.transition_prob
            
        
    def viterbi(self,tweet):
        n = len(tweet)
        possible_path = {} #(current state k, current tag): (prev tag, score from prev tag)
        optimal_path = []
        
        for k in range(n+2):            
            if k == 0:
                score = math.log(1)
                
            elif k == 1:
                word = tweet[k-1]

                for tag in self.tags: 
                    if ("START",tag) in self.transition_prob:                  
                        a = self.transition_prob[("START",tag)]
                        
                    else:
                        a = 0
                    
                    if (word,tag) in self.word_tag_em:
                        b = self.word_tag_em[(word,tag)]
                       
                    elif word not in self.join_train_words:
                        b = self.word_tag_em["new word"][tag]
                        
                    else:
                        b = 0
                    
                    try:
                        score = math.log(a*b)
                    except ValueError:
                        score = -math.inf
                    
                    possible_path[(1,tag)] = (score,"START")
                            
            elif k <= n:
                word = tweet[k-1]
                           
                for current_tag in self.tags:
                    temp = []
                    
                    if (word,current_tag) in self.word_tag_em:
                        b = self.word_tag_em[(word,current_tag)]
                        
                    elif word not in self.join_train_words:
                        b = self.word_tag_em["new word"][current_tag]
                        
                    else:
                        b = 0
                            
                    for prev_tag in self.tags:
                        if (prev_tag,current_tag) in self.transition_prob:
                            a = self.transition_prob[(prev_tag,current_tag)]
                        else:
                            a = 0

                        try:
                            prev_score = possible_path[(k-1, prev_tag)][0]
                            score = prev_score + math.log(a) + math.log(b)
                            
                        except ValueError:
                            score = -math.inf
                        
                        if score != -math.inf:
                            temp.append((score, prev_tag))
                            
                    #if all previous tags are of negative infinity i.e. a*b = 0, assume path from "O"
                    if temp == []:
                        possible_path[(k,current_tag)] = (-math.inf,"O")
                    else:
                        optimal_prev_tag = max(temp)[1]
                        possible_path[(k,current_tag)]=(max(temp)[0],optimal_prev_tag)  
                                    
            # when k = n+1
            else:
                temp = []
                for tag in self.tags: 
                    if (tag,"STOP") in self.transition_prob:
                        a = self.transition_prob[(tag,"STOP")]
                    else:
                        a = 0
                    
                    try:
                        prev_score = possible_path[(k-1, tag)][0]
                        score = prev_score+math.log(a)
                    except ValueError:
                        score = -math.inf
                        
                    if score != -math.inf:
                        temp.append((score,tag))
                if temp == []:
                    possible_path[(k,"STOP")] = (-math.inf,"O")
                else:
                    optimal_prev_tag = max(temp)[1]    
                    possible_path[(k,"STOP")]=(max(temp)[0],optimal_prev_tag)

        current_tag = "STOP"
        for i in range(n+1,1,-1):  
            if (i, current_tag) in possible_path:
                prev_tag = possible_path[(i,current_tag)][1]
                optimal_path.append(prev_tag)
                current_tag = prev_tag
                
#         if optimal_path == []:
#             print(possible_path)
        optimal_path.reverse()     
        return optimal_path
        
    def run_viterbi(self,test_data):
        test_tags = []
        for tweet in test_data:
            test_tags.append(self.viterbi(tweet))
        return test_tags    




#======================================================================
#============================ PART 4 ================================== 
#======================================================================    


    def top_k(self, kvalue, tweet):
        if (kvalue < 1):
            print ("Not a valid k-value")
            return None
            
        n = len(tweet)

        possible_path = {} #(current state k, current tag): (prev tag, score from prev tag)
        optimal_path = []
        top_k_path = {}
        
        for k in range(n+2):            
            if k == 0:
                score = math.log(1)
                
            elif k == 1:
                word = tweet[k-1]

                for tag in self.tags: 
                    if ("START",tag) in self.transition_prob:                  
                        a = self.transition_prob[("START",tag)]
                        
                    else:
                        a = 0
                    
                    if (word,tag) in self.word_tag_em:
                        b = self.word_tag_em[(word,tag)]
                       
                    elif word not in self.join_train_words:
                        b = self.word_tag_em["new word"][tag]
                        
                    else:
                        b = 0
                    
                    try:
                        score = math.log(a*b)
                    except ValueError:
                        score = -math.inf
                    
                    possible_path[(1,tag)] = (score,"START")
                    
            elif k==2 and k<=n:
                word = tweet[k-1]
                for current_tag in self.tags:
                    temp = []
                    
                    if (word,current_tag) in self.word_tag_em:
                        b = self.word_tag_em[(word,current_tag)]
                        
                    elif word not in self.join_train_words:
                        b = self.word_tag_em["new word"][current_tag]
                        
                    else:
                        b = 0
                            
                    for prev_tag in self.tags:
                        if (prev_tag,current_tag) in self.transition_prob:
                            a = self.transition_prob[(prev_tag,current_tag)]
                        else:
                            a = 0
                        
                                               
                        try:
                            prev_score = possible_path[(k-1, prev_tag)][0]
                            score = prev_score + math.log(a) + math.log(b)

                        except ValueError:
                            score = -math.inf  
                                                    
            
                        temp.append((score, prev_tag))
                            
                    #if all previous tags are of negative infinity i.e. a*b = 0, assume path from "O"
                    temp.sort()
                    possible_path[(k,current_tag)]={}
                    for i in range(1,kvalue+1):
                        possible_path[(k,current_tag)][i] = temp[len(temp)-i]
                    
                    
                          
                            
            elif k <= n:
                word = tweet[k-1]
                           
                for current_tag in self.tags:
                    temp = []
                    
                    if (word,current_tag) in self.word_tag_em:
                        b = self.word_tag_em[(word,current_tag)]
                        
                    elif word not in self.join_train_words:
                        b = self.word_tag_em["new word"][current_tag]
                        
                    else:
                        b = 0
                            
                    for prev_tag in self.tags:
                        if (prev_tag,current_tag) in self.transition_prob:
                            a = self.transition_prob[(prev_tag,current_tag)]
                        else:
                            a = 0
                        
                        for i in range(1,kvalue+1):
                            try:
                                
                                #print(possible_path[(k-1,prev_tag)])
                                prev_score = possible_path[(k-1, prev_tag)][i][0]
                                score = prev_score + math.log(a) + math.log(b)
                            
                            except ValueError:
                                score = -math.inf
                                
                        
            
                            temp.append((score, prev_tag, i))
                            
                    #if all previous tags are of negative infinity i.e. a*b = 0, assume path from "O"
                    temp.sort()
                    
                    
                    possible_path[(k,current_tag)]={}
                    for i in range(1,kvalue+1):
                        possible_path[(k,current_tag)][i]=temp[len(temp)-i]
                        
                        
            
                    
                                    
            # when k = n+1
            else:
                temp = []
                for tag in self.tags: 
                    if (tag,"STOP") in self.transition_prob:
                        a = self.transition_prob[(tag,"STOP")]
                    else:
                        a = 0
                    

                    
                    for i in range(1,kvalue+1):
                        if(n==1):
                            try:
                                prev_score = possible_path[(k-1, tag)][0]
                                score = prev_score + math.log(a) + math.log(b)
                            except ValueError:
                                score = -math.inf
                        else:
                            try:
                                prev_score = possible_path[(k-1, tag)][i][0]
                                score = prev_score + math.log(a) + math.log(b)

                            except ValueError:
                                score = -math.inf


                        temp.append((score,tag,i))
                    """if len(temp)!=7 :
                        print(k)
                        print(len(temp))
                        print("\n====\n")"""
                    
                possible_path[(k,"STOP")] = {}
                temp.sort()
                for i in range(1,kvalue+1):
                    possible_path[(k,"STOP")][i]=temp[len(temp)-i]
                

        
        for i in range(1,kvalue+1): 
            current_tag = "STOP"
            for j in range(n+1,1,-1):
                if (j, current_tag) in possible_path:
                    if j == 1:
                        prev_tag = possible_path[(j,current_tag)][1]
                    elif j==n+1:
                        prev_tag_kth = possible_path[(j,current_tag)][i][2]
                        prev_tag = possible_path[(j,current_tag)][i][1]
                    elif j==2:
                        prev_tag = possible_path[(j,current_tag)][prev_tag_kth][1]
                    else:
                        prev_tag_kth = possible_path[(j,current_tag)][prev_tag_kth][2]
                        prev_tag = possible_path[(j,current_tag)][prev_tag_kth][1]
                        
                        
                    optimal_path.append(prev_tag)
                    current_tag = prev_tag
            optimal_path.reverse()
            top_k_path[i] = optimal_path
            optimal_path=[]
            
        #print(top_k_path)
        return top_k_path
    
    def run_top_k(self,kvalue,kth_best,test_data):
        test_tags = []
        
        for tweet in test_data:
            top_k_path = self.top_k(kvalue,tweet)
            test_tags.append(top_k_path[kth_best])
        return test_tags


#====================================================================== 
#===============EDIT CODE HERE TO RUN ON OTHER FILES===================          
#======================================================================

#uncomment the respective part to run the code for that question
#change EN to other language ES, CN and SG here and in output file at the start of this file
train_data = read_train("EN/train")
test_data = read_dev_in("EN/dev.in")
hmm = HMM(train_data[0],train_data[1])
starttime = time.time()
#part 2
predicted_tags = hmm.sentiment_analysis(test_data)
write_devp2("EN",test_data,predicted_tags)

#part 3
"""test_tags = hmm.run_viterbi(test_data)
write_devp3("EN",test_data,test_tags)"""

#part4
"""top_5th = hmm.run_top_k(5,5,test_data)
write_devp4("EN",test_data,top_5th)"""
elapsed = time.time()-starttime

print ("time taken = "+str(elapsed)+"s")