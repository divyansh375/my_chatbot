from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
import gensim
from nltk.stem import WordNetLemmatizer
import string
import os
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.corpora.dictionary import Dictionary
import gensim
from nltk.stem import WordNetLemmatizer
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd



def filter(x):
                df=pd.read_csv("spam.csv",encoding='latin-1')
                df=df.loc[:, ~df.columns.str.contains('^Unnamed')]
                X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,1], df.iloc[:,0], test_size=0.33, random_state=53)
                count_vectorizer = CountVectorizer(stop_words='english')
                count_train = count_vectorizer.fit_transform(X_train)
                test=x
                test=[test]
                count_test = count_vectorizer.transform(test)
                nb_classifier=MultinomialNB()
                nb_classifier.fit(count_train,y_train)
                pred=nb_classifier.predict(count_test)
                if(pred[0]=="ham"):
                    print("This message is not spam!")
                elif(pred[0]=="spam"):
                    print("This  message is  spam !")
                    
                    
    




lemmatizer = WordNetLemmatizer() 
def print_topics(res):
    topics=[]
    for i in res:
        
            k=i[1].split("+")
            
            for l in k :
                m=l.split("*")
                c=m[1]
                c=c.replace('"','')
                c=c.replace(' ','')
                ll=[]
                ll.append(c)
                tagged_sent=word_tokenize(c)
                tagged_sent=nltk.pos_tag(tagged_sent)
                if(tagged_sent[0][1]=='NNP' or tagged_sent[0][1]=='NN'):
                    topics.append(c)
                    
    return(list(set(topics)))                
                
            
        

    






def rec_topic(x):
    my_doc=x.split(".")

    my_doc_fi=[]
    exclude = set(string.punctuation)
    for doc in my_doc:
        s = ''.join(ch for ch in doc if ch not in exclude)
        my_doc_fi.append(s)    


    tokenized_docs=[word_tokenize(i) for i in my_doc_fi]
    stop_tok=[]
    for doc in tokenized_docs:
        temp=[]
        for t in doc:
            if(t.lower() not in stopwords.words('english')):
                temp.append(t)
        stop_tok.append(temp)        

    lemma=[]
    for doc in stop_tok:
        temp=[]
        for t in doc:
            temp.append(lemmatizer.lemmatize(t))
        lemma.append(temp)


        
            


    dic=Dictionary(lemma)


    corpus=[dic.doc2bow(i) for i in lemma]


    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(corpus, num_topics=3, id2word = dic, passes=50)

    res=ldamodel.print_topics(num_topics=3, num_words=3)

    
    
    r=print_topics(res)
    return(r)
    












chatbot = ChatBot("Ron Obvious")
trainer = ListTrainer(chatbot)
#for files in os.listdir('./english/'):
    #data=open('./english/'+files,'r').readlines()
    #print(data)
    #trainer.train(data)












print("\n\n\n\n\n\n\n\n\n")
print("computer: Hi !what is your name?")
y=input()
y=y.split()
name=""
for i in y:
    tagged_sent=word_tokenize(i)
    tagged_sent=nltk.pos_tag(tagged_sent)
    if((tagged_sent[0][1]=='NNP') and i not in stopwords.words('english')):
        name=name+i

print("computer: oh! Thats a nice name....so how are you "+name+"?")
if(len(name)==0):
    name=name+"user"

while(1==1):
    print(name,end=": ") 
    y=input()
    
    
    set1 = set(y.split(' '))
    set2 = set(("help").split(' '))
    if(set1 == set2):
        
        
        print("press 1 for topic recognition")
        print("press 2 for spam filter")
        print("press 3 for exit")
        x=input()
        
        if(int(x)==1):
            state=input("enter the paragraph")
            print("The important words are:")
            for i in rec_topic(state):
                print(i)
        elif(int(x)==2):
            state=input("enter the paragraph")
            filter(state)
        elif(int(x)==3):
            print("bye"+name+"..have a great day!")
            break
            
           

        
    else:
        response = chatbot.get_response(y)
        print("computer: ",end=" ")
        print(response)
        
        











