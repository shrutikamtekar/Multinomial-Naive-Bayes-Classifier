import nltk
import os
from nltk import sent_tokenize,word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords



#reading data from location and making it into a list
def create_list_category(path):
    list_category=[]
    for filename in os.listdir(path):
        f = os.path.join(path, filename)
        file_content = open(f).read()
        list_category.append(file_content)    
    return list_category        

#preprocessing the document
def preprocess(sentence,category):
    tokens = nltk.word_tokenize(sentence)
    stoplist = stopwords.words('english')
    #    lemmatizer = WordNetLemmatizer()
    temp=[word.lower() for word in tokens if word.isalpha() and not word in stoplist and len(word)>2]
    doc_dict=dict((x,temp.count(x)) for x in set(temp))
    if doc_type==1: #chekcing if the document is of test pr train
        for word,occurence in doc_dict.iteritems():
            test_file.write(str(word)+":"+str(occurence)+' ')
        test_file.write('#label#:'+category+'\n')
    else:
        for word,occurence in doc_dict.iteritems():
            train_file.write(str(word)+":"+str(occurence)+' ')
        train_file.write('#label#:'+category+'\n')
        
#intializing the program
if __name__ == '__main__':
    #create test data list per category
    test_windows = create_list_category(r'C:\...\20news-bydate-test\comp.windows.x')
    test_autos = create_list_category(r'C:\...\20news-bydate-test\rec.autos')
    test_politics =create_list_category(r'C:\...\20news-bydate-test\talk.politics.guns')
    
    #create test data list for all category
    initial_test_doc = [(category, 'windows') for category in test_windows]
    initial_test_doc += [(category, 'autos') for category in test_autos]
    initial_test_doc += [(category, 'politics') for category in test_politics]
        
    #create train data list per category
    train_windows = create_list_category(r'C:\...\20news-bydate-train\comp.windows.x')
    train_autos = create_list_category(r'C:\...\20news-bydate-train\rec.autos')
    train_politics =create_list_category(r'C:\...\20news-bydate-train\talk.politics.guns')
    
    #create train data list for all category
    initial_train_doc = [(category, 'windows') for category in train_windows]
    initial_train_doc += [(category, 'autos') for category in train_autos]
    initial_train_doc += [(category, 'politics') for category in train_politics]
    
    with open("test_data.review", 'a') as test_file:
        doc_type = 1 # means its a test doc
        for text in initial_test_doc:
            preprocess(text[0],text[1])
        test_file.close()  
    print "created test.review file\n"
    
    with open("train_data.review", 'a') as train_file:
        doc_type = 0
        for text in initial_train_doc:
            preprocess(text[0],text[1])
        train_file.close()
    print "created train.review file\n"
    