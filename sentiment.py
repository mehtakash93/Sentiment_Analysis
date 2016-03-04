import sys
import collections
import sklearn.naive_bayes
from sklearn import linear_model
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
import nltk
import numpy as np
import gensim
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
#nltk.download("stopwords")          # To Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model,lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    all_words=[]
    all_words_neg=[]
    for i in range(0,len(train_pos)):
      train_pos[i]=[elem for elem in train_pos[i] if elem not in stopwords]
      temp=[]
      for elem in train_pos[i]:
          if elem not in temp:
            temp.append(elem)
            all_words.append(elem)
            
    for i in range(0,len(train_neg)):
      train_neg[i]=[elem for elem in train_neg[i] if elem not in stopwords]
      temp=[]
      for elem in train_neg[i]:
          if elem not in temp:
            temp.append(elem)
            all_words_neg.append(elem)       

    
    qualified=[]
    word_dict = collections.Counter(all_words)
    word_dict_neg = collections.Counter(all_words_neg)
    val=int(0.01*len(train_pos))
    s=set(word_dict.keys()).union(set(word_dict_neg.keys()))
    words_un=set(s);
    pos=words_un-set(word_dict)
    neg=words_un-set(word_dict_neg)
        
    m=dict((k, v) for k, v in word_dict.items() if v>=val)
    
    
    
    valn=int(0.01*len(train_neg))
    n=dict((k, v) for k, v in word_dict_neg.items() if v>=valn)
    for aword in pos:
        word_dict[aword]=0
    for nword in neg:
        word_dict_neg[nword]=0 
    
    
    qualified=[k for k in words_un if (word_dict[k]>=val or word_dict_neg[k]>=val) and (word_dict[k]>=2*word_dict_neg[k] or 2*word_dict[k]<=word_dict_neg[k])]
  
  
    

    # List of words that will be used as features. 
    # This list has the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
  
    train_pos_vec=[[0 for x in range(len(qualified))] for x in range(len(train_pos))] 
    train_neg_vec=[[0 for x in range(len(qualified))] for x in range(len(train_neg))] 
    test_pos_vec=[[0 for x in range(len(qualified))] for x in range(len(test_pos))] 
    test_neg_vec=[[0 for x in range(len(qualified))] for x in range(len(test_neg))] 
    for i in range(0,len(train_pos)):
      for j in range(len(qualified)):
        if qualified[j] in train_pos[i]:
          train_pos_vec[i][j]=1
  
        
    for i in range(0,len(train_neg)):
      for j in range(len(qualified)):
        if qualified[j] in train_neg[i]:
          train_neg_vec[i][j]=1
         
    for i in range(0,len(test_pos)):
      for j in range(len(qualified)):
        if qualified[j] in test_pos[i]:
          test_pos_vec[i][j]=1   
    
    for i in range(0,len(test_neg)):
      for j in range(len(qualified)):
        if qualified[j] in test_neg[i]:
          test_neg_vec[i][j]=1   
    
    
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turning the datasets from lists of words to lists of LabeledSentence objects.

   
    x_train_pos=[]
    x_test_pos=[]
    x_train_neg=[]
    x_test_neg=[]
    
           
    LabeledSentence = gensim.models.doc2vec.LabeledSentence
    labeled_train_pos=[]
    labeled_test_pos=[]
    labeled_train_neg=[]
    labeled_test_neg=[]
      
      
       
    
    for i in range(len(train_pos)):
        labeled_train_pos.append(LabeledSentence(words=train_pos[i], tags=['TRAIN_POS_%s' % i]))
    for i in range(len(train_neg)):
        labeled_train_neg.append(LabeledSentence(words=train_neg[i], tags=['TRAIN_NEG_%s' % i]))    
    for i in range(len(test_pos)):
        labeled_test_pos.append(LabeledSentence(words=test_pos[i], tags=['TEST_POS_%s' % i]))      
    for i in range(len(test_neg)):
        labeled_test_neg.append(LabeledSentence(words=test_neg[i], tags=['TEST_NEG_%s' % i]))     

  
    
    
    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)
    
    train_pos_vec=[]
    test_pos_vec=[]
    train_neg_vec=[]
    test_neg_vec=[]

    # Train the model
   
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Using the docvecs function to extracting the feature vectors for the training and test data

    for i in range(len(train_pos)):
      train_pos_vec.append(model.docvecs['TRAIN_POS_%s' % i])
    print "bravo"  
    for i in range(len(train_neg)):
      train_neg_vec.append(model.docvecs['TRAIN_NEG_%s' % i]) 
      
    for i in range(len(test_pos)):
      test_pos_vec.append(model.docvecs['TEST_POS_%s' % i]) 
        
    for i in range(len(test_neg)):
      test_neg_vec.append(model.docvecs['TEST_NEG_%s' % i])  
          
    
    
 

    
    
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) +["neg"]*len(train_neg_vec)
    X=train_pos_vec+train_neg_vec
 
    # Using sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
  
    clf = BernoulliNB(alpha=1.0,binarize=None)
    nb_model=clf.fit(X, Y)
 
    logreg = linear_model.LogisticRegression()
    lr_model=logreg.fit(X,Y)
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    X= train_pos_vec+train_neg_vec
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    clf = GaussianNB()
    nb_model=clf.fit(X, Y)

    logreg = linear_model.LogisticRegression()
    lr_model=logreg.fit(X,Y)

    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    X=test_pos_vec+test_neg_vec
    Y=["pos"]*len(test_pos_vec) + ["neg"]*len(test_neg_vec)
    pre=model.predict(X)
  
    tp=0
    for i in range(len(test_pos_vec)):
      if pre[i]=="pos":
        tp=tp+1
    
    fn=0
    for i in range(len(test_pos_vec)):
      if pre[i]=="neg":
        fn=fn+1
    
    fp=0
    for i in range(len(test_pos_vec),len(test_pos_vec)+len(test_neg_vec)):
      if pre[i]=="pos":
        fp=fp+1
    
    tn=0 
    for i in range(len(test_pos_vec),len(test_pos_vec)+len(test_neg_vec)):
      if pre[i]=="neg":
        tn=tn+1
      
 
    accuracy=0
    accuracy=(tp+tn)/float(tp+fn+fp+tn)
    
    print_confusion=True
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)


if __name__ == "__main__":
    main()
