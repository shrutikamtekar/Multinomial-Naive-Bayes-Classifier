 
from sentiment_reader import SentimentCorpus
from multinomial_naive_bayes import MultinomialNaiveBayes

if __name__ == '__main__':
    dataset = SentimentCorpus()
    nb = MultinomialNaiveBayes()
    
    params = nb.train(dataset.train_X, dataset.train_y)
    #loglikelihood of each word per class  row= no of words columns =classes
    predict_train = nb.test(dataset.train_X, params)
    #predict_train has the output of test doc belonging to which class
    eval_train = nb.evaluate(predict_train, dataset.train_y)
    
    predict_test = nb.test(dataset.test_X, params)
    eval_test = nb.evaluate(predict_test, dataset.test_y)
    
    print "Accuracy on training set: %f, on test set: %f" % (eval_train, eval_test)



