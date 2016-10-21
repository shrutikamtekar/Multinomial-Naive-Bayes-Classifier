import numpy as np


class LinearClassifier():

    def __init__(self):
        self.trained = False

    def train(self,x,y):
        '''
        Returns the weight vector
        '''
        raise NotImplementedError('LinearClassifier.train not implemented')

    def get_scores(self,x,w):
        '''
        Computes the dot product between X,w
        '''
        return np.dot(x,w)

    def get_label(self,x,w):
        '''
        Computes the label for each data point
        '''
        scores = np.dot(x,w)
        return np.argmax(scores,axis=1).transpose()

    def test(self,x,w):
        '''
        Classifies the points based on a weight vector.
        '''
        if self.trained == False:
            raise ValueError("Model not trained. Cannot test")
            return 0
        x = self.add_intercept_term(x)
        return self.get_label(x,w)
    
    def add_intercept_term(self,x):
        ''' Adds a column of ones to estimate the intercept term for separation boundary'''
        nr_x, nr_f = x.shape
        intercept = np.ones([nr_x,1])
        x = np.hstack((intercept,x))
        return x

    def evaluate(self,truth,predicted):
        correct = 0.0
        total = 0.0
        for i in range(len(truth)):
            if(truth[i] == predicted[i]):
                correct += 1
            total += 1
        return 1.0*correct/total
    
    # def fscore(self,truth,predicted):
    #     n_classes=np.unique(truth).shape[0]
    #     actual_predicted = np.ndarray(shape=(n_classes,n_classes))
    #     precision = np.zeros(n_classes)
    #     recall = np.zeros(n_classes)
    #     fscore = np.zeros(n_classes)
    #     for i in range(len(truth)):
    #         if (truth[i] ==0 and predicted[i] == 0):
    #             actual_predicted[0,0]+=1
    #         elif (truth[i] ==0 and predicted[i] == 1):
    #             actual_predicted[0,1]+=1
    #         elif (truth[i] ==0 and predicted[i] == 2):
    #             actual_predicted[0,2]+=1
    #         elif (truth[i] ==1 and predicted[i] == 0):
    #             actual_predicted[1,0]+=1
    #         elif (truth[i] ==1 and predicted[i] == 1):
    #             actual_predicted[1,1]+=1
    #         elif (truth[i] ==1 and predicted[i] == 2):
    #             actual_predicted[1,2]+=1            
    #         elif (truth[i] ==2 and predicted[i] == 0):
    #             actual_predicted[2,0]+=1
    #         elif (truth[i] ==2 and predicted[i] == 1):
    #             actual_predicted[2,1]+=1
    #         elif (truth[i] ==2 and predicted[i] == 2):
    #             actual_predicted[2,2]+=1  
    #     Mfscore =0.0   

    #     precision[0] = actual_predicted[0,0]/(actual_predicted[0,0]+actual_predicted[1,0]+actual_predicted[2,0])
    #     precision[1] = actual_predicted[1,1]/(actual_predicted[0,1]+actual_predicted[1,1]+actual_predicted[2,1])
    #     precision[2] = actual_predicted[2,2]/(actual_predicted[0,2]+actual_predicted[1,2]+actual_predicted[2,2])

    #     recall[0]= actual_predicted[0,0]/(actual_predicted[0,0]+actual_predicted[0,1]+actual_predicted[0,2])
    #     recall[1]= actual_predicted[1,1]/(actual_predicted[1,0]+actual_predicted[1,1]+actual_predicted[1,2])
    #     recall[2]= actual_predicted[2,2]/(actual_predicted[2,0]+actual_predicted[2,1]+actual_predicted[2,2])
        
        
    #     fscore[0] = (2*precision[0]*recall[0])/(precision[0]+recall[0])
    #     fscore[1] = (2*precision[1]*recall[1])/(precision[1]+recall[1])
    #     fscore[2] = (2*precision[2]*recall[2])/(precision[2]+recall[2])    
            
            
    #     return np.average(fscore)   
   