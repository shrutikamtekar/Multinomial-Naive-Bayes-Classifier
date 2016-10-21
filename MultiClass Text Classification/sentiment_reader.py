import codecs
import numpy as np

class SentimentCorpus:
    
    def __init__(self):
        '''
        prepare dataset
        1) build feature dictionaries
        2) split data into train/dev/test sets 
        '''
        
        train_X, train_y, feat_dict, feat_counts = build_dicts_train()
        test_X, test_y = build_dicts_test(feat_dict) #for testing set same dictonary
        self.nr_instances_train = train_y.shape[0]
        self.nr_features_train = train_X.shape[1]
        self.nr_instances_test = test_y.shape[0]
        self.nr_features_test = test_X.shape[1]
        self.feat_dict = feat_dict
        self.feat_counts = feat_counts
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        


def build_dicts_train():
    '''
    builds feature dictionaries
    ''' 
    feat_counts = {}

    # build feature dictionary with counts
    nr_win = 0
    with codecs.open("train_windows.review", 'r', 'utf8') as windows_file:
        for line in windows_file:
            nr_win += 1
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name not in feat_counts:
                    feat_counts[name] = 0
                feat_counts[name] += int(counts)
    nr_auto = 0
    with codecs.open("train_autos.review", 'r', 'utf8') as auto_file:
        for line in auto_file:
            nr_auto += 1
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name not in feat_counts:
                    feat_counts[name] = 0
                feat_counts[name] += int(counts)
    nr_pol = 0
    with codecs.open("train_politics.review", 'r', 'utf8') as politics_file:
        for line in politics_file:
            nr_pol += 1
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name not in feat_counts:
                    feat_counts[name] = 0
                feat_counts[name] += int(counts)

    # remove all features that occur less than 5 (threshold) times
    to_remove = []
    for key, value in feat_counts.iteritems():
        if value < 5:
            to_remove.append(key)
    for key in to_remove:
        del feat_counts[key]
        
   
    # map feature to index
    feat_dict = {}
    i = 0
    for key in feat_counts.keys():
        feat_dict[key] = i
        i += 1

    nr_feat = len(feat_counts) 
    nr_instances = nr_win + nr_auto + nr_pol
    X = np.zeros((nr_instances, nr_feat), dtype=float)
    y = np.vstack((np.zeros([nr_win,1], dtype=int), np.ones([nr_auto,1], dtype=int), 2*np.ones([nr_pol,1], dtype=int)))
    
    with codecs.open("train_windows.review", 'r', 'utf8') as windows_file:
        nr_win = 0
        for line in windows_file:
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name in feat_dict:
                    X[nr_win,feat_dict[name]] = int(counts)
            nr_win += 1
        
    with codecs.open("train_autos.review", 'r', 'utf8') as auto_file:
        nr_auto = 0
        for line in auto_file:
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name in feat_dict:
                    X[nr_win+nr_auto,feat_dict[name]] = int(counts)
            nr_auto += 1
    
    with codecs.open("train_politics.review", 'r', 'utf8') as politics_file:
        nr_pol = 0
        for line in politics_file:
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name in feat_dict:
                    X[nr_win+nr_auto+nr_pol,feat_dict[name]] = int(counts)
            nr_pol += 1
            
       
    # shuffle the order, mix windows,autos and politics examples
    new_order = np.arange(nr_instances)
    np.random.seed(0) # set seed
    np.random.shuffle(new_order)
    X = X[new_order,:]
    y = y[new_order,:]

   
    return X, y, feat_dict, feat_counts

def build_dicts_test(feat_dict):
  
    
    # counting number of dcuments per category
    nr_win = 0
    with codecs.open("test_windows.review", 'r', 'utf8') as windows_file:
        for line in windows_file:
            nr_win += 1

    nr_auto = 0
    with codecs.open("test_autos.review", 'r', 'utf8') as auto_file:
        for line in auto_file:
            nr_auto += 1

    nr_pol = 0
    with codecs.open("test_politics.review", 'r', 'utf8') as politics_file:
        for line in politics_file:
            nr_pol += 1

    nr_feat = len(feat_dict)
    nr_instances = nr_win + nr_auto + nr_pol
    X = np.zeros((nr_instances, nr_feat), dtype=float)
    y = np.vstack((np.zeros([nr_win,1], dtype=int), np.ones([nr_auto,1], dtype=int), 2*np.ones([nr_pol,1], dtype=int)))
    
    with codecs.open("test_windows.review", 'r', 'utf8') as windows_file:
        nr_win = 0
        for line in windows_file:
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name in feat_dict:
                    X[nr_win,feat_dict[name]] = int(counts)
            nr_win += 1
        
    with codecs.open("test_autos.review", 'r', 'utf8') as auto_file:
        nr_auto = 0
        for line in auto_file:
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name in feat_dict:
                    X[nr_win+nr_auto,feat_dict[name]] = int(counts)
            nr_auto += 1
    
    with codecs.open("test_politics.review", 'r', 'utf8') as politics_file:
        nr_pol = 0
        for line in politics_file:
            toks = line.split(" ")
            for feat in toks[0:-1]:
                name, counts = feat.split(":")
                if name in feat_dict:
                    X[nr_win+nr_auto+nr_pol,feat_dict[name]] = int(counts)
            nr_pol += 1
            
  # shuffle the order, mix windows,autos and politics examples
    new_order = np.arange(nr_instances)
    np.random.seed(0) # set seed
    np.random.shuffle(new_order)
    X = X[new_order,:]
    y = y[new_order,:]

   
    return X, y







