# Multinomial-Naive-Bayes-Classifier
<p>Python implementation of multinomial naive bayes classifier for :<br>
1. Binary Text Classification of positive and negative book review files.<br>
2. Multiclass Text Classification for <b>20 Newsgroups dataset</b>  <a>http://qwone.com/~jason/20Newsgroups/</a>.</p>  

## Binary Text Classification
There are two data files in the package: positive.review and negative.review. They correspond to
positive and negative book reviews. The text has been preprocessed so that each line contains a review
document; each token (e.g., year:2) represents a word and its frequency in the document. The last token
(e.g., #label#:negative) in each line indicates the polarity (label) of the document.  
**On executing the run_classifier.py , it will return the following results:**  
Accuracy
on training set: 0.972500, on test set: 0.835000

## MultiClass Text Classification  
<p>Using three categories:comp.windows.x, rec.autos, and talk.politics.guns of the **20 Newsgroups dataset** classification.<a>http://qwone.com/~jason/20Newsgroups/</a> for a multiclass text. 
<br><br>This folder is providing the implementation of a multinomial naive Bayes classifier for text classification.</p>  
**On executing the preprocessing_type_data.py , it will return the following results:**  
Accuracy on training set: 0.993653, on test set: 0.980087
Macro Averaged F1 score : 0.980158


