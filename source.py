import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from bs4 import BeautifulSoup
from time import time
from sklearn import metrics



dataset = pd.read_csv('/home/ammar/Data/COMPUTER SCIENCE BSCS/Semester 7/IR/assignment/IR_Assi3/Datasets/Question2 Dataset.tsv',delimiter="\t", encoding='utf-8')

X = dataset.iloc[:, 2].values
y = dataset.iloc[:, 1].values

index=0
#vectorizer = TfidfVectorizer()
#    sublinear_tf=True, max_df=0.5,stop_words='english')
#X_train = 


for x in X:
  soup = BeautifulSoup(x, 'html.parser')
  X[index]=soup.get_text()
  #temp=soup.get_text()
  #print(temp)
  index+=1
#print(y[0:10])



correct=False
vectorizer=None
option=input('Enter 1 for Raw Count or\n 2 for Tfidf:')
while(not correct):
  if option=='1':
    vectorizer = CountVectorizer()
    correct=True
  elif option=='2':
    vectorizer=TfidfVectorizer()
    correct=True
  else:
    print('You entered Incorrect Option\n')
    option=input('Enter 1 for Raw Count or\n 2 for Tfidf:')

#print(vectorizer.get_feature_names())

X = vectorizer.fit_transform(X)

#print(X[0:15])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


#print(Xtemp)
print(X_test.shape)
print(X_train.shape)
print(y_test.shape)
print(y_train.shape)

def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    return score, train_time, test_time




results=[]

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
#results.append(benchmark(BernoulliNB(alpha=.01)))
#results.append(benchmark(ComplementNB(alpha=.1)))


print(results)
