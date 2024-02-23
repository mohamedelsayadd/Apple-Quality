# 1 - import packges : 
import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns

# 2 - reed the file : 
df = pd.read_csv('C:\\Users\\moham\\Desktop\\VScode\\ML Projects\\Apple Quality\\Apple_quality.csv')
df = pd.DataFrame(df)

# 3 - get some informaition about the data :
# print(df.head(5))
# print(df.describe())
# print(df.shape)
# print(df.dtypes)
# print(df.info())
# print(df.duplicated().sum()) 
# print(df.columns)

# 3 - Data Preprocessing :
df.dropna(inplace=True)  # remove rows with missing values
df['Acidity'] = df['Acidity'].astype(float)
df= df.drop(columns=['A_id'])
from sklearn.preprocessing import LabelEncoder
col = [ 'Quality' ]
df[col] = df[col].apply(LabelEncoder().fit_transform)


# function for get the num of outliers  in each column.
def detect_outliers_zscore(data, threshold=3):
    num_outliers = {}
    for column in data.columns:
        col_data = data[column]
        z_scores = (col_data - np.mean(col_data)) / np.std(col_data)
        num_outliers[column] = np.sum(np.abs(z_scores) > threshold)
    return num_outliers
outliers_count = detect_outliers_zscore(df)
# print("Number of outliers for each column:", outliers_count)


# remove the outliers 
threshold = 3
for column in df.columns:
    col_data = df[column]
    z_scores = (col_data - np.mean(col_data)) / np.std(col_data)
    outliers_mask = np.abs(z_scores) > threshold
    df = df[~outliers_mask]
outliers_count = detect_outliers_zscore(df)
# print("Number of outliers for each column:", outliers_count)

# 4 - split the data :
X = df.drop(columns='Quality')
Y = df['Quality']
# print(X.shape)
# print(Y.shape)

# 5- train the data :
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20 , random_state=3)


# 6 - import the packges for models : 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 7 - choose the best model : 
def find_best_model(X, Y):
    models = {
        'logistic_regression': {
            'model': LogisticRegression(solver='lbfgs', multi_class='auto'),
            'parameters': {
                'C': [1,5,10]
               }
        },
        
        'decision_tree': {
            'model': DecisionTreeClassifier(splitter='best'),
            'parameters': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [5,10]
            }
        },
        
        'random_forest': {
            'model': RandomForestClassifier(criterion='gini'),
            'parameters': {
                'n_estimators': [10,15,20,50,100,200]
            }
        },
        
        'svm': {
            'model': SVC(gamma='auto'),
            'parameters': {
                'C': [1,10,20],
                'kernel': ['rbf','linear']
            }
        }

    }
    
    scores = [] 
    cv_shuffle = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
        
    for model_name, model_params in models.items():
        gs = GridSearchCV(model_params['model'], model_params['parameters'], cv = cv_shuffle, return_train_score=False)
        gs.fit(X, Y)
        scores.append({
            'model': model_name,
            'best_parameters': gs.best_params_,
            'score': gs.best_score_
        })
        
    return pd.DataFrame(scores, columns=['model','best_parameters','score'])
# print(find_best_model(x_train, y_train))

# 8 - fit the model : 
svm_model = SVC(C=20, kernel='rbf', gamma='auto')
svm_model.fit(x_train, y_train)
y_pred = svm_model.predict(x_test)

#9 - Evaluate classification model performance :
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
# print(cm)
sns.heatmap(cm,annot=True)
# plt.show()

# 10 - save the model :
import pickle
pickle.dump(svm_model,open('apple.pkl','wb'))
model = pickle.load(open( 'apple.pkl', 'rb' ))



# now we can try the model : 
prediction_data = pd.DataFrame(data=np.array([-2,0.13,-1,2,-2,0.2,-2]).reshape(1,7))
prediction = model.predict(prediction_data)
if prediction:
  print('Positive')
else:
  print("Negative")