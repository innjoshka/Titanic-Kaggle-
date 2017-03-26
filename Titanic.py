import numpy as np
from sklearn import cross_validation, neighbors
from sklearn import tree
import pandas as pd

'''
Ivan Shelonik

The task https://www.kaggle.com/c/titanic

The training set should be used to build your machine learning models.
For the training set, we provide the outcome (also known as the “ground truth”)
for each passenger. Your model will be based on “features” like passengers’ gender
and class. You can also use feature engineering to create new features.

The test set should be used to see how well your model performs on unseen data.
For the test set, we do not provide the ground truth for each passenger.
It is your job to predict these outcomes. For each passenger in the test set,
use the model you trained to predict whether or not they survived the sinking of the Titanic.

Data Dictionary

Variable	Definition	Key
survival	Survival	0 = No, 1 = Yes
pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
sex	Sex
Age	Age in years
sibsp	# of siblings / spouses aboard the Titanic
parch	# of parents / children aboard the Titanic
ticket	Ticket number
fare	Passenger fare
cabin	Cabin number
embarked	Port of Embarkation	C = Cherbourg, Q = Queenstown, S = Southampton
'''


#Loading data
df_train = pd.read_csv('train.csv')


#creating our dataframes
df_train = df_train[['Survived',  'Pclass',  'Name',  'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]


df_train['Relatives'] = df_train['SibSp'] + df_train['Parch']
df_train['Fare_per_person'] = df_train['Fare'] / np.mean(df_train['SibSp'] + df_train['Parch'] + 1)

df_train.drop(['SibSp', 'Parch'],1, inplace=True)

df_train.Sex = pd.get_dummies(df_train.Sex,)

df_train.Age.fillna(value=df_train.Age.mean(), inplace=True)
	#the same as
		#df_train.Age.replace('NaN',Average_Age, inplace=True)

title_list = [
			'Dr', 'Mr', 'Master',
			'Miss', 'Major', 'Rev',
			'Mrs', 'Ms', 'Mlle','Col',
			'Capt', 'Mme', 'Countess',
			'Don', 'Jonkheer'
							]

#replacing all people's name by their titles
def replace_names_titles(x):
	for title in title_list:
		if title in x:
			return title

df_train['Title'] = df_train.Name.apply(replace_names_titles)
df_train['Title'] = df_train.Title.map({
										'Dr':1, 'Mr':2, 'Master':3, 'Miss':4, 'Major':5, 'Rev':6, 'Mrs':7, 'Ms':8, 'Mlle':9,
										'Col':10, 'Capt':11, 'Mme':12, 'Countess':13, 'Don': 14, 'Jonkheer':15
										})

df_train.drop(['Name'], 1, inplace=True)

# NaN rows in Cabin column are float type. To iterate over has to do all rows identical - string type !!!
df_train.Cabin.fillna(value='NaN', inplace=True)


cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'NaN']


def get_cabine(x, position):
	return x[position]

df_train['Deck'] = df_train['Cabin'].apply(get_cabine, position = 0)
		#or
#df_train['Deck'] = df_train.Cabin.apply(lambda x: x[0])


df_train['Deck'] = df_train.Deck.map({
										'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'T':7, 'G':8
										})

df_train.Deck.fillna('-99999',inplace=True)

df_train.drop(['Cabin'], 1, inplace=True)

df_train['Embarked'] = df_train['Embarked'].fillna('S')
df_train['Embarked'] = df_train.Embarked.map({
										'S':1, 'Q':2, 'C':3,
										})



#Creating Features & Labeles
X = np.array(df_train.drop(['Survived'], 1))
y = np.array(df_train['Survived'])


#Creating classifier and training the data
clf = tree.DecisionTreeClassifier()
clf.fit(X, y)







#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


#Loading data
df_test = pd.DataFrame(pd.read_csv('test.csv'))


#creating our dataframes
df_test = df_test[['Pclass',  'Name',  'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']]

df_test['Relatives'] = df_test['SibSp'] + df_test['Parch']
df_test.Fare.fillna(value=df_test.Fare.mean(), inplace=True)
df_test['Fare_per_person'] = df_test['Fare'] / np.mean(df_test['SibSp'] + df_test['Parch'] + 1)

df_test.drop(['SibSp', 'Parch'],1, inplace=True)

df_test.Sex = pd.get_dummies(df_test.Sex, prefix='is_')

df_test.Age.fillna(value=df_test.Age.mean(), inplace=True)

title_list = [
			'Dr', 'Mr', 'Master',
			'Miss', 'Major', 'Rev',
			'Mrs', 'Ms', 'Mlle','Col',
			'Capt', 'Mme', 'Countess',
			'Don', 'Jonkheer'
							]

#replacing all people's name by their titles
def replace_names_titles(x):
	for title in title_list:
		if title in x:
			return title

df_test['Title'] = df_test.Name.apply(replace_names_titles)
df_test['Title'] = df_test.Title.map({
										'Dr':1, 'Mr':2, 'Master':3, 'Miss':4, 'Major':5, 'Rev':6, 'Mrs':7, 'Ms':8, 'Mlle':9,
										'Col':10, 'Capt':11, 'Mme':12, 'Countess':13, 'Don': 14, 'Jonkheer':15
										})

df_test.drop(['Name'], 1, inplace=True)


# NaN rows in Cabin column are float type. To iterate over has to do all rows identical - string type !!!
df_test.Cabin.fillna(value='NaN', inplace=True)

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'NaN']

def get_cabine(x, position):
	return x[position]

df_test['Deck'] = df_test['Cabin'].apply(get_cabine, position = 0)

df_test['Deck'] = df_test.Deck.map({
										'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'T':7, 'G':8
										})

df_test.Deck.fillna('-99999',inplace=True)

df_test.drop(['Cabin'], 1, inplace=True)

df_test['Embarked'] = df_test['Embarked'].fillna('S')
df_test['Embarked'] = df_test.Embarked.map({
										'S':1, 'Q':2, 'C':3,
										})

#Predicting who would survive in Titanic Disaster from the given list of passengers.
prediction = (clf.predict(df_test))
print('Predictions: where 0 - not survived and 1 - survived \n',prediction)


submission = pd.DataFrame({
	"PassengerId": df_test.index+892,
	"Survived": prediction
	})

submission.to_csv("gender_submission.csv", index=False)
