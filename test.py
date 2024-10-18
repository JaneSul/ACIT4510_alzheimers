import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
pd.set_option('display.max_columns', None)

df = pd.read_csv("data/alzheimers_disease_data.csv")
print(df.head())
RANDOM_STATE = 42

df.info()
df.describe().T

# Count duplicated rows in the DataFrame
sum(df.duplicated())

# Drop unnecessary columns from the DataFrame
df.drop(['PatientID', 'DoctorInCharge'], axis=1, inplace=True)
# Identify numerical columns: columns with more than 10 unique values are considered numerical
numerical_columns = [col for col in df.columns if df[col].nunique() > 10]

# Identify categorical columns: columns that are not numerical and not 'Diagnosis'
categorical_columns = df.columns.difference(numerical_columns).difference(['Diagnosis']).to_list()


# Custom labels for the categorical columns
custom_labels = {
    'Gender': ['Male', 'Female'],
    'Ethnicity': ['Caucasian', 'African American', 'Asian', 'Other'],
    'EducationLevel': ['None', 'High School', 'Bachelor\'s', 'Higher'],
    'Smoking': ['No', 'Yes'],
    'FamilyHistoryAlzheimers': ['No', 'Yes'],
    'CardiovascularDisease': ['No', 'Yes'],
    'Diabetes': ['No', 'Yes'],
    'Depression': ['No', 'Yes'],
    'HeadInjury': ['No', 'Yes'],
    'Hypertension': ['No', 'Yes'],
    'MemoryComplaints': ['No', 'Yes'],
'BehavioralProblems': ['No', 'Yes'],
    'Confusion': ['No', 'Yes'],
    'Disorientation': ['No', 'Yes'],
    'PersonalityChanges': ['No', 'Yes'],
    'DifficultyCompletingTasks': ['No', 'Yes'],
    'Forgetfulness': ['No', 'Yes']
}

# Plot countplots for each categorical column
for column in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=column)
    plt.title(f'Countplot of {column}')

    # Directly set custom labels
    labels = custom_labels[column]
    ticks = range(len(labels))
    plt.xticks(ticks=ticks, labels=labels)

    plt.show()

    # Plot histogram for each numerical column
    for column in numerical_columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(data=df, x=column, kde=True, bins=20)
        plt.title(f'Distribution of {column}')
        plt.show()

        # Compute Pearson correlation coefficients
        correlations = df.corr(numeric_only=True)['Diagnosis'][:-1].sort_values()

        # Set the size of the figure
        plt.figure(figsize=(20, 7))

        # Create a bar plot of the Pearson correlation coefficients
        ax = correlations.plot(kind='bar', width=0.7)

        # Set the y-axis limits and labels
        ax.set(ylim=[-1, 1], ylabel='Pearson Correlation', xlabel='Features',
               title=' Correlation with diagnosis')

        # Rotate x-axis labels for better readability
        ax.set_xticklabels(correlations.index, rotation=45, ha='right')

        plt.tight_layout()
        plt.show()



#Apply the Descition tree classification


from sklearn.model_selection import train_test_split

x =df.drop(columns=['Diagnosis'])
y = df['Diagnosis']
#Spliting the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
#Using Decision Tree Classifier and train it
from sklearn.tree import DecisionTreeClassifier

#intialize the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=RANDOM_STATE)

#Fitting the model on the training data
clf.fit(x_train, y_train)


#using the training model to make prediction on this data set
y_pred = clf.predict(x_test)

#Evaluate the Model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy:.4f}")
# Displaying the confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp= ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap= 'Blues')
plt.title("Confusion Matrix for Decision Tree")
plt.show()