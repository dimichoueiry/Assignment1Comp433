import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# [X] (a) Use statistical methods and graphs/plots to describe your daataset.

#Load data
data = pd.read_csv('Copy of lung cancer dataset.csv')
print(data.head())

import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(dataframe):
    """
    This function takes a pandas dataframe, converts any categorical columns to numeric values if necessary,
    computes the correlation matrix, and plots a heatmap of the correlations.
    
    Args:
    dataframe (pd.DataFrame): Input pandas dataframe with columns that may include categorical data.
    
    Returns:
    None: The function displays the heatmap directly.
    """
    # Create a copy of the dataframe to avoid modifying the original one
    data_numeric = dataframe.copy()

    # Convert categorical columns to numeric values where necessary
    data_numeric['GENDER'] = data_numeric['GENDER'].apply(lambda x: 1 if x == 'M' else 0)
    data_numeric['LUNG_CANCER'] = data_numeric['LUNG_CANCER'].apply(lambda x: 1 if x == 'YES' else 0)

    # Calculate the correlation matrix
    corr_matrix = data_numeric.corr()

    # Plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

def stat_description(dataframe):
    """
    This function takes a pandas dataframe and displays the statistical description of the data.
    
    Args:
    dataframe (pd.DataFrame): Input pandas dataframe.
    
    Returns:
    None: The function prints the statistical description directly.
    """
    print("\n**Description of data**: ", dataframe.describe())
    
    print("\n**Gender Description**: ", dataframe['GENDER'].value_counts())
    
    print("\n**Lung Cancer Description**: ", dataframe['LUNG_CANCER'].value_counts())

def all_plots(dataframe):
    # Set up the figure for multiple histograms
    plt.figure(figsize=(14, 10))

    # Plot histograms for all numeric columns
    dataframe.hist(bins=10, figsize=(14, 10), layout=(4, 4), edgecolor='black')

    # Display the plots
    plt.tight_layout()
    plt.show()

def show_smokers_and_lungCancer():
    # Clean up the data for better readability
    data['SMOKING'] = data['SMOKING'].replace({1: 'Non-smoker', 2: 'Smoker'})
    data['LUNG_CANCER'] = data['LUNG_CANCER'].replace({'YES': 'Has Cancer', 'NO': 'No Cancer'})

    # Group by age, smoking, and lung cancer
    grouped_data = data.groupby(['AGE', 'SMOKING', 'LUNG_CANCER']).size().reset_index(name='count')

    # Pivot table to separate smokers/non-smokers and cancer/no cancer
    pivot_data = grouped_data.pivot_table(index='AGE', columns=['SMOKING', 'LUNG_CANCER'], values='count', fill_value=0)

    # Create a figure with one subplot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot stacked bars
    pivot_data.plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

    # Adding labels and title
    ax.set_title('Age vs Smoking and Cancer Status')
    ax.set_xlabel('Age')
    ax.set_ylabel('Number of People')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.legend(title='Smoking and Cancer Status')
    plt.tight_layout()

    # Show the plot
    plt.show()


##### uncomment Later#####
stat_description(data)
plot_correlation_heatmap(data)
all_plots(data)
show_smokers_and_lungCancer()




#[X] (b) Split your dataset into train and test sets with a 7:3 ratio. Use the train_test_split tool from scikit-learn.

# Split the data into features and target
X = data.drop(columns=['LUNG_CANCER'])
y = data['LUNG_CANCER']

# Convert categorical columns before train-test split
X['GENDER'] = X['GENDER'].apply(lambda x: 1 if x == 'M' else 0)
X['SMOKING'] = X['SMOKING'].apply(lambda x: 1 if x == 'Smoker' else 0)
# If other categorical columns exist, you need to handle them similarly


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\n**Training Data**")

print("\nX_train:")
print(X_train.head())

print("\ny_train:")
print(y_train.head())

print("\n**Testing Data**")

print("\nX_test:")
print(X_test.head())

print("\ny_test:")
print(y_test.head())

#[X] (c) Build and train a Logistic Regression model using scikit-learn. Explore the
# parameters of the model in scikit-learn, and aim for higher classification accuracies.

# Import the Logistic Regression model from scikit-learn
from sklearn.linear_model import LogisticRegression

# Create a Logistic Regression model
model = LogisticRegression(max_iter=500, C=10)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
test_accuracy = model.score(X_test, y_test)
print("\n**Model Accuracy**")
print(f"Accuracy: {test_accuracy * 100:.2f}%")

train_accuracy = model.score(X_train, y_train)
print("**Train accuracy**")
print(f"Accuracy: {train_accuracy * 100:.2f}%")

# [X] (d) Report and discuss the performance (not just accuracy) of your developed model on both the
# train and test sets (separately). You can use scikit-learn’s classification report tool 

#scikit-learn's classification report tool
from sklearn.metrics import classification_report

# Classification report for the test set
print("\n**Classification Report for Test Set**")
print(classification_report(y_test, y_pred))

# Classification report for the training set
y_pred_train = model.predict(X_train)
print("\n**Classification Report for Training Set**")
print(classification_report(y_train, y_pred_train))
