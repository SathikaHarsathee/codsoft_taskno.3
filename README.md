This Python script utilizes a Random Forest Classifier to predict churn in a dataset ('Churn_Modelling.csv').
It begins by loading and exploring the dataset, dropping unnecessary columns, and applying one-hot encoding to categorical features.
The dataset is then split into training and testing sets, and numerical features are standardized using StandardScaler.
A Random Forest Classifier is trained on the scaled training data, and predictions are made on the test set.
The script calculates and displays accuracy, a confusion matrix, and a classification report to assess model performance.
Additionally, it analyzes feature importance, presenting a bar plot showcasing the importance of each feature.
Finally, a heatmap visualizes the confusion matrix, providing insights into the distribution of true and predicted labels.
Overall, the script offers a comprehensive approach to churn prediction, incorporating data preprocessing, model training, evaluation metrics, and insightful visualizations.
