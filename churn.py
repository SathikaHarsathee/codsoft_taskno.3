import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('Churn_Modelling.csv')

print(data.head())
print(data.info())


data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
data = pd.get_dummies(data, columns=['Geography', 'Gender'], drop_first=True)


X = data.drop('Exited', axis=1)
y = data['Exited']


X_data_train, X_data_test, y_data_train, y_data_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_data_train)
X_test_scaled = scaler.transform(X_data_test)

model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, min_samples_split=5)
model.fit(X_train_scaled, y_data_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_data_test, y_pred)
confusion = confusion_matrix(y_data_test, y_pred)
classification_rep = classification_report(y_data_test, y_pred)

print(f"Model: Random Forest Classifier")
print(f"Accuracy: {accuracy:.2f}")
print(f"Confusion Matrix:\n{confusion}")
print(f"Classification Report:\n{classification_rep}")

feature_importance = model.feature_importances_
feature_names = X.columns
df_feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
df_feature_importance = df_feature_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=df_feature_importance, palette='Purples_r')
plt.title('Feature Importance')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Dark2_r', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
