import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib

data = pd.read_csv("final_test.csv")
# data.info()

#here we are going to clean the data to make sure that there is no null values and duplicates values
data.dropna(inplace=True)
# data.info()

###NOW our data set is well cleaned There is no null vallues and missing values
plt.figure(figsize=(15,8))
sns.heatmap(data.corr(numeric_only=True),annot=True, cmap='YlGnBu')
# plt.show()


###Another step its to convert the size into the numeric values
# df = pd.get_dummies(data, columns=['size'], drop_first=True) ###there is no need to dummies those data
# df.hist()
# plt.show()
# df.info()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

label_data = LabelEncoder()
data['size'] = label_data.fit_transform(data['size']) ## This stand before dummies

X = data[['weight', 'height', 'age']]  
y = data['size'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)  
model.fit(X_train, y_train)

model_data = {
    "model":model,
    "label_data":label_data
}
joblib.dump(model_data, 'mymodel.joblib')

weight_value = int(input("Enter your weight: "))
height_value = float(input("Enter your height: "))
age_value = int(input("Enter your age: "))

new_data = [[weight_value, height_value, age_value]]  
new_data_df = pd.DataFrame(data=new_data, columns=["weight", "height", "age"])  
predicted_size = model.predict(new_data_df)[0]
predicted_size_label = label_data.inverse_transform([predicted_size])[0]

print(f"Predicted size for the new data point: {predicted_size_label}")

