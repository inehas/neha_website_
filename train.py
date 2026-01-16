import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE # You might need to pip install imbalanced-learn
import pickle

# 1. Load Data
df = pd.read_csv('data/churn.csv')

# 2. Preprocessing
# Drop irrelevant columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode Categorical Variables
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Geography'] = le.fit_transform(df['Geography'])

# 3. Feature Engineering (Simple Interaction)
df['BalanceSalaryRatio'] = df['Balance'] / df['EstimatedSalary']

# 4. Split Data
X = df.drop('Exited', axis=1)
y = df['Exited']

# Handle Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 5. Train Model (XGBoost)
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 6. Save Model
with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")