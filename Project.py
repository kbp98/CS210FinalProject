import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

input_file = "Data.csv"
output_file = "Cleaned_Data.csv"
data = pd.read_csv(input_file)

columns_to_drop = [
    'INDICATOR', 'PANEL_NUM', 'UNIT', 'UNIT_NUM',
    'STUB_NAME', 'STUB_NAME_NUM', 'STUB_LABEL_NUM',
    'YEAR_NUM', 'AGE_NUM', 'FLAG'
]
data_cleaned = data.drop(columns=columns_to_drop)
data_cleaned = data_cleaned.dropna(subset=['ESTIMATE'])
data_cleaned.to_csv(output_file, index=False)

print(data_cleaned)

data = pd.read_csv(output_file)
age_data = data[['YEAR', 'AGE', 'ESTIMATE']]
grouped_data = age_data.groupby(['YEAR', 'AGE'], as_index=False)['ESTIMATE'].sum()

pivot_data = grouped_data.pivot(index='YEAR', columns='AGE', values='ESTIMATE')
pivot_data = pivot_data.drop(columns=['All ages'], errors='ignore')

plt.figure(figsize=(14, 8))
for age_group in pivot_data.columns:
    plt.plot(
        pivot_data.index,
        pivot_data[age_group],
        label=age_group,
        marker='o'  
    )

plt.xlabel('Year')
plt.ylabel('Deaths per 100,000 Population')
plt.title('Drug Overdose Deaths by Age Group (Including All Ages Baseline)')
plt.legend(title='Age Group', loc='center left', bbox_to_anchor=(1, 0.5))  # Adjust legend placement
plt.grid(True)
plt.tight_layout()
plt.show()

pivot_data = pivot_data.drop(columns=['All ages'], errors='ignore')
age_group_increases = pivot_data.iloc[-1] - pivot_data.iloc[0]

largest_increase_group = age_group_increases.idxmax()
largest_increase_value = age_group_increases.max()

print(f"The age group with the largest increase in overdose deaths is: {largest_increase_group}")
print(f"Largest increase: {largest_increase_value} deaths per 100,000 population")


data = pd.read_csv("Cleaned_Data.csv")
race_data = data[['YEAR', 'STUB_LABEL', 'ESTIMATE']].copy() 

race_mapping = {
    'Male: White': 'White',
    'Female: White': 'White',
    'Male: Black or African American': 'Black or African American',
    'Female: Black or African American': 'Black or African American',
    'Male: Hispanic or Latino: All races': 'Hispanic or Latino',
    'Female: Hispanic or Latino: All races': 'Hispanic or Latino',
    'Male: Asian or Pacific Islander': 'Asian or Pacific Islander',
    'Female: Asian or Pacific Islander': 'Asian or Pacific Islander',
    'All persons': 'All persons'  
}

race_data = race_data.assign(Normalized_Race=race_data['STUB_LABEL'].map(race_mapping))
race_data = race_data.dropna(subset=['Normalized_Race'])
grouped_race_data = race_data.groupby(['YEAR', 'Normalized_Race'], as_index=False)['ESTIMATE'].sum()
pivot_race_data = grouped_race_data.pivot(index='YEAR', columns='Normalized_Race', values='ESTIMATE')

plt.figure(figsize=(14, 8))
for race_group in pivot_race_data.columns:
    plt.plot(
        pivot_race_data.index,
        pivot_race_data[race_group],
        label=race_group,
        marker='o'  
    )

plt.xlabel('Year')
plt.ylabel('Deaths per 100,000 Population')
plt.title('Drug Overdose Deaths by Race and Ethnicity')
plt.legend(title='Race/Ethnicity', loc='center left', bbox_to_anchor=(1, 0.5))  # Adjust legend position
plt.grid(True)
plt.tight_layout()
plt.show()

total_race_data = race_data.groupby('Normalized_Race')['ESTIMATE'].sum().reset_index()
most_vulnerable_race = total_race_data.loc[total_race_data['ESTIMATE'].idxmax()]
race_name = most_vulnerable_race['Normalized_Race']
race_total = most_vulnerable_race['ESTIMATE']

print(f"The race/ethnicity most vulnerable to overdoses is: {race_name}")
print(f"Total overdose deaths: {race_total:.2f} deaths per 100,000 population")


data = pd.read_csv("Cleaned_Data.csv")
gender_data = data[['YEAR', 'STUB_LABEL', 'ESTIMATE']].copy()  # Use .copy() to avoid SettingWithCopyWarning

gender_mapping = {
    'Male': 'Male',
    'Female': 'Female',
    'Male: White': 'Male',
    'Female: White': 'Female',
    'Male: Black or African American': 'Male',
    'Female: Black or African American': 'Female',
    'Male: Hispanic or Latino: All races': 'Male',
    'Female: Hispanic or Latino: All races': 'Female',
    'Male: Asian or Pacific Islander': 'Male',
    'Female: Asian or Pacific Islander': 'Female',
    'Male: Not Hispanic or Latino: White': 'Male',
    'Female: Not Hispanic or Latino: White': 'Female',
    'All persons': 'Unknown',
    'Male: Not Hispanic or Latino: Black': 'Male',
    'Female: Not Hispanic or Latino: Black': 'Female',
    'Male: Not Hispanic or Latino: Asian or Pacific Islander': 'Male',
    'Female: Not Hispanic or Latino: Asian or Pacific Islander': 'Female',
    'Male: American Indian or Alaska Native': 'Male',
    'Female: American Indian or Alaska Native': 'Female',
    'Male: Not Hispanic or Latino: American Indian or Alaska Native': 'Male',
    'Female: Not Hispanic or Latino: American Indian or Alaska Native': 'Female',
    'Male: Hispanic or Latino': 'Male',
    'Female: Hispanic or Latino': 'Female',
}

gender_data['Gender'] = gender_data['STUB_LABEL'].map(gender_mapping).fillna('Unknown')
grouped_gender_data = gender_data.groupby(['YEAR', 'Gender'], as_index=False)['ESTIMATE'].sum()
pivot_gender_data = grouped_gender_data.pivot(index='YEAR', columns='Gender', values='ESTIMATE')

plt.figure(figsize=(12, 8))
for gender in pivot_gender_data.columns:
    plt.plot(
        pivot_gender_data.index,
        pivot_gender_data[gender],
        label=gender,
        marker='o'
    )

plt.xlabel('Year')
plt.ylabel('Deaths per 100,000 Population')
plt.title('Drug Overdose Deaths by Gender')
plt.legend(title='Gender', loc='center left', bbox_to_anchor=(1, 0.5))  # Adjust legend placement
plt.grid(True)
plt.tight_layout()
plt.show()

filtered_gender_data = gender_data[gender_data['Gender'] != 'Unknown']
total_gender_data = filtered_gender_data.groupby('Gender')['ESTIMATE'].sum().reset_index()

most_vulnerable_gender = total_gender_data.loc[total_gender_data['ESTIMATE'].idxmax()]
gender_name = most_vulnerable_gender['Gender']
gender_total = most_vulnerable_gender['ESTIMATE']

print(f"The gender most vulnerable to overdoses is: {gender_name}")
print(f"Total overdose deaths: {gender_total:.2f} deaths per 100,000 population")



data = pd.read_csv("Cleaned_Data.csv")
drug_data = data[['YEAR', 'PANEL', 'ESTIMATE']]
grouped_drug_data = drug_data.groupby(['YEAR', 'PANEL'], as_index=False)['ESTIMATE'].sum()
pivot_drug_data = grouped_drug_data.pivot(index='YEAR', columns='PANEL', values='ESTIMATE')

plt.figure(figsize=(14, 8))
for panel_type in pivot_drug_data.columns:
    plt.plot(
        pivot_drug_data.index,
        pivot_drug_data[panel_type],
        label=panel_type,
        marker='o'
    )

plt.xlabel('Year')
plt.ylabel('Deaths per 100,000 Population')
plt.title('Drug Overdose Deaths by Type of Drug')
plt.legend(title='Drug Type', loc='center left', bbox_to_anchor=(1, 0.5))  # Adjust legend placement
plt.grid(True)
plt.tight_layout()
plt.show()

filtered_drug_data = grouped_drug_data[grouped_drug_data['PANEL'] != 'All drug overdose deaths']
total_drug_data = filtered_drug_data.groupby('PANEL')['ESTIMATE'].sum().reset_index()

most_vulnerable_drug = total_drug_data.loc[total_drug_data['ESTIMATE'].idxmax()]
drug_name = most_vulnerable_drug['PANEL']
drug_total = most_vulnerable_drug['ESTIMATE']

print(f"The drug type causing the highest overdose deaths is: {drug_name}")
print(f"Total overdose deaths: {drug_total:.2f} deaths per 100,000 population")



data = pd.read_csv("Cleaned_Data.csv")

def split_stub_label(label):
    if "Male" in label or "Female" in label:
        if ":" in label:
            parts = label.split(": ")
            return parts[0], parts[1]
        elif "years" in label:
            parts = label.split(": ", maxsplit=1)
            return parts[0], parts[1]
    return "Other", label

data[['Gender', 'Demographic']] = data['STUB_LABEL'].apply(split_stub_label).apply(pd.Series)

data['Age_Group'] = data['Demographic'].apply(lambda x: x if "years" in x else None)
data['Race/Ethnicity'] = data['Demographic'].apply(lambda x: None if "years" in x else x)

data = data.copy()
data['Age_Group'] = data['Age_Group'].fillna('All Ages')
data['Race/Ethnicity'] = data['Race/Ethnicity'].fillna('All Races')

cleaned_data = data[['YEAR', 'PANEL', 'Gender', 'Age_Group', 'Race/Ethnicity', 'ESTIMATE']]
grouped_data = cleaned_data.groupby(['YEAR', 'Gender', 'Age_Group', 'Race/Ethnicity', 'PANEL'], as_index=False)['ESTIMATE'].sum()
gender_group = grouped_data.groupby(['Gender', 'YEAR'], as_index=False)['ESTIMATE'].sum()

vulnerable_group = grouped_data.groupby(['Gender', 'Age_Group', 'Race/Ethnicity'], as_index=False)['ESTIMATE'].sum()
vulnerable_group = vulnerable_group.sort_values(by='ESTIMATE', ascending=False)

print("Most Vulnerable Groups by Overdose Deaths:")
print(vulnerable_group)

top_vulnerable = vulnerable_group
demographic_labels = top_vulnerable.apply(lambda row: f"{row['Gender']}, {row['Age_Group']}, {row['Race/Ethnicity']}", axis=1)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_vulnerable,
    x='ESTIMATE',
    y=demographic_labels,
    palette="viridis",
    hue=demographic_labels,
    dodge=False
)
plt.title("Top 10 Most Vulnerable Groups by Overdose Deaths")
plt.xlabel("Deaths per 100,000 Population")
plt.ylabel("Demographic Group")
plt.legend([],[], frameon=False)
plt.tight_layout()
plt.show()




data = pd.read_csv("Cleaned_Data.csv")

filtered_data = data[
    (data['PANEL'] == 'All drug overdose deaths') &
    (data['STUB_LABEL'] == 'Male: 35-44 years')
]

filtered_data = filtered_data.groupby('YEAR')['ESTIMATE'].sum().reset_index()

def create_lagged_features(data, lag=3):
    X, y = [], []
    for i in range(len(data) - lag):
        X.append(data[i:i+lag])  
        y.append(data[i+lag])   
    return np.array(X), np.array(y)

lag = 3
X, y = create_lagged_features(filtered_data['ESTIMATE'].values, lag)

split_index = len(X) - 2 
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

class LinearRegressionManual:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        self.weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X.dot(self.weights)

model = LinearRegressionManual()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

future_years = np.arange(2018, 2026) 
last_known = filtered_data['ESTIMATE'].values[-lag:] 
future_forecasts = []

for _ in future_years:
    next_value = model.predict(last_known.reshape(1, -1))[0]
    future_forecasts.append(next_value)
    last_known = np.roll(last_known, -1)  
    last_known[-1] = next_value

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"Train MAE: {train_mae:.2f}")
print(f"Test MAE: {test_mae:.2f}")

plt.figure(figsize=(12, 6))

plt.plot(filtered_data['YEAR'], filtered_data['ESTIMATE'], label='Actual Data', marker='o')

train_years = filtered_data['YEAR'][lag:split_index+lag]
plt.plot(train_years, y_pred_train, label='Train Predictions', linestyle='--')

test_years = filtered_data['YEAR'][split_index+lag:split_index+lag+len(y_pred_test)]
plt.plot(test_years, y_pred_test, label='Test Predictions', linestyle='--')

plt.plot(future_years, future_forecasts, label='Forecasts', marker='x')

plt.xlabel('Year')
plt.ylabel('Deaths per 100,000 Population')
plt.title('Linear Regression Model: Overdose Death Predictions for Male: 35-44 years (All Drug Overdose Deaths)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



data = pd.read_csv("Cleaned_Data.csv")

filtered_data = data[
    (data['PANEL'] == 'All drug overdose deaths') &
    (data['STUB_LABEL'] == 'Male: 35-44 years')
]

filtered_data = filtered_data.groupby('YEAR')['ESTIMATE'].sum().reset_index()

def create_polynomial_features(data, degree=3):
    X = np.array(data['YEAR']).reshape(-1, 1)
    X_poly = np.hstack([X**i for i in range(degree + 1)])
    return X_poly, np.array(data['ESTIMATE'])

degree = 3 
X_poly, y = create_polynomial_features(filtered_data, degree)

split_index = len(filtered_data) - 2
X_train, X_test = X_poly[:split_index], X_poly[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

class PolynomialRegressionManual:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        return X @ self.weights

model_poly = PolynomialRegressionManual()
model_poly.fit(X_train, y_train)

y_pred_train = model_poly.predict(X_train)
y_pred_test = model_poly.predict(X_test)

future_years = np.arange(2018, 2026).reshape(-1, 1)
future_poly_features = np.hstack([future_years**i for i in range(degree + 1)])
future_forecasts = model_poly.predict(future_poly_features)

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"Train MAE: {train_mae:.2f}")
print(f"Test MAE: {test_mae:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(filtered_data['YEAR'], filtered_data['ESTIMATE'], label='Actual Data', marker='o')
plt.plot(filtered_data['YEAR'][:split_index], y_pred_train, label='Train Predictions', linestyle='--')
plt.plot(filtered_data['YEAR'][split_index:], y_pred_test, label='Test Predictions', linestyle='--')
plt.plot(future_years.flatten(), future_forecasts, label='Forecasts', marker='x')
plt.xlabel('Year')
plt.ylabel('Deaths per 100,000 Population')
plt.title('Polynomial Regression: Overdose Death Predictions for Male: 35-44 years')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



def create_lagged_features(data, lag=3):
    X, y = [], []
    for i in range(len(data) - lag):
        X.append(data[i:i+lag])  
        y.append(data[i+lag])   
    return np.array(X), np.array(y)

lag = 3
X_ar, y_ar = create_lagged_features(filtered_data['ESTIMATE'].values, lag)

split_index = len(X_ar) - 2
X_train_ar, X_test_ar = X_ar[:split_index], X_ar[split_index:]
y_train_ar, y_test_ar = y_ar[:split_index], y_ar[split_index:]

class ARModelManual:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        return X @ self.weights

model_ar = ARModelManual()
model_ar.fit(X_train_ar, y_train_ar)

y_pred_train_ar = model_ar.predict(X_train_ar)
y_pred_test_ar = model_ar.predict(X_test_ar)

last_known = filtered_data['ESTIMATE'].values[-lag:]
future_forecasts_ar = []
for _ in range(len(future_years)):
    next_value = model_ar.predict(last_known.reshape(1, -1))[0]
    future_forecasts_ar.append(next_value)
    last_known = np.roll(last_known, -1)
    last_known[-1] = next_value

train_mae_ar = mean_absolute_error(y_train_ar, y_pred_train_ar)
test_mae_ar = mean_absolute_error(y_test_ar, y_pred_test_ar)

print(f"AR Model Train MAE: {train_mae_ar:.2f}")
print(f"AR Model Test MAE: {test_mae_ar:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(filtered_data['YEAR'], filtered_data['ESTIMATE'], label='Actual Data', marker='o')
plt.plot(filtered_data['YEAR'][lag:split_index+lag], y_pred_train_ar, label='Train Predictions (AR)', linestyle='--')
plt.plot(filtered_data['YEAR'][split_index+lag:], y_pred_test_ar, label='Test Predictions (AR)', linestyle='--')
plt.plot(future_years.flatten(), future_forecasts_ar, label='Forecasts (AR)', marker='x')
plt.xlabel('Year')
plt.ylabel('Deaths per 100,000 Population')
plt.title('AR Model: Overdose Death Predictions for Male: 35-44 years')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
