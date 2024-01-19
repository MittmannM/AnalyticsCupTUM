import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

seed = 2024
np.random.seed(seed)

diet_csv = pd.read_csv("diet.csv").copy()
recipes_csv = pd.read_csv("recipes.csv").copy()
requests_csv = pd.read_csv("requests.csv").copy()
reviews_csv = pd.read_csv("reviews.csv").copy()

# merge diet + request
request_with_diet = pd.merge(diet_csv, requests_csv, how="inner", on="AuthorId")
# merge diet + request + recipe
request_with_diet_and_recipe = pd.merge(recipes_csv, request_with_diet, how="inner", on="RecipeId")
# merge diet + request + recipe + review
df = pd.merge(reviews_csv, request_with_diet_and_recipe, how="inner", on=["AuthorId", "RecipeId"])

# Rename AuthorId column
df.rename(columns= {"AuthorId" : "CustomerId"}, inplace=True)

# Change types of Diet & RecipeCategory into category
df["Diet"] = df["Diet"].astype("category")
df["RecipeCategory"] = df["RecipeCategory"].astype("category")

# Change types to boolean
df["HighCalories"] = df["HighCalories"].astype("bool")
df["LowFat"] = df["LowFat"].astype("bool")
df["HighFiber"] = df["HighFiber"].astype("bool")
df["Like"] = df["Like"].astype("boolean")

# Remove NA rows and Rating column
df = df.drop("Rating", axis=1)
df['Diet'].fillna('Vegetarian', inplace=True)

# Map indifferent values for HighProtein and LowSugar
mapping_protein = {'Yes': 'Yes', 'Indifferent': 'Indifferent', 'No': 'No', '0': 'No', '1': 'Yes'}
df["HighProtein"] = df["HighProtein"].map(mapping_protein).astype('category')

mapping_sugar = {'1': 'Yes', 'Indifferent': 'Indifferent', '0': 'No', }
df["LowSugar"] = df["LowSugar"].map(mapping_sugar).astype('category')

# One hot encoding for categorical variables
df = pd.get_dummies(df, columns=["Diet", "RecipeCategory", "HighProtein", "LowSugar"], drop_first=True)

# Split data into train and test set
train_set = df[df["TestSetId"].isna()]
test_set = df[df["TestSetId"].notnull()]

# diet_csv = diet_csv.dropna(axis="rows") and TestSetId
train_set.dropna(subset=["Like"], inplace=True)
train_set = train_set.drop("TestSetId", axis=1)

#Handle outliers
maxtime_val = train_set['Time'].max()
outliers = train_set['Time'] >= maxtime_val
median_without_outliers = train_set.loc[~outliers, 'Time'].median()
train_set.loc[outliers, 'Time'] = median_without_outliers

maxtime_val = train_set['PrepTime'].max()
outliers = train_set['PrepTime'] >= maxtime_val
median_without_outliers = train_set.loc[~outliers, 'PrepTime'].median()
train_set.loc[outliers, 'PrepTime'] = median_without_outliers
outliers = (train_set['Like'] == True) & (train_set['PrepTime'] > 3000000)
train_set.loc[outliers, 'PrepTime'] = median_without_outliers

outliers = (train_set['Like'] == True) & (train_set['Calories'] > 30000)
median_without_outliers = train_set.loc[~outliers, 'Calories'].median()
train_set.loc[outliers, 'Calories'] = median_without_outliers

outliers = train_set['FatContent'] > 25000
median_without_outliers = train_set.loc[~outliers, 'FatContent'].median()
train_set.loc[outliers, 'FatContent'] = median_without_outliers
outliers = (train_set['Like'] == True) & (train_set['FatContent'] > 2500)
train_set.loc[outliers, 'FatContent'] = median_without_outliers

outliers = train_set['SaturatedFatContent'] > 12000
median_without_outliers = train_set.loc[~outliers, 'SaturatedFatContent'].median()
train_set.loc[outliers, 'SaturatedFatContent'] = median_without_outliers

outliers = train_set['CholesterolContent'] > 35000
median_without_outliers = train_set.loc[~outliers, 'CholesterolContent'].median()
train_set.loc[outliers, 'CholesterolContent'] = median_without_outliers
outliers = (train_set['Like'] == True) & (train_set['CholesterolContent'] > 10000)
train_set.loc[outliers, 'CholesterolContent'] = median_without_outliers

outliers = (train_set['Like'] == True) & (train_set['CarbohydrateContent'] > 4000)
median_without_outliers = train_set.loc[~outliers, 'CarbohydrateContent'].median()
train_set.loc[outliers, 'CarbohydrateContent'] = median_without_outliers

outliers = (train_set['Like'] == True) & (train_set['FiberContent'] > 400)
median_without_outliers = train_set.loc[~outliers, 'FiberContent'].median()
train_set.loc[outliers, 'FiberContent'] = median_without_outliers

outliers = (train_set['Like'] == True) & (train_set['SugarContent'] > 4000)
median_without_outliers = train_set.loc[~outliers, 'SugarContent'].median()
train_set.loc[outliers, 'SugarContent'] = median_without_outliers

outliers = train_set['ProteinContent'] > 17500
median_without_outliers = train_set.loc[~outliers, 'ProteinContent'].median()
train_set.loc[outliers, 'ProteinContent'] = median_without_outliers
outliers = (train_set['Like'] == True) & (train_set['ProteinContent'] > 3000)
train_set.loc[outliers, 'ProteinContent'] = median_without_outliers

outliers = train_set['RecipeServings'] > 30000
median_without_outliers = train_set.loc[~outliers, 'RecipeServings'].median()
train_set.loc[outliers, 'RecipeServings'] = median_without_outliers
outliers = (train_set['Like'] == True) & (train_set['RecipeServings'] > 400)
train_set.loc[outliers, 'RecipeServings'] = median_without_outliers

# needs to be done after outlier removal
recipesServings_mean = train_set['RecipeServings'].mean()
#fill na rows with the mean
train_set['RecipeServings'].fillna(recipesServings_mean, inplace=True)
test_set['RecipeServings'].fillna(recipesServings_mean, inplace=True)

# Add additional varibale for difference in requested time and recipe time
train_set["DifferenceRequestedAndTimeNeeded"] = train_set["Time"] - (train_set["CookTime"] + train_set["PrepTime"])
test_set["DifferenceRequestedAndTimeNeeded"] = test_set["Time"] - (test_set["CookTime"] + test_set["PrepTime"])

# prepare X and y for the model
variables_to_drop = ['CustomerId', 'RecipeId', 'Like', 'Name', 'RecipeIngredientQuantities', 'RecipeIngredientParts',
                     'RecipeYield', 'RecipeServings']

X = train_set.drop(variables_to_drop, axis=1)
y = train_set['Like']

test_set = test_set.drop(variables_to_drop, axis=1)


scaler = MinMaxScaler()

scaled_df = scaler.fit_transform(X)

transform_scaler = StandardScaler()

scaled_df = transform_scaler.fit_transform(scaled_df)

X_train, X_test, y_train, y_test = \
    train_test_split(scaled_df, y,
                     test_size=0.3,
                     shuffle=True,
                     random_state=3)

# Gradient Boosting

train_model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.6, random_state=seed)

train_model.fit(X_train,y_train)

score = np.mean(cross_val_score(train_model, X_train, y_train, cv=4, scoring="balanced_accuracy"))
print("Cross validation score for balanced accuracy: " + str(score))

test_predictions = train_model.predict(X_test)
test_probabilities = train_model.predict_proba(X_test)

test_predictions_df = pd.DataFrame({'Like': y_test,
                                     'Predicted_Like': test_predictions,
                                     'Probability_Like=0': test_probabilities[:, 0],
                                     'Probability_Like=1': test_probabilities[:, 1]})
print(test_predictions_df)


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, test_predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Precision, accuracy, recall
print("Test-Precision:", precision_score(y_test, test_predictions))
print("Test-Accuracy:", accuracy_score(y_test, test_predictions))
print("Test-Recall:", recall_score(y_test, test_predictions))
print("Test Balanced Accuracy:", balanced_accuracy_score(y_test, test_predictions))

test_set["prediction"] = train_model.predict(test_set.drop("TestSetId", axis=1))
test_output = pd.DataFrame(columns=["TestSetId", "prediction"])
test_output["TestSetId"] = test_set["TestSetId"].astype(int)
test_output["prediction"] = test_set["prediction"].astype(int)
test_output.to_csv("predictions_LetsSeePaulAllens'BAC_1.csv", index=False)