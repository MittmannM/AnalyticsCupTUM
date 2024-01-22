import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score

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
# merge whole df with own generated RecipeMatchesDiet

# drop na diet column
df = df.dropna(subset=['Diet'])
# Rename AuthorId column
df.rename(columns= {
    "AuthorId" : "CustomerId",
    "Time": "MaxTime"
}, inplace=True)

df["Like"] = df["Like"].astype("boolean")
# Change types into category and mapping values
df["Diet"] = df["Diet"].astype("category")

df["RecipeCategory"] = df["RecipeCategory"].astype("category")

mapping_cal = {1: 'Yes', 0.0: 'No'}
df['HighCalories'] = df['HighCalories'].map(mapping_cal).astype('category')

mapping_protein = {'Yes': 'Yes', 'Indifferent': 'Indifferent', 'No': 'No' }
df['HighProtein'] = df['HighProtein'].map(mapping_protein).astype('category')

mapping_cal = {1: 'Yes', 0.0: 'No'}
df['LowFat'] = df['LowFat'].map(mapping_cal).astype('category')

mapping_sugar = {'1': 'Yes', 'Indifferent': 'Indifferent', '0': 'No' }
df['LowSugar'] = df['LowSugar'].map(mapping_sugar).astype('category')

mapping_cal = {1: 'Yes', 0.0: 'No'}
df['HighFiber'] = df['HighFiber'].map(mapping_cal).astype('category')

# Remove NA rows and Rating column
df = df.drop("Rating", axis=1)


# One hot encoding for categorical variables
df = pd.get_dummies(df, columns=['Diet','RecipeCategory', 'HighCalories', 'LowFat', 'HighFiber', 'HighProtein', 'LowSugar'], drop_first=True)

df.rename(columns={
    'HighCalories_Yes': 'want_HighCalories',
    'LowFat_Yes':'want_LowFat',
    'HighFiber_Yes':'want_HighFiber',
    'HighProtein_Yes':'want_HighProtein',
}, inplace=True)

df["RecipeIngredientParts"] = df["RecipeIngredientParts"].str.replace(")", '')
df["RecipeIngredientParts"] = df["RecipeIngredientParts"].str.replace("(", '')
df["RecipeIngredientParts"] = df["RecipeIngredientParts"].str.replace("\"", '')
df["RecipeIngredientParts"] = df["RecipeIngredientParts"].str.replace("\\", '')
df['RecipeIngredientParts'] = df['RecipeIngredientParts'].str.replace('^c', '', regex=True)

def check_keywords(ingredients):
    has_animal_product = any(any(keyword in ingredient.lower() for keyword in ["meat", "chicken", "lamb", "beef", "pork", "bacon", "fish", "sausage", "turkey", "milk", "butter", "egg", "cheese", "breast", "gelatin", "honey", "tuna", "steak", "salmon"]) for ingredient in ingredients)
    has_fish_or_meat = any(any(keyword in ingredient.lower() for keyword in ["meat", "chicken", "lamb", "beef", "pork", "bacon", "fish", "sausage", "turkey", "tuna", "steak", "salmon"]) for ingredient in ingredients)
    return has_animal_product, has_fish_or_meat

df[['has_animal_product', 'has_fish_meat']] = df['RecipeIngredientParts'].str.split(',').apply(check_keywords).apply(pd.Series)

df['for_Vegan'] = ~df['has_animal_product'] & ~df['has_fish_meat']
df['for_Vegetarian'] = (df['has_animal_product'] & ~df['has_fish_meat']) | (~df['has_animal_product'] & ~df['has_fish_meat'])
df['Correct_Diet'] = (~df['Diet_Vegetarian'] & ~df['Diet_Vegan']) | (df['Diet_Vegan'] & df['for_Vegan']) | (df['Diet_Vegetarian']  & df['for_Vegetarian'] )
df["DifferenceRequestedAndTimeNeeded"] = df["MaxTime"] - (df["CookTime"] + df["PrepTime"])

# Split data into train and test set
train_set = df[df["TestSetId"].isna()]
test_set = df[df["TestSetId"].notnull()]

train_set = train_set[train_set["Calories"] < 300000]

train_set.dropna(subset=["Like"], inplace=True)
train_set = train_set.drop("TestSetId", axis=1)

# needs to be done after outlier removal
recipesServings_mean = train_set['RecipeServings'].mean()
#fill na rows with the mean
train_set.loc[:, 'RecipeServings'] = train_set['RecipeServings'].fillna(recipesServings_mean)
test_set.loc[:, 'RecipeServings'] = test_set['RecipeServings'].fillna(recipesServings_mean)

# Variables that are good according to xgboost:

# variables_to_drop = ['CustomerId', 'RecipeId', 'Like', 'Name', 'RecipeIngredientQuantities', 'RecipeIngredientParts', 'RecipeYield', "Calories", "SaturatedFatContent", "SugarContent", "CookTime", "PrepTime", "Diet_Vegan", "Diet_Vegetarian", "RecipeCategory_Bread", "RecipeCategory_Other", "RecipeCategory_Breakfast", "RecipeCategory_Lunch", "RecipeCategory_Soup", "RecipeCategory_One dish meal", "LowSugar_No", "CholesterolContent", "SodiumContent", "FiberContent", "RecipeServings", "Time", "HighCalories", "HighFiber"]

variables_to_drop = ['CustomerId', 'RecipeId', 'Like', 'Name', 'RecipeIngredientQuantities', 'RecipeIngredientParts', 'RecipeYield','MaxTime', 'for_Vegetarian', 'for_Vegan', 'has_fish_meat', 'has_animal_product']
X = train_set.drop(variables_to_drop, axis=1)
y = train_set['Like']
test_set = test_set.drop(variables_to_drop, axis=1)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y,
                     test_size=0.3,
                     shuffle=True,
                     random_state=seed)

train_model = GradientBoostingClassifier(n_estimators=400, learning_rate=0.1, max_depth=7, random_state=seed)
train_model.fit(X_train,y_train)

test_set["prediction"] = train_model.predict(test_set.drop("TestSetId", axis=1))
test_output = pd.DataFrame(columns=["TestSetId", "prediction"])
test_output["TestSetId"] = test_set["TestSetId"].astype(int)
test_output["prediction"] = test_set["prediction"].astype(int)
test_output.to_csv("predictions_LetsSeePaulAllens'BAC_1.csv", index=False)

