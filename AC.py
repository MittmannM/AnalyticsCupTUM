import pandas as pd
import numpy as np

# set seed
seed = 2024
np.random.seed(seed)

# read in data
diet_csv = pd.read_csv("diet.csv")
recipes_csv = pd.read_csv("recipes.csv")
requests_csv = pd.read_csv("requests.csv")
reviews_csv = pd.read_csv("reviews.csv")

# data cleaning

# diet_csv
diet_csv["Diet"] = diet_csv["Diet"].astype("category")

# recipes_csv
# TODO CookTime, PrepTime - is it in Minutes/Seconds? RecipeIngredientParts & RecipeIngredientQuantities
# TODO What to do with Servings and Yield?
recipes_csv["RecipeCategory"] = recipes_csv["RecipeCategory"].astype("category")

# requests_csv
# TODO Time - Is it in Minutes/Seconds?
requests_csv["HighCalories"] = requests_csv["HighCalories"].astype("boolean")
requests_csv["LowFat"] = requests_csv["LowFat"].astype("boolean")
requests_csv["HighFiber"] = requests_csv["HighFiber"].astype("boolean")

requests_csv["HighProtein"] = requests_csv["HighProtein"].map({
    "Indifferent": 0,
    "0": 0,
    "1": 1,
    "Yes": 1
})
requests_csv["LowSugar"] = requests_csv["LowSugar"].map({
    "Indifferent": 0,
    "0": 0,
    "1": 1
})
requests_csv["HighProtein"] = requests_csv["HighProtein"].astype("boolean")
requests_csv["LowSugar"] = requests_csv["LowSugar"].astype("boolean")


# reviews_csv
reviews_csv.info()
# TODO wtf rating only has values 2/NA -> delete?
# TODO need to fix NA values for Like -> delete rows with NA?
# TODO why are there missing values for TestSetId?
#print(reviews_csv["Rating"].values.unique())

# TODO Join the tables
