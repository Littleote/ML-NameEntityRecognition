import pandas as pd
from nltk.tokenize import word_tokenize as tokenize

NAME = "Common name"
POSITION = "Token number"

drug_names = pd.read_csv("drugbank vocabulary.csv", usecols=[NAME])
drug_names[NAME] = drug_names[NAME].apply(tokenize)
drug_names[POSITION] = drug_names[NAME].apply(lambda x: list(range(len(x))))
drug_names = drug_names.explode([NAME, POSITION])
drug_names = drug_names.drop_duplicates()
drug_names.to_csv("drugbank tokens.csv", index=False)
