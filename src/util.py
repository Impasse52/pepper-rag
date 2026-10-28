import re

import numpy as np
import pandas as pd


# It has been noticed that there are some entries that consist only of a date in a particular format
def remove_dates(s: str) -> str:
    regex = r"\d{1,2}-(gen|feb|mar|apr|mag|giu|lug|ago|set|ott|nov|dic)-\d{4}"

    return re.sub(regex, "", s)


# Empty strings are replaced with the numpy NaN representation so that dropna can be used
def remove_empty_docs(df: str) -> str:
    df["text"] = df["text"].replace("", np.nan)
    df.dropna(subset=["text"], inplace=True)

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = df["text"].apply(remove_dates)
    df["text"] = df["text"].apply(lambda x: x.strip())

    df = remove_empty_docs(df)

    # Wpecify minimum document length
    # Most documents below threshold tend to be garbage documents
    df = df[df["text"].str.len() >= 200]

    df.reset_index(inplace=True, drop=True)

    return df
