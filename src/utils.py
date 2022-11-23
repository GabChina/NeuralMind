import pandas as pd
import json
import numpy as np


def pandas2json(df, fname: str):
    """Convert pandas to json file

    Args:
        df (pd.DataFrame): Dataframe Object
        fname (str): file name
    """

    texts = []
    for i in range(len(df)):
        text_dict = {
            "text": df['text'].iloc[i],
            "tags": df['tags'].iloc[i]
        }
        texts.append(text_dict)

    with open(fname, 'w', encoding='utf8') as file:
        for text in texts:
            json.dump(text, file, ensure_ascii=False)
            file.write('\n')
