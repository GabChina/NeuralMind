import json

from numpyencoder import NumpyEncoder


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


def json2dict(fname: str, mode='r', encoding='utf8'):
    """Loads data from a json file into a dict object
    """
    with open(fname, mode, encoding=encoding) as jfile:
        data = json.load(jfile)

    return data


def dict2json(data: list[dict], fname: str,
                sort_keys=False, indent=None):
    """Saves the data in a json file
    Args:
        data (list[dict]): data in NM format:
            {'text': str,
            'entities': list[{'start': int, 'end': int, 'label': str, 'value': str}],
            'anottation_status': str,
            'notes': str}
        fname (str): output file
    """

    with open(fname, 'w', encoding='utf8') as file:
        json.dump(data, file, ensure_ascii=False,
                    sort_keys=sort_keys, indent=indent,
                    cls=NumpyEncoder)
