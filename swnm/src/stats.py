def get_entities_percentage(entities_d1, entities_d2, print_results=True):
    percents = [e1/(e1+e2)
    for e1, e2 in zip(entities_d1['ent_count'].values(), entities_d2['ent_count'].values())
    ]

    text = ''
    for percent, entity in zip(percents, entities_d1['names']):
        text += f'{percent}\t{entity}\n'

    if print_results:
        print(text, end='')

    return text


def get_entities_count(entities_d1, print_results=True):
    text = ''
    for count, entity in zip(entities_d1['ent_count'].values(), entities_d1['names']):
        text += f'{count} \t{entity}\n'

    if print_results:
        print(text, end='')

    return text
