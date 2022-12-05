def balance_datasets(d1: list[dict], d2: list[dict], upper_limit=0.75,
                    balancing_range=0.2, names_list=None):
    """Balance NM NER dataset
    """
    # entities in each dataset
    entities_d1, entities_d2 = (__count_entities(d1, names_list),
                                __count_entities(d2, names_list))

    __realizar_correcao(d1, d2, entities_d1, entities_d2,
                        upper_limit=upper_limit,
                        balancing_range=balancing_range)

    __remove_null(d1); __remove_null(d2)


def __count_entities(dataset, names_list=None):
    """Returns a entities dict in the format:
    {
        'names': list[str]
        'ent_count': {'name': count (int)},     # dataset-wise
        'doc_count': [{'name': count (int)}]    # element-wise
        'pos': {'name': list[int]}
    }
    """
    names, ent_count, doc_count, pos = [], {}, [], {}
    if  names_list:
        for name in names_list:
            names.append(name)
            ent_count[name] = 0
            pos[name] = []

    for idx, doc in enumerate(dataset):
        if doc is None: continue
        doc_ent_count = {k: 0 for k in names}
        for entity in doc['entities']:
            ent_name = entity['label']
            if ent_name not in names:
                names.append(ent_name)
                ent_count[ent_name] = 0
                pos[ent_name] = []
                doc_ent_count[ent_name] = 0
                for doc in doc_count: doc.update({ent_name: 0})

            ent_count[ent_name] += 1
            pos[ent_name].append(idx)
            doc_ent_count[ent_name] += 1

        doc_count.append(doc_ent_count)

    return {'names': names, 'ent_count': ent_count,
            'doc_count': doc_count, 'pos': pos}


def __transfer_entity(destination, source, idx):
    destination.append(source[idx])
    source[idx] = None


def __balance_entity(destination, source, entities_dest, entities_src,
                    qtd, entity):
    """Transfere 'qtd' documentos que contém uma entidade
    do dataset de origem (source) para o dataset de destino (destination).
    """

    qtd = abs(qtd)
    while qtd > 0:
        for idx, doc in enumerate(entities_src['doc_count']):
            if doc[entity]:
                qtd -= doc[entity]
                __transfer_entity(destination, source, idx)

                for entity_name in doc.keys():
                    entities_src['ent_count'][entity_name] -= doc[entity_name]
                    entities_dest['ent_count'][entity_name] += doc[entity_name]
                    doc[entity_name] = 0
                break


def __realizar_correcao(d1, d2, entities_d1, entities_d2, upper_limit=0.75,
                        balancing_range=0.10):
    for entity in entities_d1['names']:
        e1, e2 = entities_d1['ent_count'][entity], entities_d2['ent_count'][entity]
        percent = e1/(e1+e2)
        unit_percent = 1/(e1+e2)

        # destination = d2, source = d1
        if percent > upper_limit:
            qtd = (percent - upper_limit + balancing_range/2) / unit_percent
            __balance_entity(d2, d1, entities_d2, entities_d1, round(qtd), entity)

        # destination = d1, source = d2
        if percent < upper_limit - balancing_range:
            qtd = (upper_limit - percent - balancing_range/2) / unit_percent
            __balance_entity(d1, d2, entities_d1, entities_d2, round(qtd), entity)


def __remove_null(dataset):
    for doc in reversed(dataset):
        if doc is None:
            dataset.remove(doc)
