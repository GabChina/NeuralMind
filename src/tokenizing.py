import numpy as np

def get_ent_label(entity_name: str) -> int:
    label_n = 0
    if entity_name=='CABECALHO':
        label_n=1
    elif entity_name=='SUBCABECALHO':
        label_n=3
    else:
        label_n=5
    return label_n

def create_label_vector(doc, input_ids, tokenizer):
    vetor=np.zeros(512)
    for ent_dict in doc['entities']:
        ent_label = get_ent_label(ent_dict['label'])
        entidade = doc['text'][ent_dict['start'] : ent_dict['end']]
        tokenized_entity = tokenizer(entidade, is_split_into_words=False)

        for token_idx, input_id in enumerate(input_ids):
            entity_ids = tokenized_entity['input_ids']
            if entity_ids[1] == input_id:
                if entity_ids[1:-1] == input_ids[token_idx : token_idx+(len(entity_ids)-2)]:
                    vetor[token_idx] = ent_label
                    vetor[token_idx+1:token_idx+(len(entity_ids)-2)] = ent_label+1
                    break

    for idx, id in enumerate(input_ids):
        if id == 101 or id ==102:
            vetor[idx] = -100

    return vetor.tolist()

def tokenize_dataset(dataset, tokenizer, stride=0):
    tokenized_dataset = []
    for doc in dataset:
        tokenized_text = tokenizer(doc['text'], padding='max_length', truncation=True,
                                    stride = stride,
                                    max_length=512, is_split_into_words=False,
                                    return_overflowing_tokens=True,)

        for idx, _ in enumerate(tokenized_text['overflow_to_sample_mapping']):
            new_doc = {
                'input_ids': tokenized_text.input_ids[idx],
                'attention_mask': tokenized_text.attention_mask[idx],
                'labels': create_label_vector(doc, tokenized_text.input_ids[idx], tokenizer),
            }
            tokenized_dataset.append(new_doc)

    return tokenized_dataset
