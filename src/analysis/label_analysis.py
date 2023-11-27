from utils.CommonVariables import (
    COMMON_HUGGINGFACE_ACCESS_TOKEN, 
    COMMON_HUGGINGFACE_WRITE_TOKEN, 
    COMMON_DSTL_DOCUMENTS, 
    COMMON_BERT_OUTPUT_DIR,
    COMMON_BERT_LABELS, 
    COMMON_DSTL_ENTITIES,
    COMMON_BERT_MODEL_NAME,
    COMMON_BERT_LABEL2ID,
    COMMON_BERT_ID2LABEL,
    COMMON_BERT_REDUCE_LABELS,
    COMMON_BERT_LEARNING_RATE,
    COMMON_BERT_TRAIN_BATCH_SIZE,
    COMMON_BERT_EVAL_BATCH_SIZE,
    COMMON_BERT_EPOCHS,
    COMMON_BERT_WEIGHT_DECAY,
    COMMON_BERT_LOGGING_DIR,
    COMMON_BERT_LOGGING_STEPS
    )
from utils.BERTDataLoader import map_entities, map_entities_seperate
from utils.CommonDataLoader import construct_global_docMap


## BERT4 
## COMMON_BERT_SELF_LABELED_DATA = "../data/selv-labeled-data/B04/selfLabeledDataFilteredB04.conll"
## COMMON_BERT_REDUCE_LABELS = False


entities = construct_global_docMap(COMMON_DSTL_ENTITIES)

df_word_dstl, df_word_self = map_entities_seperate(entities,COMMON_DSTL_DOCUMENTS,COMMON_BERT_REDUCE_LABELS)

map_dstl = {}
sum_dstl = 0

for index, row in df_word_dstl.iterrows():
    sum_dstl += 1
    if len(row['ner_tags']) > 2:
        tag = row['ner_tags'][2:]
    else:
        tag = row['ner_tags']

    if tag in map_dstl.keys(): 
        map_dstl[tag] += 1
    else:
        map_dstl[tag] = 1

map_dstl['sum'] = sum_dstl

for key, value in map_dstl.items():
    map_dstl[key] = (value, value/sum_dstl * 100)

map_self = {}
sum_self = 0

for index, row in df_word_self.iterrows():
    sum_self += 1
    if len(row['ner_tags']) > 2:
        tag = row['ner_tags'][2:]
    else:
        tag = row['ner_tags']

    if tag in map_self.keys(): 
        map_self[tag] += 1
    else:
        map_self[tag] = 1

map_self['sum'] = sum_self

for key, value in map_self.items():
    map_self[key] = (value, value/sum_self * 100)


print('dstl', map_dstl)
print('self', map_self)
        
