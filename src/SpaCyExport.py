import spacy
import pandas as pd
import json
from itertools import groupby

# Download spaCy models:
models = {
    'en_core_web_sm': spacy.load("en_core_web_sm"),
    'en_core_web_lg': spacy.load("en_core_web_lg")
}

# This function converts spaCy docs to the list of named entity spans in Label Studio compatible JSON format:
def doc_to_spans(doc):
    tokens = [(tok.text, tok.idx, tok.ent_type_) for tok in doc]
    results = []
    entities = set()
    for entity, group in groupby(tokens, key=lambda t: t[-1]):
        if not entity:
            continue
        group = list(group)
        _, start, _ = group[0]
        word, last, _ = group[-1]
        text = ' '.join(item[0] for item in group)
        end = last + len(word)
        results.append({
            'from_name': 'label',
            'to_name': 'text',
            'type': 'labels',
            'value': {
                'start': start,
                'end': end,
                'text': text,
                'labels': [entity]
            }
        })
        entities.add(entity)

    return results, entities

# Now load the dataset and include only lines containing "Easter ":
df = pd.read_csv('lines_clean.csv')
df = df[df['line_text'].str.contains("Easter ", na=False)]
print(df.head())
texts = df['line_text']

# Prepare Label Studio tasks in import JSON format with the model predictions:
entities = set()
tasks = []
for text in texts:
    predictions = []
    for model_name, nlp in models.items():
        doc = nlp(text)
        spans, ents = doc_to_spans(doc)
        entities |= ents
        predictions.append({'model_version': model_name, 'result': spans})
    tasks.append({
        'data': {'text': text},
        'predictions': predictions
    })

# Save Label Studio tasks.json
print(f'Save {len(tasks)} tasks to "tasks.json"')
with open('tasks.json', mode='w') as f:
    json.dump(tasks, f, indent=2)
    
# Save class labels as a txt file
print('Named entities are saved to "named_entities.txt"')
with open('named_entities.txt', mode='w') as f:
    f.write('\n'.join(sorted(entities)))