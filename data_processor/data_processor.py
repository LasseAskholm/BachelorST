import os
import json

# Find documents
path = "./data"
fname = []
for root,d_names,f_names in os.walk(path):
    for f in f_names:
        if "documents.json" in f:
            fname.append(os.path.join(root, f))

# Dictionary of DocumentID: Text
dict_with_documents = {}
for doc in fname:
     with open(doc, encoding="utf-8") as document:
            for line in document:
                content = json.loads(line)

                if content['_id'] not in dict_with_documents:
                     dict_with_documents[content['_id']] = ""

                dict_with_documents[content['_id']] = content['text']

# Dictionary of DocumentID: list((Entity, Label))
dict_with_entities = {}
for doc in fname:
    
    entity_path = doc.replace('documents.json', 'entities.json')

    with open(entity_path, encoding="utf-8") as entity:
            for line in entity:
                document = json.loads(line)

                if document['documentId'] not in dict_with_entities:
                     dict_with_entities[document['documentId']] = []

                dict_with_entities[document['documentId']].append((document['value'], document['type']))

# Write to file
with open("./data_processor/extracted.txt", "w", encoding="utf-8") as file:
    for docId, entity in dict_with_entities.items():
        file.write("DOCUMENT: " + docId + "\n")
        file.write("TEXT:\n" + dict_with_documents[docId] + "\n")
        
        entities = ""
        for entry in entity:
            if len(entry) > 1:
                entities += entry[0] + ": " + entry[1] + ", "
        file.write("ENTITIES:\n" + entities + "\n\n")        