import pandas as pd
import json 

def json_to_ner_dataset(json_data):
    # Initialize empty lists to store tokens and labels
    tokens = []
    labels = []

    for record in json_data:
        # Extract relevant fields from the JSON record
        begin = record["begin"]
        end = record["end"]
        entity_type = record["type"]
        value = record["value"]

        # Tokenize the value
        value_tokens = value.split()

        # Initialize a flag to keep track of token position
        token_position = 0

        for token in value_tokens:
            # Determine the label for each token based on its position
            if token_position == 0:
                labels.append(f"B-{entity_type}")  # Beginning of entity
            else:
                labels.append(f"I-{entity_type}")  # Inside of entity

            tokens.append(token)
            token_position += 1

    # Create a DataFrame from the tokens and labels
    df = pd.DataFrame({"Token": tokens, "Label": labels})

    return df

def parse_json(path):
    parsed_json=[]

    with open (path, 'r') as json_file:
        for line in json_file:
            parsed_record = json.loads(line)
            parsed_json.append(parsed_record)
    
    return parsed_json

# Example usage:
new_data = parse_json('C:/Git/Bachelores/BachelorST/data/re3d-master/US State Department/entities.json') 
df = json_to_ner_dataset(new_data)
print(df)
