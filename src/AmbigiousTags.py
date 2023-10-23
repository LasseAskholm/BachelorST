import itertools
import json

# Run from /data/

def create_map():
    tag_map = {}
    duplicate_map = {}
    o_tags_set = O_tags()
    o_tag_duplicates = []

    path = "../data/selv-labeled-data/selfLabeledDataJSONFiltered.json"
    file = open(path)
    obj = json.load(file)
    for i in range (len(obj)):

        json_obj = obj[i]

        labels_present = True

        if json_obj.get("label") == None:
            labels_present = False

        if labels_present:
            for label in json_obj['label']:
                if label['text'] in tag_map.keys():
                    # label['labels'] length > 1?
                    if label['labels'][0] not in tag_map[label['text']]:
                        tag_map[label['text']].append(label['labels'][0])
                        
                        if label['text'] not in duplicate_map.keys():
                            duplicate_map[label['text']] = []
                        # else:
                        #     duplicate_map[label['text']].append(json_obj['id'])
                        # print(i, tag_map[label['text']])
                else:
                    tag_map[label['text']] = [label['labels'][0]]
                    if label['text'] in o_tags_set:
                        o_tag_duplicates.append(label['text'])
        
        
        for i in range (len(obj)):

            json_obj = obj[i]

            labels_present = True

            if json_obj.get("label") == None:
                labels_present = False

            if labels_present:
                for label in json_obj['label']:
                    if label['text'] in duplicate_map.keys():
                        if json_obj['id'] not in duplicate_map[label['text']]:
                            duplicate_map[label['text']].append(label['labels'][0])
                            duplicate_map[label['text']].append(json_obj['id'])


    with open('OtagAndLabel.json', 'w') as fp:
        json.dump(o_tag_duplicates, fp, indent=4)

    with open('multipleTags.json', 'w') as fp:
        json.dump(duplicate_map, fp, indent=4)

    return duplicate_map


def O_tags():

    O_tags_set = set()
    filename = "../data/selv-labeled-data/selfLabeledDataFiltered.conll"
    with open(filename, 'r', encoding="utf8") as f:
        lines = f.readlines()
        
        split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]

        entities = [[x.split('\t')[1][:-1] for x in y] for y in split_list]
        
        for y in split_list:
            for x in y:
                text, label = x.split('\t')
                label = label.strip('\n')
                if label == "O":
                    O_tags_set.add(text)
                
    return O_tags_set


create_map()
# O_tags()