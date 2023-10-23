import json


# Run from /data/

def create_map():
    tag_map = {}
    duplicate_map = {}
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

    with open('multipleTags.json', 'w') as fp:
        json.dump(duplicate_map, fp, indent=4)

    return duplicate_map

print(create_map())