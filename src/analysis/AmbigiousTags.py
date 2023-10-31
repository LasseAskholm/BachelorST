import itertools
import json
import pandas as pd

# Run from /data/

def create_map():
    tag_map = {}
    duplicate_map = {}
    o_tags_list = O_tags()
    o_tag_duplicates = {}

    path = "../data/selv-labeled-data/fixed_v5/v5.json"
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
                    for tup in o_tags_list:
                        if label['text'] == tup[0]:
                            if label['text'] in o_tag_duplicates.keys():
                                o_tag_duplicates[label['text']].append(tup[1])
                            else:
                                o_tag_duplicates[label['text']] = label['labels']
                                o_tag_duplicates[label['text']].append(tup[1])
        
        
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


    # with open('OtagAndLabelTuple.json', 'w') as fp:
    #     json.dump(o_tag_duplicates, fp, indent=4)

    # with open('multipleTags.json', 'w') as fp:
    #     json.dump(duplicate_map, fp, indent=4)

    df_otags = pd.DataFrame.from_dict(o_tag_duplicates, orient='index')
    df_multitags = pd.DataFrame.from_dict(duplicate_map, orient='index')

    df_otags.to_csv('OtagAndLabel_fix5.csv', encoding='utf-8')
    df_multitags.to_csv('multipleTags_fix5.csv', encoding='utf-8')


    return duplicate_map


def O_tags():

    O_tags_list = []
    filename = "../data/selv-labeled-data/fixed_v5/v5.conll"
    with open(filename, 'r', encoding="utf8") as f:
        lines = f.readlines()
        counter = -1
        split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]
        for y in split_list:
            counter += 1
            for x in y:
                counter += 1
                text, label = x.split('\t')
                label = label.strip('\n')
                if label == "O":
                    O_tags_list.append((text, counter))

    return O_tags_list


create_map()
#O_tags()