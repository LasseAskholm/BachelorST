import itertools
import json
import pandas as pd

# Run from /data/

def create_map(Json_path,conll_path, makeCSV):
    tag_map = {}
    duplicate_map = {}
    o_tags_list = O_tags(conll_path)
    o_tag_duplicates = {}

    path = Json_path
    file = open(path, encoding = 'utf-8')
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
    if makeCSV:
        df_otags = pd.DataFrame.from_dict(o_tag_duplicates, orient='index')
        df_multitags = pd.DataFrame.from_dict(duplicate_map, orient='index')

        df_otags.to_csv('OtagAndLabel_fix5.csv', encoding='utf-8')
        df_multitags.to_csv('multipleTags_fix5.csv', encoding='utf-8')


    return duplicate_map


def O_tags(path):

    O_tags_list = []
    filename = path
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

def create_correct_labels_map():
    tag_map = {}

    data_path ="../../data/selv-labeled-data/fixed_v5/v5.conll"
    with open(data_path,'r',encoding="utf-8") as f:
        lines = f.readlines()
        split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]
        for y in split_list:
            for x in y:
                text , label = x.split('\t')
                label = label.strip('\n')
                if text not in tag_map.keys():
                    tag_map[text]= label
                continue
    return tag_map


def count_tags():

    tags_map = {}
    filename = "../data/selv-labeled-data/fixed_v5/v5.conll"
    with open(filename, 'r', encoding="utf8") as f:
        lines = f.readlines()
        counter = -1
        split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]
        temp = ""
        tempLabel = ""
        for y in split_list:
            counter += 1
            for x in y:
                counter += 1
                text, label = x.split('\t')
                label = label.strip('\n')

                if label != "O":
                    label = label[2:]

                if text not in tags_map.keys():
                    tags_map[text] = {}

                if label in tags_map[text].keys():
                    tags_map[text][label] += 1
                else:
                    tags_map[text][label] = 1

                
                if "B-" in label:
                    temp += text + " "
                    tempLabel = label[2:]
                elif "I-" in label:
                    temp += text + " "
                else:

                    if temp != "":
                    temp = temp[:-1]
                    if temp not in tags_maps.keys():
                        tags_map[temp] = tempLabel
                        temp = ""
                        continue
                    tags_map[temp] = tempLabel
                    temp = ""

                    else:
                    if text not in tags_maps.keys():
                        if label != "O":
                        label = label[2:]
                        tags_maps[text] = label
                    continue


        # with open('tag_count.json', 'w') as fp:
        #     json.dump(tags_map, fp, indent=4)

        # df_counttags = pd.DataFrame.from_dict(tags_map, orient='index')
        # df_counttags.to_csv('count_tags.csv', encoding='utf-8')
        

    return tags_map

# create_map()
#O_tags()

print(count_tags())