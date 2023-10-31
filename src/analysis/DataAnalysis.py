import json
import itertools
from collections import Counter 
from LoadTestData import map_entities
from ner_training import construct_global_docMap

entities_file_ptah = "../data/re3d-master/*/entities_cleaned_sorted_and_filtered.json"
documens_file_path = "../data/re3d-master/*/documents.json"

def analyse_conll_file(filename):
    with open(filename, 'r', encoding="utf8") as f:
        lines = f.readlines()
        split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]
        entities = [[x.split('\t')[1][:-1] for x in y] for y in split_list]
    return entities

def count_labe_occourences(entities_list):
    combined_entites_list = []
    for entities in entities_list:
        for entity in entities:
            combined_entites_list.append(entity)
    print("Length")
    print(len(combined_entites_list))
    print("----------------------")
    print("Amount of Labels vs O-tag")
    print("Totoal amount of labels: ", len(combined_entites_list) - Counter(combined_entites_list)["O"])
    print("Total amount of O-tags: ", Counter(combined_entites_list)["O"])
    print("----------------------")
    print("Ratio of O tag")
    print(Counter(combined_entites_list)["O"] / len(combined_entites_list) * 100)
    print("----------------------")
    print("Ratio of amount of labels")
    print((len(Counter(combined_entites_list).keys()) - 1))
    print((len(Counter(combined_entites_list).keys()) - 1) / (len(combined_entites_list)) * 100)

if __name__ == '__main__':
    entities_dict = construct_global_docMap(entities_file_ptah)
    _, _, entities_json = map_entities(entities_dict, documens_file_path, False)
    entites_conll = analyse_conll_file("../data/selfLabeled.conll")
    entities_military_data = entities_json + entites_conll
    count_labe_occourences(entities_military_data)
    print("*****************************")
    _, _, entities_json = map_entities(entities_dict, documens_file_path, True)
    entites_conll = analyse_conll_file("../data/selfLabeledWithReducedLabels.conll")
    entities_military_data = entities_json + entites_conll
    count_labe_occourences(entities_military_data)
    print("*****************************")
    count_labe_occourences(analyse_conll_file("../data/train.conll"))