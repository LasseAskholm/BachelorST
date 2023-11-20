import itertools
import json
import pandas as pd
import numpy as np


def get_truth(path):
  words = []
  labels = []
  conll_lines = open_conll(path)
  dumb_things = ["“",".",","," ","","”","(",")",":"]
  for y in conll_lines:
    for entry in y:
      text,label = entry.split('\t')
      label = label.strip('\n')
      label = label.strip("B-")
      label = label.strip("I-")
      if(text in dumb_things):
        continue
      words.append(text)
      labels.append(label)
      
  return pd.DataFrame({"words": words, "ner_tags": labels})   

  
  
  
def create_correct_labels_map(path):
    tag_map = {}
    split_list = open_conll(path)
    for y in split_list:
        temp = ""
        for x in y:
          text , label = x.split('\t')
          label = label.strip('\n')
          if "B-" in label:
            temp += text + " "
            tempLabel = label[2:]
          elif "I-" in label:
            temp += text + " "
          else:
            if temp != "":
              temp = temp[:-1]
              if temp not in tag_map.keys():
                tag_map[temp] = tempLabel
                temp = ""
                continue
              tag_map[temp] = tempLabel
              temp = ""
            else:
              if text not in tag_map.keys():
                if label != "O":
                  label = label[2:]
                tag_map[text] = label
              continue
    return tag_map

def create_stats_map(path):
  tag_map = {}
  split_list = open_conll(path)
  for y in split_list:
      temp = ""
      for x in y:
          text , label = x.split('\t')
          label = label.strip('\n')
          if "B-" in label:
            temp
            temp += text + " "
            tempLabel = label[2:]
          elif "I-" in label:
            temp += text + " "
          else:
            if temp != "":
              temp = temp[:-1]
              if temp not in tag_map.keys():
                tag_map[temp] = []
                tag_map[temp].append(tempLabel)
                temp = ""
                continue
              tag_map[temp].append(tempLabel)
              temp = ""
              
            else:
              if text not in tag_map.keys():
                tag_map[text] = []
                tag_map[text].append(label)
                continue
              tag_map[text].append(label)
  return tag_map                  

def most_frequent(List, word):
    unique, counts = np.unique(List, return_counts=True)
    if word == "military":
      print(word)
      print(unique)
      print(counts)
      print(List)
    index = np.argmax(counts)
    return unique[index] 

def stats ():
  correctLabels = create_correct_labels_map()
  conll_path = "../../data/selv-labeled-data/fixed_v5/v5.conll"
  actualLabelsV0 = create_stats_map(conll_path)
  scores = calculatePureness(data_map=actualLabelsV0, correct_map=correctLabels)
  df_scores = pd.DataFrame.from_dict(scores, orient='index')
  df_scores.to_csv('correctnessv5_3.csv', encoding='utf-8')
  
def calculatePureness(data_map, correct_map):
  scores = {}
  for word in data_map.keys():
    correctCount = 0
    incorrectCount = 0
    if word == "-DOCSTART-":
      continue
    correct_label = most_frequent(correct_map[word], word)
    
    for label in data_map[word]:
      if label == correct_label:
        correctCount += 1
        continue
      incorrectCount += 1 
    total_num_labels = len(data_map[word])
    correctPercent = (correctCount/total_num_labels)*100
    incorrectPercent = (incorrectCount/total_num_labels)*100
    scores[word] = {}
    scores[word]['Correct'] = correctPercent
    scores[word]['Incorrect'] = incorrectPercent
    scores[word]['Count'] = total_num_labels
    scores[word]['Label'] = correct_label
    
  return scores


def open_conll(path):
      with open(path,'r',encoding="utf-8") as f:
        lines = f.readlines()
        split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]
      return split_list



def main():
  stats()

if __name__ == '__main__':
    main()