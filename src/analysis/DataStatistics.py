from AmbigiousTags import *


def create_stats_map(path):
     tag_map = {}
     
     with open(path,'r',encoding="utf-8") as f:
        lines = f.readlines()
        split_list = [list(y) for x, y in itertools.groupby(lines, lambda z: z == '\n') if not x]
        for y in split_list:
            for x in y:
                text , label = x.split('\t')
                label = label.strip('\n')
                
     







def stats ():
  correctLabels = create_correct_labels_map()


  json_path ="../../data/selv-labeled-data/fixed_v0/selfLabeledDataJSONFiltered.json"
  conll_path = "../../data/selv-labeled-data/fixed_v0/selfLabeledDataFiltered.conll"
  actualLabelsV0 = create_stats_map(conll_path)
  calculatePureness(data_map=actualLabelsV0, correct_map=correctLabels)
def calculatePureness(data_map, correct_map):
   scores = {}
   print(len(data_map.keys()))

   for key in data_map.keys():
    print(key)
    print("\n")

def main():
  stats()



if __name__ == '__main__':
    main()