'''
This file is to be used for common variables such as file-paths.

This file is structured such that the all required paths are given their own vairable name, 
but they should NOT be accessed. Rahter, the variables with the 'CURRENT_<MODEL_NAME>' naming-convension should be changed, 
so that they coorospond to a given expirement. 

If new expirements are added, a new section must be created defining all the variables for that given expirement. 
'''

'''
COMMON VARIABLES
'''
COMMON_HUGGINGFACE_ACCESS_TOKEN = "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"
COMMON_HUGGINGFACE_WRITE_TOKEN = "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"
COMMON_SKIPPED_LABELES = ["DocumentReference", "Nationality", "Quantity", "CommsIdentifier", "Coordinate", "Frequency"]
COMMON_DSTL_ENTITIES = "../../data/re3d-master/*/entities_cleaned_sorted_and_filtered.json"
COMMON_DSTL_DOCUMENTS = "../../data/re3d-master/*/documents.json"

'''
COMMON VARIABLES FOR BERT
'''
CONMON_BERT_SELF_LABELED_DATA = "../../data/selv-labeled-data/selfLabeledDataFiltered.conll"
COMMON_BERT_MODEL_NAME = "distilbert-base-multilingual-cased"
COMMON_BERT_LABELS = "../../resources/labelsReduced.txt"
COMMON_BERT_OUTPUT_DIR = "../../BERT"
COMMON_BERT_LEARNING_RATE = 4e-5
COMMON_BERT_TRAIN_BATCH_SIZE = 16
COMMON_BERT_EVAL_BATCH_SIZE = 16
COMMON_BERT_EPOCHS = 5
COMMON_BERT_WEIGHT_DECAY = 1e-5
COMMON_BERT_LOGGING_DIR = COMMON_BERT_OUTPUT_DIR + "/logging"
COMMON_BERT_LOGGING_STEPS = 10
COMMON_BERT_LABEL2ID = {"O": 0, 
            "B-Organisation": 1, 
            "I-Organisation": 2, 
            "B-Person": 3,
            "I-Person": 4, 
            "B-Location": 5,
            "I-Location": 6,
            "B-Money": 7,
            "I-Money": 8,
            "B-Temporal": 9,
            "I-Temporal": 10,
            "B-Weapon": 11,
            "I-Weapon": 12,
            "B-MilitaryPlatform": 13,
            "I-MilitaryPlatform": 14}
COMMON_BERT_ID2LABEL = {0 : "O", 
            1 : "B-Organisation", 
            2 : "I-Organisation", 
            3 : "B-Person",
            4 : "I-Person", 
            5 : "B-Location",
            6 : "I-Location",
            7 : "B-Money",
            8 : "I-Money",
            9 : "B-Temporal",
            10 : "I-Temporal",
            11 : "B-Weapon",
            12 : "I-Weapon",
            13 : "B-MilitaryPlatform",
            14 : "I-MilitaryPlatform"}


'''
COMMON VARIABLES FOR Llama2
'''
COMMON_LLAMA2_SELF_LABELED_DATA = "../../data/selv-labeled-data/selfLabeledDataJSONFiltered.json"
COMMON_LLAMA2_PROMT_INPUT = "Extract all entities in the following context along with their label from the entities at your disposal:"
COMMON_LLAMA2_PROMT_LABELES = "Organisation, Person, Location, Money, Temporal, Weapon, MilitaryPlatform"
COMMON_LLAMA2_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
COMMON_LLAMA2_OUTPUT_DIR = "../../Llama2"
COMMON_LLAMA2_LEARNING_RATE = 3e-4
COMMON_LLAMA2_TRAIN_BATCH_SIZE = 16
COMMON_LLAMA2_EVAL_BATCH_SIZE = 16
COMMON_LLAMA2_EPOCHS = 10
COMMON_LLAMA2_WEIGHT_DECAY = 1e-5
COMMON_LLAMA2_LOGGING_DIR = COMMON_LLAMA2_OUTPUT_DIR + "/logging"
COMMON_LLAMA2_LOGGING_STEPS = 10


'''
EXPIREMENT ONE BERT
'''


'''
EXPIREMENT TWO BERT
'''


'''
EXPIREMENT ONE Llama2
'''


'''
EXPIREMENT TWP Llama2
'''

