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
COMMON_DSTL_ENTITIES = "../data/re3d-master/*/entities_cleaned_sorted.json"
COMMON_DSTL_DOCUMENTS = "../data/re3d-master/*/documents.json"
COMMON_RUN_WITH_DSTL = True

'''
COMMON VARIABLES FOR BERT
'''
COMMON_BERT_SELF_LABELED_DATA = "../data/selv-labeled-data/B04/selfLabeledDataFilteredB04.conll"
COMMON_BERT_MODEL_NAME = "distilbert-base-multilingual-cased"
COMMON_BERT_LABELS = "../../resources/labelsReduced.txt"
COMMON_BERT_RAW_LABELS= "../../resources/raw_labels_reduced.txt"
COMMON_BERT_OUTPUT_DIR = "../../BERT_B09"
COMMON_BERT_REDUCE_LABELS = True
COMMON_BERT_ENABLE_ADDITIONAL_DATA = True
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
COMMON_LLAMA2_SELF_LABELED_DATA = "../../data/selv-labeled-data/L09/selfLabeledDataJSONL09.json"
COMMON_LLAMA2_PROMT_INPUT = "Extract all entities in the following context along with the label associated that was defined above:"
COMMON_LLAMA2_PROMT_LABELES = "Organisation, Person, Location, Money, Temporal, Weapon, MilitaryPlatform"
COMMON_LLAMA2_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
COMMON_LLAMA2_OUTPUT_DIR = "../../L09"
COMMON_LLAMA2_SENTENCE_LABELS = [
        "Organisation",
        "Person",
        "Location",
        "Money",
        "Temporal",
        "Weapon",
        "MilitaryPlatform"
    ]
COMMON_LLAMA2_LEARNING_RATE = 3e-4
COMMON_LLAMA2_TRAIN_BATCH_SIZE = 16
COMMON_LLAMA2_EVAL_BATCH_SIZE = 16
COMMON_LLAMA2_EPOCHS = 4
COMMON_LLAMA2_WEIGHT_DECAY = 1e-5
COMMON_LLAMA2_LOGGING_DIR = COMMON_LLAMA2_OUTPUT_DIR + "/logging"
COMMON_LLAMA2_LOGGING_STEPS = 10
COMMON_LLAMA2_RUN_WITH_SENTENCE_SETTING = False
COMMON_LLAMA2_RUN_WITH_DOCUMENT_SETTING = False
COMMON_LLAMA2_REDUCED_LABELS = True
COMMON_LLAMA2_RUN_WITH_SINGLE_LABEL = True


'''
EXPIREMENT ONE BERT
'''
B01_BERT_SELF_LABELED_DATA = "../../data/selv-labeled-data/selfLabeledDataFiltered.conll"
B01_BERT_MODEL_NAME = "distilbert-base-multilingual-cased"
B01_BERT_LABELS = "../../resources/labels.txt"
B01_BERT_OUTPUT_DIR = "../../BERT_B01"
B01_BERT_REDUCE_LABELS = False
B01_BERT_ENABLE_ADDITIONAL_DATA = False
B01_RUN_WITH_DSTL = True
B01_BERT_LEARNING_RATE = 2e-5
B01_BERT_TRAIN_BATCH_SIZE = 16
B01_BERT_EVAL_BATCH_SIZE = 16
B01_BERT_EPOCHS = 30
B01_BERT_WEIGHT_DECAY = 1e-5
B01_BERT_LOGGING_DIR = B01_BERT_OUTPUT_DIR + "/logging"
B01_BERT_LOGGING_STEPS = 10
B01_BERT_LABEL2ID = {"O": 0, 
            "B-Organisation": 1, 
            "I-Organisation": 2, 
            "B-Nationality": 3, 
            "I-Nationality": 4, 
            "B-Person": 5,
            "I-Person": 6, 
            "B-DocumentReference": 7,
            "I-DocumentReference": 8,
            "B-Location": 9,
            "I-Location": 10,
            "B-Money": 11,
            "I-Money": 12,
            "B-Vehicle": 13,
            "I-Vehicle": 14,
            "B-Temporal": 15,
            "I-Temporal": 16,
            "B-Weapon": 17,
            "I-Weapon": 18,
            "B-Quantity": 19,
            "I-Quantity": 20,
            "B-CommsIdentifier": 21,
            "B-MilitaryPlatform": 22,
            "I-MilitaryPlatform": 23,
            "B-Coordinate": 24,
            "I-Coordinate": 25,
            "B-Frequency": 26,
            "I-Frequency": 27}
B01_BERT_ID2LABEL = {0 : "O", 
            1 : "B-Organisation", 
            2 : "I-Organisation", 
            3 : "B-Nationality", 
            4 : "I-Nationality", 
            5 : "B-Person",
            6 : "I-Person", 
            7 : "B-DocumentReference",
            8 : "I-DocumentReference",
            9 : "B-Location",
            10 : "I-Location",
            11 : "B-Money",
            12 : "I-Money",
            13 : "B-Vehicle",
            14 : "I-Vehicle",
            15 : "B-Temporal",
            16 : "I-Temporal",
            17 : "B-Weapon",
            18 : "I-Weapon",
            19 : "B-Quantity",
            20 : "I-Quantity",
            21 : "B-CommsIdentifier",
            22 : "B-MilitaryPlatform",
            23 : "I-MilitaryPlatform",
            24 : "B-Coordinate",
            25 : "I-Coordinate",
            26 : "B-Frequency",
            27 : "I-Frequency"}


'''
EXPIREMENT TWO BERT
'''
B02_BERT_SELF_LABELED_DATA = "../../data/selv-labeled-data/selfLabeledDataFiltered.conll"
B02_BERT_MODEL_NAME = "distilbert-base-multilingual-cased"
B02_BERT_LABELS = "../../resources/labels.txt"
B02_BERT_OUTPUT_DIR = "../../BERT_B02"
B02_BERT_REDUCE_LABELS = False
B02_BERT_ENABLE_ADDITIONAL_DATA = False
B02_RUN_WITH_DSTL = True
B02_BERT_LEARNING_RATE = 4e-5
B02_BERT_TRAIN_BATCH_SIZE = 16
B02_BERT_EVAL_BATCH_SIZE = 16
B02_BERT_EPOCHS = 5
B02_BERT_WEIGHT_DECAY = 1e-5
B02_BERT_LOGGING_DIR = B02_BERT_OUTPUT_DIR + "/logging"
B02_BERT_LOGGING_STEPS = 10
B02_BERT_LABEL2ID = {"O": 0, 
            "B-Organisation": 1, 
            "I-Organisation": 2, 
            "B-Nationality": 3, 
            "I-Nationality": 4, 
            "B-Person": 5,
            "I-Person": 6, 
            "B-DocumentReference": 7,
            "I-DocumentReference": 8,
            "B-Location": 9,
            "I-Location": 10,
            "B-Money": 11,
            "I-Money": 12,
            "B-Vehicle": 13,
            "I-Vehicle": 14,
            "B-Temporal": 15,
            "I-Temporal": 16,
            "B-Weapon": 17,
            "I-Weapon": 18,
            "B-Quantity": 19,
            "I-Quantity": 20,
            "B-CommsIdentifier": 21,
            "B-MilitaryPlatform": 22,
            "I-MilitaryPlatform": 23,
            "B-Coordinate": 24,
            "I-Coordinate": 25,
            "B-Frequency": 26,
            "I-Frequency": 27}
B02_BERT_ID2LABEL = {0 : "O", 
            1 : "B-Organisation", 
            2 : "I-Organisation", 
            3 : "B-Nationality", 
            4 : "I-Nationality", 
            5 : "B-Person",
            6 : "I-Person", 
            7 : "B-DocumentReference",
            8 : "I-DocumentReference",
            9 : "B-Location",
            10 : "I-Location",
            11 : "B-Money",
            12 : "I-Money",
            13 : "B-Vehicle",
            14 : "I-Vehicle",
            15 : "B-Temporal",
            16 : "I-Temporal",
            17 : "B-Weapon",
            18 : "I-Weapon",
            19 : "B-Quantity",
            20 : "I-Quantity",
            21 : "B-CommsIdentifier",
            22 : "B-MilitaryPlatform",
            23 : "I-MilitaryPlatform",
            24 : "B-Coordinate",
            25 : "I-Coordinate",
            26 : "B-Frequency",
            27 : "I-Frequency"}

'''
EXPIREMENT THREE BERT
'''
B03_BERT_SELF_LABELED_DATA = "../../data/selv-labeled-data/B03/selfLabeledDataB03.conll"
B03_BERT_MODEL_NAME = "distilbert-base-multilingual-cased"
B03_BERT_LABELS = "../../resources/labels.txt"
B03_BERT_OUTPUT_DIR = "../../BERT_B03"
B03_BERT_REDUCE_LABELS = False
B03_BERT_ENABLE_ADDITIONAL_DATA = True
B03_RUN_WITH_DSTL = True
B03_BERT_LEARNING_RATE = 4e-5
B03_BERT_TRAIN_BATCH_SIZE = 16
B03_BERT_EVAL_BATCH_SIZE = 16
B03_BERT_EPOCHS = 5
B03_BERT_WEIGHT_DECAY = 1e-5
B03_BERT_LOGGING_DIR = B03_BERT_OUTPUT_DIR + "/logging"
B03_BERT_LOGGING_STEPS = 10
B03_BERT_LABEL2ID = {"O": 0, 
            "B-Organisation": 1, 
            "I-Organisation": 2, 
            "B-Nationality": 3, 
            "I-Nationality": 4, 
            "B-Person": 5,
            "I-Person": 6, 
            "B-DocumentReference": 7,
            "I-DocumentReference": 8,
            "B-Location": 9,
            "I-Location": 10,
            "B-Money": 11,
            "I-Money": 12,
            "B-Vehicle": 13,
            "I-Vehicle": 14,
            "B-Temporal": 15,
            "I-Temporal": 16,
            "B-Weapon": 17,
            "I-Weapon": 18,
            "B-Quantity": 19,
            "I-Quantity": 20,
            "B-CommsIdentifier": 21,
            "B-MilitaryPlatform": 22,
            "I-MilitaryPlatform": 23,
            "B-Coordinate": 24,
            "I-Coordinate": 25,
            "B-Frequency": 26,
            "I-Frequency": 27}
B03_BERT_ID2LABEL = {0 : "O", 
            1 : "B-Organisation", 
            2 : "I-Organisation", 
            3 : "B-Nationality", 
            4 : "I-Nationality", 
            5 : "B-Person",
            6 : "I-Person", 
            7 : "B-DocumentReference",
            8 : "I-DocumentReference",
            9 : "B-Location",
            10 : "I-Location",
            11 : "B-Money",
            12 : "I-Money",
            13 : "B-Vehicle",
            14 : "I-Vehicle",
            15 : "B-Temporal",
            16 : "I-Temporal",
            17 : "B-Weapon",
            18 : "I-Weapon",
            19 : "B-Quantity",
            20 : "I-Quantity",
            21 : "B-CommsIdentifier",
            22 : "B-MilitaryPlatform",
            23 : "I-MilitaryPlatform",
            24 : "B-Coordinate",
            25 : "I-Coordinate",
            26 : "B-Frequency",
            27 : "I-Frequency"}

'''
EXPIREMENT FOUR BERT
'''
B04_BERT_SELF_LABELED_DATA = "../../data/selv-labeled-data/B04/selfLabeledDataFilteredB04.conll"
B04_BERT_MODEL_NAME = "distilbert-base-multilingual-cased"
B04_BERT_LABELS = "../../resources/labelsReduced.txt"
B04_BERT_OUTPUT_DIR = "../../BERT_B04"
B04_BERT_REDUCE_LABELS = True
B04_BERT_ENABLE_ADDITIONAL_DATA = True
B04_RUN_WITH_DSTL = True
B04_BERT_LEARNING_RATE = 4e-5
B04_BERT_TRAIN_BATCH_SIZE = 16
B04_BERT_EVAL_BATCH_SIZE = 16
B04_BERT_EPOCHS = 5
B04_BERT_WEIGHT_DECAY = 1e-5
B04_BERT_LOGGING_DIR = B04_BERT_OUTPUT_DIR + "/logging"
B04_BERT_LOGGING_STEPS = 10
B04_BERT_LABEL2ID = {"O": 0, 
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
B04_BERT_ID2LABEL = {0 : "O", 
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
EXPIREMENT FIVE BERT
'''
B05_BERT_SELF_LABELED_DATA = "../../data/selv-labeled-data/B05/selfLabeledDataFilteredB05.conll"
B05_BERT_MODEL_NAME = "distilbert-base-multilingual-cased"
B05_BERT_LABELS = "../../resources/labelsReduced.txt"
B05_BERT_OUTPUT_DIR = "../../BERT_B05"
B05_BERT_REDUCE_LABELS = True
B05_BERT_ENABLE_ADDITIONAL_DATA = True
B05_RUN_WITH_DSTL = True
B05_BERT_LEARNING_RATE = 4e-5
B05_BERT_TRAIN_BATCH_SIZE = 16
B05_BERT_EVAL_BATCH_SIZE = 16
B05_BERT_EPOCHS = 5
B05_BERT_WEIGHT_DECAY = 1e-5
B05_BERT_LOGGING_DIR = B05_BERT_OUTPUT_DIR + "/logging"
B05_BERT_LOGGING_STEPS = 10
B05_BERT_LABEL2ID = {"O": 0, 
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
B05_BERT_ID2LABEL = {0 : "O", 
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
EXPIREMENT SIX BERT
'''
B06_BERT_SELF_LABELED_DATA = "../../data/selv-labeled-data/B06/selfLabeledDataFilteredB06.conll"
B06_BERT_MODEL_NAME = "distilbert-base-multilingual-cased"
B06_BERT_LABELS = "../../resources/labelsReduced.txt"
B06_BERT_OUTPUT_DIR = "../../BERT_B06"
B06_BERT_REDUCE_LABELS = True
B06_BERT_ENABLE_ADDITIONAL_DATA = True
B05_RUN_WITH_DSTL = False
B06_BERT_LEARNING_RATE = 4e-5
B06_BERT_TRAIN_BATCH_SIZE = 16
B06_BERT_EVAL_BATCH_SIZE = 16
B06_BERT_EPOCHS = 5
B06_BERT_WEIGHT_DECAY = 1e-5
B06_BERT_LOGGING_DIR = B06_BERT_OUTPUT_DIR + "/logging"
B06_BERT_LOGGING_STEPS = 10
B06_BERT_LABEL2ID = {"O": 0, 
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
B06_BERT_ID2LABEL = {0 : "O", 
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
EXPIREMENT ONE Llama2 

N/A
'''


'''
EXPIREMENT TWO Llama2

TIME: around 5 min
'''
L02_LLAMA2_SELF_LABELED_DATA = "../../data/selv-labeled-data/L02/selfLabeledDataJSONL02.json"
L02_LLAMA2_PROMT_INPUT = "Extract all entities in the following context along with their label from the entities at your disposal:"
L02_LLAMA2_PROMT_LABELES = "Organisation, Nationality, DocumentReference, Person, Location, Money, Vehicle, Temporal, Weapon, Quantity, CommsIdentifier, MilitaryPlatform, Coordinate, Frequency"
L02_LLAMA2_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
L02_LLAMA2_OUTPUT_DIR = "../../L02"
L02_LLAMA2_SENTENCE_LABELS = [
        "Organisation",
        "Person",
        "Location",
        "Money",
        "Temporal",
        "Weapon",
        "MilitaryPlatform"
    ]
L02_LLAMA2_LEARNING_RATE = 3e-4
L02_LLAMA2_TRAIN_BATCH_SIZE = 16
L02_LLAMA2_EVAL_BATCH_SIZE = 16
L02_LLAMA2_EPOCHS = 4
L02_LLAMA2_WEIGHT_DECAY = 1e-5
L02_LLAMA2_LOGGING_DIR = L02_LLAMA2_OUTPUT_DIR + "/logging"
L02_LLAMA2_LOGGING_STEPS = 10
L02_LLAMA2_RUN_WITH_SENTENCE_SETTING = False
L02_LLAMA2_RUN_WITH_DOCUMENT_SETTING = True
L02_LLAMA2_REDUCED_LABELS = False
L02_LLAMA2_RUN_WITH_SINGLE_LABEL = False


'''
EXPIREMENT THREE Llama2

TIME: around 12 min
'''
L03_LLAMA2_SELF_LABELED_DATA = "../../data/selv-labeled-data/L04/selfLabeledDataJSONL04.json"
L03_LLAMA2_PROMT_INPUT = "Extract all entities in the following context along with their label from the entities at your disposal:"
L03_LLAMA2_PROMT_LABELES = "Organisation, Nationality, DocumentReference, Person, Location, Money, Vehicle, Temporal, Weapon, Quantity, CommsIdentifier, MilitaryPlatform, Coordinate, Frequency"
L03_LLAMA2_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
L03_LLAMA2_OUTPUT_DIR = "../../L03"
L03_LLAMA2_SENTENCE_LABELS = [
        "Organisation",
        "Person",
        "Location",
        "Money",
        "Temporal",
        "Weapon",
        "MilitaryPlatform"
    ]
L03_LLAMA2_LEARNING_RATE = 3e-4
L03_LLAMA2_TRAIN_BATCH_SIZE = 16
L03_LLAMA2_EVAL_BATCH_SIZE = 16
L03_LLAMA2_EPOCHS = 8
L03_LLAMA2_WEIGHT_DECAY = 1e-5
L03_LLAMA2_LOGGING_DIR = L03_LLAMA2_OUTPUT_DIR + "/logging"
L03_LLAMA2_LOGGING_STEPS = 10
L03_LLAMA2_RUN_WITH_SENTENCE_SETTING = False
L03_LLAMA2_RUN_WITH_DOCUMENT_SETTING = True
L03_LLAMA2_REDUCED_LABELS = False
L03_LLAMA2_RUN_WITH_SINGLE_LABEL = False



'''
EXPIREMENT FOUR Llama2

TIME: around 1,5 hours
'''
L04_LLAMA2_SELF_LABELED_DATA = "../../data/selv-labeled-data/L04/selfLabeledDataJSONL04.json"
L04_LLAMA2_PROMT_INPUT = "Extract all entities in the following context along with their label from the entities at your disposal:"
L04_LLAMA2_PROMT_LABELES = "Organisation, Nationality, DocumentReference, Person, Location, Money, Vehicle, Temporal, Weapon, Quantity, CommsIdentifier, MilitaryPlatform, Coordinate, Frequency"
L04_LLAMA2_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
L04_LLAMA2_OUTPUT_DIR = "../../L04"
L04_LLAMA2_SENTENCE_LABELS = [
        "Organisation",
        "Person",
        "Location",
        "Money",
        "Temporal",
        "Weapon",
        "MilitaryPlatform"
    ]
L04_LLAMA2_LEARNING_RATE = 3e-4
L04_LLAMA2_TRAIN_BATCH_SIZE = 16
L04_LLAMA2_EVAL_BATCH_SIZE = 16
L04_LLAMA2_EPOCHS = 4
L04_LLAMA2_WEIGHT_DECAY = 1e-5
L04_LLAMA2_LOGGING_DIR = L04_LLAMA2_OUTPUT_DIR + "/logging"
L04_LLAMA2_LOGGING_STEPS = 10
L04_LLAMA2_RUN_WITH_SENTENCE_SETTING = True
L04_LLAMA2_RUN_WITH_DOCUMENT_SETTING = False
L04_LLAMA2_REDUCED_LABELS = False
L04_LLAMA2_RUN_WITH_SINGLE_LABEL = False



'''
EXPIREMENT FIVE Llama2

TIME: around 1,5 hours
'''
L05_LLAMA2_SELF_LABELED_DATA = "../../data/selv-labeled-data/L05/selfLabeledDataJSONL05.json"
L05_LLAMA2_PROMT_INPUT = "Extract all entities in the following context along with their label from the entities at your disposal:"
L05_LLAMA2_PROMT_LABELES = "Organisation, Person, Location, Money, Temporal, Weapon, MilitaryPlatform"
L05_LLAMA2_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
L05_LLAMA2_OUTPUT_DIR = "../../L05"
L05_LLAMA2_SENTENCE_LABELS = [
        "Organisation",
        "Person",
        "Location",
        "Money",
        "Temporal",
        "Weapon",
        "MilitaryPlatform"
    ]
L05_LLAMA2_LEARNING_RATE = 3e-4
L05_LLAMA2_TRAIN_BATCH_SIZE = 16
L05_LLAMA2_EVAL_BATCH_SIZE = 16
L05_LLAMA2_EPOCHS = 4
L05_LLAMA2_WEIGHT_DECAY = 1e-5
L05_LLAMA2_LOGGING_DIR = L05_LLAMA2_OUTPUT_DIR + "/logging"
L05_LLAMA2_LOGGING_STEPS = 10
L05_LLAMA2_RUN_WITH_SENTENCE_SETTING = True
L05_LLAMA2_RUN_WITH_DOCUMENT_SETTING = False
L05_LLAMA2_REDUCED_LABELS = True
L05_LLAMA2_RUN_WITH_SINGLE_LABEL = False



'''
EXPIREMENT SIX Llama2

TIME: Around 2 hours
'''
L06_LLAMA2_SELF_LABELED_DATA = "../../data/selv-labeled-data/L06/selfLabeledDataJSONL06.json"
L06_LLAMA2_PROMT_INPUT = "Extract all entities in the following context along with their label from the entities at your disposal:"
L06_LLAMA2_PROMT_LABELES = "Organisation, Person, Location, Money, Temporal, Weapon, MilitaryPlatform"
L06_LLAMA2_SENTENCE_LABELS = [
        "Organisation",
        "Person",
        "Location",
        "Money",
        "Temporal",
        "Weapon",
        "MilitaryPlatform"
    ]
L06_LLAMA2_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
L06_LLAMA2_OUTPUT_DIR = "../../L06"
L06_LLAMA2_LEARNING_RATE = 3e-4
L06_LLAMA2_TRAIN_BATCH_SIZE = 16
L06_LLAMA2_EVAL_BATCH_SIZE = 16
L06_LLAMA2_EPOCHS = 4
L06_LLAMA2_WEIGHT_DECAY = 1e-5
L06_LLAMA2_LOGGING_DIR = L06_LLAMA2_OUTPUT_DIR + "/logging"
L06_LLAMA2_LOGGING_STEPS = 10
L06_LLAMA2_RUN_WITH_SENTENCE_SETTING = True
L06_LLAMA2_RUN_WITH_DOCUMENT_SETTING = False
L06_LLAMA2_REDUCED_LABELS = True,
L06_LLAMA2_RUN_WITH_SINGLE_LABEL = False


'''
EXPIREMENT SEVEN Llama2

TIME: around 10 hours
'''
L07_LLAMA2_SELF_LABELED_DATA = "../../data/selv-labeled-data/L07/selfLabeledDataJSONL07.json"
L07_LLAMA2_PROMT_INPUT = "Extract all entities in the following context along with the label associated that was defined above:"
L07_LLAMA2_PROMT_LABELES = "Organisation, Person, Location, Money, Temporal, Weapon, MilitaryPlatform"
L07_LLAMA2_SENTENCE_LABELS = [
        "Organisation",
        "Person",
        "Location",
        "Money",
        "Temporal",
        "Weapon",
        "MilitaryPlatform"
    ]
L07_LLAMA2_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
L07_LLAMA2_OUTPUT_DIR = "../../L07"
L07_LLAMA2_LEARNING_RATE = 3e-4
L07_LLAMA2_TRAIN_BATCH_SIZE = 16
L07_LLAMA2_EVAL_BATCH_SIZE = 16
L07_LLAMA2_EPOCHS = 4
L07_LLAMA2_WEIGHT_DECAY = 1e-5
L07_LLAMA2_LOGGING_DIR = L07_LLAMA2_OUTPUT_DIR + "/logging"
L07_LLAMA2_LOGGING_STEPS = 10
L07_LLAMA2_RUN_WITH_SENTENCE_SETTING = False
L07_LLAMA2_RUN_WITH_DOCUMENT_SETTING = False
L07_LLAMA2_REDUCED_LABELS = True,
L07_LLAMA2_RUN_WITH_SINGLE_LABEL = True


'''
EXPIREMENT EIGHT Llama2

TIME: around 2 hours
'''
L08_LLAMA2_SELF_LABELED_DATA = "../../data/selv-labeled-data/L08/selfLabeledDataJSONL08.json"
L08_LLAMA2_PROMT_INPUT = "Extract all entities in the following context along with their label from the entities at your disposal:"
L08_LLAMA2_PROMT_LABELES = "Organisation, Person, Location, Money, Temporal, Weapon, MilitaryPlatform"
L08_LLAMA2_SENTENCE_LABELS = [
        "Organisation",
        "Person",
        "Location",
        "Money",
        "Temporal",
        "Weapon",
        "MilitaryPlatform"
    ]
L08_LLAMA2_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
L08_LLAMA2_OUTPUT_DIR = "../../L08"
L08_LLAMA2_LEARNING_RATE = 3e-4
L08_LLAMA2_TRAIN_BATCH_SIZE = 16
L08_LLAMA2_EVAL_BATCH_SIZE = 16
L08_LLAMA2_EPOCHS = 4
L08_LLAMA2_WEIGHT_DECAY = 1e-5
L08_LLAMA2_LOGGING_DIR = L08_LLAMA2_OUTPUT_DIR + "/logging"
L08_LLAMA2_LOGGING_STEPS = 10
L08_LLAMA2_RUN_WITH_SENTENCE_SETTING = True
L08_LLAMA2_RUN_WITH_DOCUMENT_SETTING = False
L08_LLAMA2_REDUCED_LABELS = True,
L08_LLAMA2_RUN_WITH_SINGLE_LABEL = False


'''
EXPIREMENT NINE Llama2

TIME: around 5 hours
'''
L09_LLAMA2_SELF_LABELED_DATA = "../../data/selv-labeled-data/L09/selfLabeledDataJSONL09.json"
L09_LLAMA2_PROMT_INPUT = "Extract all entities in the following context along with their label from the entities at your disposal:"
L09_LLAMA2_PROMT_LABELES = "Organisation, Person, Location, Money, Temporal, Weapon, MilitaryPlatform"
L09_LLAMA2_SENTENCE_LABELS = [
        "Organisation",
        "Person",
        "Location",
        "Money",
        "Temporal",
        "Weapon",
        "MilitaryPlatform"
    ]
L09_LLAMA2_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
L09_LLAMA2_OUTPUT_DIR = "../../L09"
L09_LLAMA2_LEARNING_RATE = 3e-4
L09_LLAMA2_TRAIN_BATCH_SIZE = 16
L09_LLAMA2_EVAL_BATCH_SIZE = 16
L09_LLAMA2_EPOCHS = 4
L09_LLAMA2_WEIGHT_DECAY = 1e-5
L09_LLAMA2_LOGGING_DIR = L09_LLAMA2_OUTPUT_DIR + "/logging"
L09_LLAMA2_LOGGING_STEPS = 10
L09_LLAMA2_RUN_WITH_SENTENCE_SETTING = False
L09_LLAMA2_RUN_WITH_DOCUMENT_SETTING = False
L09_LLAMA2_REDUCED_LABELS = True,
L09_LLAMA2_RUN_WITH_SINGLE_LABEL = True