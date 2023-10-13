def generate_prompt_ner(context, output="", input = "Extract all entities in the following context along with their label from the provided label list"):
     return f""""
     Your task is to harness the capabilities of a robust entity extraction model. 
     Equipped with the knowledge of various entity types, your mission is to analyze a provided text from the military context, which includes both a question and context, and identify entities within it. 
     Your goal is to generate a comprehensive, comma-separated list that presents the identified entities alongside their respective labels. 
     The entity types at your disposal include:

Organisation,
Nationality,
Person,
DocumentReference,
Location,
Money,
Vehicle,
Temporal,
Weapon,
Quantity,
CommsIdentifier

Your output should reflect the discovered entities in the given text, each tagged with its corresponding label.

### Input:
{input}

### Context:
{context}

### Response:
{output}"""

###Input
#Extract all entities in the following context along with their label from the provided label list:

### context:
"""
Foreign Minister Julie Bishop met with the Minister of Foreign Affairs of Iraq,
HE Dr Ibrahim Al-Ja’afari,
in Canberra on 12 February to discuss bilateral
elations and reaffirm our shared commitment to combating
the terrorist threat of Daesh (also known as ISIL).
"""
### Response:
# Foreign Minister Julie Bishop - Person,
# the Minister of Foreing affairs - Person,
# 12 February - temporal 

### Test promt:
""" Extract all entities from the following context along with their associated label. The labels must be one of the following but all labels might not be present in the context provided. Furthermore, not every word necessarily be labeled: Organisation,
Nationality,
Person,
DocumentReference,
Location,
Money,
Vehicle,
Temporal,
Weapon,
Quantity,
CommsIdentifier,
MilitaryPlatform,
Coordinate,
Frequency.

Extract from the following context: "Foreign Minister Julie Bishop met with the Minister of Foreign Affairs of Iraq, HE Dr Ibrahim Al-Ja’afari, in Canberra on 12 February to discuss bilateral relations and reaffirm our shared commitment to combating the terrorist threat of Daesh (also known as ISIL)." """