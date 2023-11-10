from utils_gui.CommonVariables import COMMON_LLAMA2_PROMT_INPUT, COMMON_LLAMA2_PROMT_LABELES
import textwrap

def generate_prompt_ner(context, output=""):
     input = COMMON_LLAMA2_PROMT_INPUT
     labels = COMMON_LLAMA2_PROMT_LABELES

     formated_output = ""
     for entry in output: 
          if formated_output == "":
               formated_output = entry
          else:
               formated_output += ", " + entry
     
     prompt = textwrap.dedent(f""""<s> [INST] <<SYS>>
     Your task is to harness the capabilities of a robust entity extraction model. 
     Equipped with the knowledge of various entity types, your mission is to analyze a provided text from the military context, which includes both a question and context, and identify entities within it. 
     Your goal is to generate a comprehensive, comma-separated list that presents the identified entities alongside their respective labels. 
     The entity types at your disposal include: {labels}


     Your output should reflect the discovered entities in the given text, each tagged with its corresponding label. <</SYS>>

     ### Input:
     {input}

     ### Context:
     {context[0]} [/INST]

     ### Response:
     {formated_output} </s>""")
     
     return prompt


def generate_single_label_prompt_ner(label, context, output=""):
     input = COMMON_LLAMA2_PROMT_INPUT
     formated_output = ""
     
     for entry in output: 
          if formated_output == "":
               formated_output = entry
          else:
               formated_output += ", " + entry

     prompt = textwrap.dedent(f"""<s> [INST] <<SYS>>
     Your task is to harness the capabilities of a robust entity extraction model. 
     Equipped with the knowledge of various entity types, your mission is to analyze a provided text from the military context, which includes both a question and context, and identify entities within it. 
     Your goal is to generate a comprehensive, comma-separated list that presents the identified entities alongside the specified label. 

     Your output should reflect the discovered entities in the given text, that matches this label: {label} <</SYS>>

     ### Input:
     {input}

     ### Context:
     {context[0]} [/INST]

     ### Response:
     {output} </s>""")

     return prompt

def generate_prompt_ner_inference(context):
     input = COMMON_LLAMA2_PROMT_INPUT
     labels = COMMON_LLAMA2_PROMT_LABELES
     
     prompt = textwrap.dedent(f"""<s> [INST] <<SYS>>
     Your task is to harness the capabilities of a robust entity extraction model. 
     Equipped with the knowledge of various entity types, your mission is to analyze a provided text from the military context, which includes both a question and context, and identify entities within it. 
     Your goal is to generate a comprehensive, comma-separated list that presents the identified entities alongside their respective labels. 
     The entity types at your disposal include: {labels}


     Your output should reflect the discovered entities in the given text, each tagged with its corresponding label. <</SYS>>

     ### Input:
     {input}

     ### Context:
     {context} [/INST]""")
     
     return prompt

def generate_single_label_prompt_ner_inference(label, context):
     input = COMMON_LLAMA2_PROMT_INPUT

     prompt = textwrap.dedent(f"""<s> [INST] <<SYS>>
     Your task is to harness the capabilities of a robust entity extraction model. 
     Equipped with the knowledge of various entity types, your mission is to analyze a provided text from the military context, which includes both a question and context, and identify entities within it. 
     Your goal is to generate a comprehensive, comma-separated list that presents the identified entities alongside the specified label. 

     Your output should reflect the discovered entities in the given text, that matches this label: {label} <</SYS>>

     ### Input:
     {input}

     ### Context:
     {context[0]} [/INST] """)

     return prompt

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
