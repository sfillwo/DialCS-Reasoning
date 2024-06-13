"""

"""

from modeling.data.dialogues_struct import Response
from collections import namedtuple
import re
import csv
import random

type_shortphrase = {
    "What could have caused the last thing said to happen?": "Caused by:",
    "How does Listener feel because of what Speaker just said?": "Listener reaction:",
    "How is Speaker feeling after what they just said?": "Speaker reaction:",
    "What might happen after what Speaker just said?": "Next:",
    "What is a likely characteristic of Speaker based on what they just said?": "Speaker characteristic:",
    "What will Listener want to do next based on what Speaker just said?": "Listener desire:",
    "What is a breakdown of the last thing said into a series of required subevents?": "Subevents:",
    "What is an emotion or human drive that motivates Speaker based on what they just said?": "Speaker motivation:",
    "What prerequisites are required for the last thing said to occur?": "Prerequisites:",
    "What does Speaker want to do next?": "Speaker desire:"
}

type_to_question = {
    "Cause": "What could have caused the last thing said to happen?",
    "React_o": "How does Listener feel because of what Speaker just said?",
    "React": "How is Speaker feeling after what they just said?",
    "Subsequent": "What might happen after what Speaker just said?",
    "Attribute": "What is a likely characteristic of Speaker based on what they just said?",
    "Desire_o": "What will Listener want to do next based on what Speaker just said?",
    "Constituent": "What is a breakdown of the last thing said into a series of required subevents?",
    "Motivation": "What is an emotion or human drive that motivates Speaker based on what they just said?",
    "Prerequisite": "What prerequisites are required for the last thing said to occur?",
    "Desire": "What does Speaker want to do next?"
}

type_sentence_prefix = {
    "What could have caused the last thing said to happen?": "I think it is possible the previous dialogue turn was caused by",
    "How does Listener feel because of what Speaker just said?": "The Listener (You) feels",
    "How is Speaker feeling after what they just said?": "I think the Speaker (Other) feels",
    "What might happen after what Speaker just said?": "Next, I predict",
    "What is a likely characteristic of Speaker based on what they just said?": "I think the Speaker (Other) is",
    "What will Listener want to do next based on what Speaker just said?": "The Listener (You) wants",
    "What is a breakdown of the last thing said into a series of required subevents?": "I think it is possible the previous dialogue turn depends on",
    "What is an emotion or human drive that motivates Speaker based on what they just said?": "I think the Speaker (Other) is motivated",
    "What prerequisites are required for the last thing said to occur?": "I think it is possible the previous dialogue turn requires",
    "What does Speaker want to do next?": "I think the Speaker (Other) wants"
}

fewshot_cs_mapping = {
    'The previous dialogue turn': 'I think it is possible the previous dialogue turn',
    'The Speaker (Other)': 'I think the Speaker (Other)',
    'Next,': 'Next, I predict'
}

example_csv_dict = csv.DictReader(open('modeling/data/fewshots/convosense_humanwritten_training.csv'))
selection_fewshot_dialogues = [
    'soda_13322_surethank_eliyahu_crackers_irritable',
    'soda_10349_projectbased_shes_strict_engage',
    'soda_6909_stressinducing_azariyah_festive_clerk',
    'soda_7572_undercooked_undercook_preheated_burnt',
    'soda_16098_councilman_outspend_council_flagstaff',
    'soda_5218_seeps_ol_humidity_juggling',
    'soda_8089_alligator_alligators_swamp_caves',
    'soda_13857_khloie_lol_teleport_foster',
    'soda_12811_spiraled_reconciling_disagreement_disagreements',
    'soda_6970_daenerysi_naysayers_daenerys_insurmountable'
]
cs_examples_str = ''
response_examples_str = {}
for row in example_csv_dict:
    if row['Selection'].strip() != "" and "*" in row['Selection']:
        converted_cs = row['All CS']
        for k, v in fewshot_cs_mapping.items():
            converted_cs = converted_cs.replace(k, v)
        converted_selection = row['Selection']
        for k, v in fewshot_cs_mapping.items():
            converted_selection = converted_selection.replace(k, v)
        all_cs = '\n'.join([f'* {cs}' for cs in converted_cs.split('\n') if cs != ''])

        if row['Dialogue ID'] in selection_fewshot_dialogues:
            cs_examples_str += f"[Example]\n\n# Dialogue History\n{row['Dialogue']}\n\n# Talking Points\n{all_cs}\n\nSelection:\n{converted_selection}\n\n"

        cstype_of_example = [k for k, v in row.items() if v == '1' and k in type_to_question]
        assert len(cstype_of_example) == 1
        cstype_of_example = type_to_question[cstype_of_example[0]]
        response_examples_str.setdefault(cstype_of_example, '')
        response_examples_str[cstype_of_example] += f"[Example]\n\n# Dialogue History\n{row['Dialogue']}\n\n# Talking Points\n{converted_selection}\n\nListener's Response:\n{row['Response']}\n\n"

response_examples_str = {k: v.strip() for k,v in response_examples_str.items()}
cs_examples_str = cs_examples_str.strip()

cs_prompt="""\
You find yourself in the role of a conversational architect, who is responsible for setting up the next exchange in the ongoing dialogue presented in "Dialogue History." Specifically, your task is to review the series of talking points provided in "Talking Points" and select the best 1 idea that will craft an engaging and cohesive response for the Listener to say. Write your selected talking point into a field titled "Selection".

Review the following examples of good selections for different pairs of "Dialogue History" and "Talking Points".

{examples}

Now, select the best talking point for the following pair:

# Dialogue History
{context}

# Talking Points
{inferences}

Selection:
"""

response_prompt="""\
You are the Listener in a conversation shown in "Dialogue History".

Your goal is write a casual yet engaging and appropriate next response for the Listener (You) in the provided dialogue. First, sufficiently answer all questions posed by Speaker (Other) in their preceding turn. Then, continue your response by including the talking points shown in "Talking Points" since you want to cover them in your next response too.

Write the response in the following format:

Listener's Response:
___

Review the following examples to understand how to write a response given a "Dialogue History" and set of "Talking Points".

{examples}

Now, complete the tasks for the following situation:

# Dialogue History
{context}

# Talking Points
{inferences}

Listener's Response:
"""

cs_prompt = cs_prompt.format(examples=cs_examples_str, context='{context}', inferences='{inferences}')

TwoStagePrompt = namedtuple('TwoStagePrompt', ['cs_prompt', 'response_prompt'])
prompt = TwoStagePrompt(cs_prompt, response_prompt)


def parse(item, stage='final'):
    if stage == 'final':
        return Response(
            text=item['output_response'].text,
            selected_cs=item['output_cs'].text.split('\n'),
            original=f"{item['output_cs'].original} ||| {item['output_response'].original}"
        )
    elif stage == 'cs' or stage == 'response':
        section = [
            x.strip() 
            for x in item.split('\n') 
        ]
        section = '\n'.join(section)
    else:
        raise NotImplementedError(f'Stage {stage} is not supported...')  
    return Response(
        text=section,
        original=item
    )  