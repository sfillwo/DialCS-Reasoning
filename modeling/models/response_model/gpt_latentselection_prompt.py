
from modeling.data.dialogues_struct import Response
import re
from collections import namedtuple
import csv

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

response_examples_str = ''
for row in example_csv_dict:
    if row['Selection'].strip() != "" and "*" in row['Selection']:
        converted_cs = row['All CS']
        for k, v in fewshot_cs_mapping.items():
            converted_cs = converted_cs.replace(k, v)
        all_cs = '\n'.join([f'* {cs}' for cs in converted_cs.split('\n') if cs != ''])

        if row['Dialogue ID'] in selection_fewshot_dialogues:
            response_examples_str += f"[Example]\n\n# Dialogue History\n{row['Dialogue']}\n\n# Talking Points\n{all_cs}\n\nListener's Response:\n{row['Response']}\n\n"

response_examples_str = response_examples_str.strip()


response_prompt = """\
You are the Listener in a conversation shown in "Dialogue History".

Your goal is write a casual yet engaging and appropriate next response for the Listener (You) in the provided dialogue. You will consider a list of possible "Talking Points" to include as you think about the best response to give, being careful to ignore any talking points that are irrelevant or unlikely predictions for the shown conversation. 

Based on the talking points, write the best response you can think of in the following format:

Listener's Response:
___

Review the following examples to understand how to write a response given a "Dialogue History" and set of possible "Talking Points".

{examples}

Now, construct the best response from the Listener for the following dialogue, based on the possible talking points:

# Dialogue History
{context}

# Talking Points
{inferences}

Listener's Response:
"""

response_prompt = response_prompt.format(examples=response_examples_str, context='{context}', inferences='{inferences}')

TwoStagePrompt = namedtuple('TwoStagePrompt', ['cs_prompt', 'response_prompt'])
prompt = TwoStagePrompt(response_prompt, None)

def parse(item, stage='final'):
    if stage == 'final':
        return Response(
            text=item['output_response'],
            original=item['output_response']
        )
    return Response(
        text=item,
        original=item
    )