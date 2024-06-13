
from modeling.data.dialogues_struct import Response
from collections import namedtuple
import re

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

type_sentence_prefix = {
    "What could have caused the last thing said to happen?": "The previous dialogue turn was caused by",
    "How does Listener feel because of what Speaker just said?": "The Listener (You) feels",
    "How is Speaker feeling after what they just said?": "The Speaker (Other) feels",
    "What might happen after what Speaker just said?": "Next,",
    "What is a likely characteristic of Speaker based on what they just said?": "The Speaker (Other) is",
    "What will Listener want to do next based on what Speaker just said?": "The Listener (You) wants",
    "What is a breakdown of the last thing said into a series of required subevents?": "The previous dialogue turn depends on",
    "What is an emotion or human drive that motivates Speaker based on what they just said?": "The Speaker (Other) is motivated",
    "What prerequisites are required for the last thing said to occur?": "The previous dialogue turn requires",
    "What does Speaker want to do next?": "The Speaker (Other) wants"
}


prompt = """\
Generate the most plausible next response cosidering the dialogue history. You should follow the style of the history.
[Example 1]
History:
{context}
Next Response:
A
"""

TwoStagePrompt = namedtuple('TwoStagePrompt', ['cs_prompt', 'response_prompt'])
prompt = TwoStagePrompt(prompt, None)


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