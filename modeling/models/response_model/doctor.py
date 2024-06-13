from __future__ import annotations

from dataclasses import dataclass
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM
)
from collections import namedtuple
from modeling.models.model_settings import ModelWrapper, GenerationConfig, DataConfig
from modeling.data.dialogues_struct import Response
from modeling.models.response_model.twostage_select_respond import TwoStageSelectRespond
from modeling.utils import chunks

from promptium.gpt import OpenAiAccount, GPT
from promptium.prompt import Prompt
account = OpenAiAccount()

cs_prompt = "{context}"

response_prompt = """\
Generate the most plausible next response considering the dialogue history. You can refer to the rationale, but you should ignore the rationale if it misleads the next response. Do not try to put too much information in the next response. You should follow the style of the history.

Rationale:
{inferences}
History:
{context}
Next Response:
A
"""

TwoStagePrompt = namedtuple('TwoStagePrompt', ['cs_prompt', 'response_prompt'])
prompt = TwoStagePrompt(cs_prompt, response_prompt)

def parse(item, stage='final'):
    if stage == 'final':
        return Response(
            text=item['output_response'].text,
            selected_cs=item['output_cs'].text.split('\n'),
            original=f"{item['output_cs'].original} ||| {item['output_response'].original}"
        )
    return Response(
        text=item,
        original=item
    ) 

class Doctor(TwoStageSelectRespond):

    def load_model(self, checkpoint):
        super().load_model(checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained("DLI-Lab/DOCTOR")
        self.model = AutoModelForCausalLM.from_pretrained("DLI-Lab/DOCTOR")
        self.attach()
        self.cs_collator = DataCollatorForDoctor(self.tokenizer)
        self.gpt = GPT(
            account, 
            model=checkpoint, 
            temperature=0.7
        )
        print('model loaded...')

    def generate(self, data, disable_tqdm=False):
        batch_size = self.batch_size
        for k in tqdm(range(0, len(data), batch_size), desc='Getting generations', disable=disable_tqdm):
            items_ = data[k: min(k + batch_size, len(data))]
            items = self.cs_collator(items_)
            input_ids = items["input_ids"]
            attention_mask = items["attention_mask"]
            beam_outputs = self.model.generate(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device)
            )
            for idx, chunk in enumerate(chunks(beam_outputs, self.num_to_gen)):
                chunk = chunk[0]
                only_output = chunk[len(input_ids[idx]):]
                sgeneration = self.tokenizer.decode(only_output, skip_special_tokens=True)
                gen_to_keep = sgeneration
                answer1_start = sgeneration.find('Subquestion 1')
                if answer1_start > -1:
                    sgeneration = sgeneration[answer1_start:]
                answer3_start = sgeneration.find('Subanswer 3')
                if answer3_start > -1:
                    answer3_end = sgeneration.find('Subquestion', answer3_start)
                    if answer3_end > -1:
                        gen_to_keep = sgeneration[:answer3_end]
                items_[idx]['output_cs'] = self.data_config.parse_func(gen_to_keep, stage='cs')

        formatted_for_response_gen = self.format_data_two(data)
        for item, formatted_for_response in tqdm(list(zip(data, formatted_for_response_gen)), desc='gpt responses', disable=disable_tqdm):
            response_prompt = Prompt(self.gpt, template=formatted_for_response['input'], store=self.store)
            response_output = response_prompt()
            item['output_response'] = self.data_config.parse_func(response_output, stage='response')
            item['input'] += ' ||| ' + formatted_for_response['input']


@dataclass
class DataCollatorForDoctor(object):
    tokenizer: PreTrainedTokenizer

    def __call__(self, instances):
        inputs = [d['input'] for d in instances]
        model_inputs = self.tokenizer(inputs, padding=True, return_tensors='pt')
        return model_inputs


DOCTOR = Doctor(
    name="doctor",
    modelpath='gpt-3.5-turbo-0125',
    device='cuda',
    use_commonsense=True,
    generation_config=GenerationConfig(
        repetition_penalty=None,
        num_beams=None,
        num_beam_groups=None,
        diversity_penalty=None,
        temperature=None
    ),
    data_config=DataConfig(
        context_length=None,
        format=prompt,
        prefix=None,
        parse_func=parse,
        inference_representation=None,
        fewshots=None
    ),
    quantization_config=None,
    batch_size=16,
    num_to_gen=1,
    store='llm_cache_0125',
    listener_tag='A:',
    speaker_tag='B:'
)