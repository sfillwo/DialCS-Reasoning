import os
import textwrap
from dataclasses import dataclass
from modeling.models.model_settings import ModelWrapper
from tqdm import tqdm
from modeling.utils import chunks
from modeling.data.dialogues_struct import Response
from promptium.gpt import OpenAiAccount, GPT
from promptium.prompt import Prompt

account = OpenAiAccount()


class GPTChatResponderOnestage(ModelWrapper):

    def attach(self):
        return
    
    def unattach(self):
        return

    def load_data_processing(self, checkpoint):
        return None, None

    def load_model(self, checkpoint):
        self.model = GPT(
            account, 
            model=checkpoint, 
            temperature=self.generation_config.temperature
        )
        # self.model = chatgpt

    def format_data(self, data):
        formatted_data = []
        for turn_dict in data:
            inferences = None
            if self.use_commonsense:
                if self.inference_preprocessor is None:
                    inferences = [
                        (f"* {self.data_config.inference_representation[key] if self.data_config.inference_representation else key} {value}", key)
                        for key, value in turn_dict['cs'].items()
                    ]
                    inferences = '\n'.join([x[0] for x in inferences])
                else:
                    inferences = self.inference_preprocessor(self, turn_dict['cs'])
            formatted_prompt = textwrap.dedent(self.data_config.format.cs_prompt)
            turns = turn_dict['context'].replace('Speaker:', self.speaker_tag).replace('Listener:', self.listener_tag)
            filled_prompt = formatted_prompt.format(context=turns, inferences=inferences)
            formatted_data.append({
                'input': filled_prompt,
                **{k:v for k,v in turn_dict.items() if k != 'input'}
            })
        return formatted_data

    def generate(self, data, disable_tqdm=False):
        for item in tqdm(data, desc='Generating from gpt', disable=disable_tqdm):
            response_prompt = Prompt(self.model, template=item['input'], store=self.store)
            response_output = response_prompt()
            item['output_response'] = response_output
        return data
            