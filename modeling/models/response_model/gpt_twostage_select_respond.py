
import os
import textwrap
from dataclasses import dataclass
from tqdm import tqdm
from modeling.utils import chunks
from modeling.data.dialogues_struct import Response
import random
from promptium.gpt import OpenAiAccount, GPT
from promptium.prompt import Prompt
from modeling.models.response_model.twostage_select_respond import TwoStageSelectRespond

account = OpenAiAccount()

class GPTTwoStageSelectRespond(TwoStageSelectRespond):

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

    def generate(self, data, disable_tqdm=False):
        for item in tqdm(data, desc='Generating from gpt', disable=disable_tqdm):
            selection_prompt = Prompt(self.model, template=item['input'], store=self.store)
            selection_output = selection_prompt()
            item['output_cs'] = self.data_config.parse_func(selection_output, stage='cs')
            formatted_data_two = self.format_data_two([item])
            response_prompt = Prompt(self.model, template=formatted_data_two[0]['input'], store=self.store)
            response_output = response_prompt()
            item['output_response'] = self.data_config.parse_func(response_output, stage='response')
            item['input'] += ' ||| ' + formatted_data_two[0]['input']
        return data
            