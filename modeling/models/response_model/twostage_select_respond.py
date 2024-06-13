# import random
# random.seed(1234)

import textwrap
from modeling.models.model_settings import ModelWrapper
from thefuzz import process as find_fuzzy, fuzz

class TwoStageSelectRespond(ModelWrapper):

    def format_data(self, data):
        """
        Format Data for Stage 1 (CS Selection)
        """
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
    
    def format_data_two(self, data):
        """
        Format Data for Stage 2 (Response Generation)
        """
        formatted_data = []
        for turn_dict in data:
            inferences = None
            if self.use_commonsense:
                inferences = turn_dict['output_cs'].text
            formatted_prompt = textwrap.dedent(self.data_config.format.response_prompt)
            turns = turn_dict['context'].replace('Speaker:', self.speaker_tag).replace('Listener:', self.listener_tag)
            if '{examples}' in formatted_prompt:
                cs_strs = [(k, f"* {self.data_config.inference_representation[k] if self.data_config.inference_representation else k} {v}") for k, v in turn_dict['cs'].items()]
                selection_matches = find_fuzzy.extract(inferences, [x[1] for x in cs_strs], scorer=fuzz.partial_ratio, limit=1)
                best_match = selection_matches[0]
                best_match_type = [x for x in cs_strs if x[1] == best_match[0]][0][0]
                fewshots = self.data_config.fewshots[best_match_type]
                filled_prompt = formatted_prompt.format(context=turns, inferences=inferences, examples=fewshots)
            else:
                filled_prompt = formatted_prompt.format(context=turns, inferences=inferences)
            formatted_data.append({
                'input': filled_prompt,
                **{k:v for k,v in turn_dict.items() if k != 'input'}
            })
        return formatted_data
            