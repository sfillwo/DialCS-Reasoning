import os
from dataclasses import dataclass
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    PreTrainedTokenizer
)
from modeling.models.convosense_model.utils import BEAMED_GENERATIONS, BEAMED_GENERATIONS_SCORES
from modeling.models.model_settings import GenerationConfig, DataConfig, ModelWrapper
from modeling.models.convosense_model.utils import format_data
from modeling.utils import chunks

class CommonsenseGenerator(ModelWrapper):

    def load_data_processing(self, checkpoint):
        print(f'Loading {checkpoint}: tokenizer, collator...')
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        collator = DataCollatorForT5(
            tokenizer=tokenizer,
            source_max_len=768,
            predict_with_generate=True,
        )
        return tokenizer, collator

    def load_model(self, checkpoint):
        super().load_model(checkpoint)
        self.model = T5ForConditionalGeneration.from_pretrained(
            checkpoint, 
            quantization_config=self.quantization_config
        )
        self.model.config.n_positions = 768
        self.attach()

    def format_data(self, data):
        return format_data(
            data=data,
            format_string=self.data_config.format,
            context_length=self.data_config.context_length,
            prefix=self.data_config.prefix,
            disable_tqdm=True
        )

    def generate(self, data, disable_tqdm=False, return_dict_in_generate=False, output_scores=False):
        batch_size = self.batch_size
        for k in tqdm(range(0, len(data), batch_size), desc='Getting generations', disable=disable_tqdm):
            items_ = data[k: min(k + batch_size, len(data))]
            items = self.collator([{'input': item['input'], 'output': ''} for item in items_])
            input_ids = items["input_ids"]
            attention_mask = items["attention_mask"]
            beam_outputs = self.model.generate(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                max_new_tokens=400,
                num_beams=self.generation_config.num_beams,
                num_beam_groups=self.generation_config.num_beam_groups,
                diversity_penalty=self.generation_config.diversity_penalty,
                early_stopping=True,
                num_return_sequences=self.num_to_gen,
                repetition_penalty=self.generation_config.repetition_penalty,
                return_dict_in_generate=return_dict_in_generate,
                output_scores=output_scores,
            )
            if return_dict_in_generate and output_scores:
                for idx, chunk in enumerate(zip(
                        chunks(beam_outputs.sequences, self.num_to_gen),
                        chunks(beam_outputs.sequences_scores, self.num_to_gen),
                    )
                ):
                    beamed_generations = self.tokenizer.batch_decode(chunk[0], skip_special_tokens=True)
                    sequence_scores = chunk[1].tolist()
                    items_[idx][BEAMED_GENERATIONS_SCORES] = sequence_scores
                    items_[idx][BEAMED_GENERATIONS] = beamed_generations
            else:
                for idx, chunk in enumerate(chunks(beam_outputs, self.num_to_gen)):
                    beamed_generations = self.tokenizer.batch_decode(chunk, skip_special_tokens=True)
                    items_[idx][BEAMED_GENERATIONS] = beamed_generations


@dataclass
class DataCollatorForT5(object):
    tokenizer: PreTrainedTokenizer
    source_max_len: int
    predict_with_generate: bool

    def __call__(self, instances):
        inputs = [d['input'] for d in instances]
        model_inputs = self.tokenizer(inputs, max_length=self.source_max_len, padding=True, truncation=True, return_tensors='pt')
        if not self.predict_with_generate:
            model_inputs['labels'] = self.get_label_encoding([d['output'] for d in instances])
        return model_inputs

    def get_label_encoding(self, labels):
        label_encoding = self.tokenizer(labels, return_tensors='pt', padding=True).input_ids
        label_encoding[label_encoding == 0] = -100
        return label_encoding





GPT_CS_GENERATOR = CommonsenseGenerator(
    name="best_gpt_mono_dbs",
    modelpath=f'sefinch/ConvoSenseGenerator',
    device='cuda',
    use_commonsense=False,
    generation_config=GenerationConfig(
        repetition_penalty=1.0,
        num_beams=10,
        num_beam_groups=10,
        diversity_penalty=0.5,
        temperature=None,
        top_p=None
    ),
    data_config=DataConfig(
        context_length=7,
        format="{context}\n\n[Question] {question}\n[Answer]",
        prefix="provide a reasonable answer to the question based on the dialogue:\n"
    ),
    batch_size=4,
    num_to_gen=10
)