
from modeling.models.model_settings import ModelWrapper, GenerationConfig, DataConfig
from modeling.models.response_model.gpt_twostage_select_respond import GPTTwoStageSelectRespond
from modeling.models.response_model.gpt_fewshot_curated_twostage_compositional_single_prompt import prompt, parse, type_shortphrase, type_sentence_prefix, response_examples_str

CHATGPT_35_0125_FEWSHOT_CURATED_TWOSTAGE_COMPOSITIONAL_SINGLE = GPTTwoStageSelectRespond(
    name="chatgpt_35_fewshot_curated_twostage_compositional_single_0125",
    modelpath='gpt-3.5-turbo-0125',
    device='cpu',
    use_commonsense=True,
    generation_config=GenerationConfig(
        repetition_penalty=None,
        num_beams=None,
        num_beam_groups=None,
        diversity_penalty=None,
        temperature=0.7
    ),
    data_config=DataConfig(
        context_length=None,
        format=prompt,
        prefix=None,
        parse_func=parse,
        inference_representation=type_sentence_prefix,
        fewshots=response_examples_str
    ),
    quantization_config=None,
    batch_size=1,
    num_to_gen=1,
    store='llm_cache_0125'
)

