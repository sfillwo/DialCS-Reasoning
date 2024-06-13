
from modeling.models.model_settings import ModelWrapper, GenerationConfig, DataConfig
from modeling.models.response_model.gpt_onestage import GPTChatResponderOnestage
from modeling.models.response_model.gpt_chae_vanilla_prompt import prompt, parse, type_shortphrase, type_sentence_prefix

CHATGPT_35_CHAE_VANILLA_0125 = GPTChatResponderOnestage(
    name="chatgpt_35_chae_vanilla_0125",
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
        inference_representation=type_sentence_prefix
    ),
    speaker_tag='B:',
    listener_tag='A:',
    quantization_config=None,
    batch_size=1,
    num_to_gen=1,
    store='gptcache42624'
)

