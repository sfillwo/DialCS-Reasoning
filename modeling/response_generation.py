import random
random.seed(1234)
import modeling.data.load as data_loader

from modeling.models.response_model.gpt_fewshot_curated_twostage_compositional_single_35_0125 import CHATGPT_35_0125_FEWSHOT_CURATED_TWOSTAGE_COMPOSITIONAL_SINGLE
from modeling.models.response_model.gpt_vanilla_35_0125 import CHATGPT_35_VANILLA_0125
from modeling.models.response_model.doctor import DOCTOR
from modeling.models.response_model.gpt_latentselection_35_0125 import CHATGPT_35_LATENTSELECTION_0125
from modeling.models.response_model.gpt_chae_vanilla_35_0125 import CHATGPT_35_CHAE_VANILLA_0125

if __name__ == '__main__':

    response_models = [
        (CHATGPT_35_0125_FEWSHOT_CURATED_TWOSTAGE_COMPOSITIONAL_SINGLE, 'chatgpt-35-curated-compositional-single-0125-0-100'),
        (CHATGPT_35_VANILLA_0125, 'chatgpt-35-vanilla-0125-0-100'),
        (DOCTOR, 'doctor-0125-0-100')
        (CHATGPT_35_LATENTSELECTION_0125, 'chatgpt-35-latentselection-0125-0-100')
        (CHATGPT_35_CHAE_VANILLA_0125, 'chatgpt-35-chaevanilla-0125-0-100')
    ]
    for model, savefile in response_models:
        dir = 'CommonsenseDialogues'
        inputfile = 'reflect_shuffled_predictedcs_0_100_predicteddiv5.json'
        start_idx = None
        end_idx = None
        data = data_loader.load_data(
            dir=dir, 
            file=inputfile
        )
        if start_idx is not None and end_idx is not None:
            data = data[start_idx:end_idx]

        model.load_model(model.modelpath)
        tasks_to_do = [
            (
                model.format_data([{
                    'context': dialogue.context(turn.uid), 
                    'cs': turn.cs,
                    'turns': [t.utt for t in dialogue.turns[:turn.uid+1]]
                }]), 
                turn
            ) 
            for dialogue in data for turn in dialogue.turns_to_execute()
            if model.name not in turn.response
        ]
        tasks = [t[0][0] for t in tasks_to_do]
        turns = [t[1] for t in tasks_to_do]
        model.generate(tasks, disable_tqdm=False)
        for task, turn in zip(tasks, turns):
            turn.response[model.name] = model.data_config.parse_func(task)
            turn.response[model.name].input = task['input']
        model.clear_cuda_cache()
        data_loader.rebuild_and_save_response(data=data, dir=dir, file=f'response_outputs/test/{savefile}.json')