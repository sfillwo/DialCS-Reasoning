import random
random.seed(1234)
from modeling.models.convosense_model.t5 import GPT_CS_GENERATOR, HUM_CS_GENERATOR
from modeling.models.convosense_model.utils import COMMONSENSE_LS
import modeling.data.load as data_loader

if __name__ == '__main__':
    model = GPT_CS_GENERATOR
    model.tokenizer, model.collator = model.load_data_processing(model.modelpath)
    model.load_model(model.modelpath)
    model.attach()

    dir = 'CommonsenseDialogues'
    file = 'reflect_shuffled.json'
    save_file = 'reflect_shuffled_predictedcs_0_100.json'
    data = data_loader.load_data(
        dir=dir,
        file=file
    )
    K = 100
    data = data.collection[:K]
    items_with_q = []
    for dialogue in data:
        for turn in dialogue.turns_to_execute():
            for q in COMMONSENSE_LS:
                item = {
                    'id': dialogue.dialogue_id,
                    'uid': turn.uid,
                    'turn_obj': turn,
                    'context': dialogue.context(turn.uid),
                    'cs': turn.cs,
                    'turns': [t.utt for t in dialogue.turns[:turn.uid+1]],
                    'question': q
                }
                items_with_q.append(item)

    formatted_data = model.format_data(data=items_with_q)
    model.generate(data=formatted_data)
    modelname = model.name

    for result in formatted_data:
        turn = result['turn_obj']
        turn.beam_cs[result['question']] = result['beamed_generations'][:5]

    data_loader.rebuild_and_save_cs(
        data=data, 
        dir=dir, 
        file=save_file
    )