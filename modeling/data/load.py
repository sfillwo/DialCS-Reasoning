import os, json, pickle
from datetime import datetime
import cattrs
from modeling.data.dialogues_struct import Response, Turn, Dialogue, Dialogues


def load_data(dir, file=None, all_cs=False) -> Dialogues:
    if file is None:
        file = 'dialogues.json'
    if file.endswith('.json'):
        raw_data = json.load(open(f'{os.path.dirname(__file__)}/{dir}/{file}'))
        if 'collection' not in raw_data:
            raw_data = {'collection': raw_data}
        items = cattrs.structure(raw_data, Dialogues)
    elif file.endswith('.pickle'):
        raw_data = pickle.load(open(f'{os.path.dirname(__file__)}/{dir}/{file}', 'rb'))
        dialogues = {}
        for d in raw_data:
            if d['id'] not in dialogues:
                d_obj = Dialogue(
                    dialogue_id=d['id'], 
                    on_terminal=True
                )
                dialogues[d['id']] = d_obj
                turns = d['context'].split('\n')
                for i, t in enumerate(turns):
                    t_obj = Turn(
                        uid=i, 
                        sid="S1" if t.startswith("Speaker") else "S2", 
                        utt=t.replace('Speaker: ', '').replace('Listener: ', '')
                    )
                    d_obj.turns.append(t_obj)
            else:
                d_obj = dialogues[d['id']]
            assert d['question'] not in d_obj.turns[-1].cs
            d_obj.turns[-1].cs[d['question']] = d['beamed_generations'][0] if not all_cs else d['beamed_generations']
        items = Dialogues(collection=list(dialogues.values()))
    return items


def save_data(data, dir, file, to_dataloader_dir=True):
    parentdir = os.path.dirname(__file__) + '/' if to_dataloader_dir else ''
    data_to_dump = cattrs.unstructure(data)
    json.dump(
        data_to_dump,
        open(f'{parentdir}{dir}/{file}', 'w'),
        indent=2
    )


def rebuild_and_save_cs(data, dir, file=None):
    if file is None:
        file = 'dialogues-cs.json'
    save_data(data, dir, file)


def rebuild_and_save_response(data, dir, file=None):
    if file is None:
        file = f'dialogues-response-{datetime.now().strftime("%m%d%Y-%H%M%S")}.json'
    save_data(data, dir, file)