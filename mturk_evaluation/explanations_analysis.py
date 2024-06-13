import csv, json
import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from promptium.gpt import OpenAiAccount, GPT
from promptium.prompt import Prompt
from modeling.utils import chunks
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

account = OpenAiAccount()

LEMMATIZER = WordNetLemmatizer()
LEMMATIZER.transform = LEMMATIZER.lemmatize
STEMMER = PorterStemmer()
STEMMER.transform = STEMMER.stem
PATTERN = re.compile(r'^\d+\) ') # 1) 2) ...

MODEL_MAP = {
    'CSIGSR': 'ConvoSense-E',
    'CoRP$_explicit$': 'ConvoSense-E',
    'CoRP$_latent$': 'ConvoSense-I',
    'CREST-E': 'ConvoSense-E',
    'CREST-I': 'ConvoSense-I',
}

MODELS = ['ConvoSense-E', 'ConvoSense-I', 'Doctor', 'ChatGPT']

TERM_MAP = {}

def map(transformer, word):
    result = transformer.transform(word)
    if result not in TERM_MAP:
        TERM_MAP[result] = set()
    TERM_MAP[result].add(word)
    return result

def preprocess_predicted(predicted, stemmer=False, lemmatizer=False):
    predicted = predicted.strip('-')
    predicted = re.sub(PATTERN, '', predicted)
    predicted = [x.strip().lower() for x in predicted.split(',')]
    # strip out non-aspect results from predicted
    predicted = [x for x in predicted if x not in {'high-quality', 'high quality', 'overall quality', 'overall', 'dialogue', 'quality', 'n/a', 'na', 'none'}]
    if stemmer:
        predicted = [map(STEMMER, x) for x in predicted]
    if lemmatizer:
        predicted = [map(LEMMATIZER, x) for x in predicted]
    return predicted

def get_data(file, stemmer=False, lemmatizer=False):
    reader = csv.reader(open(file))
    all_true, all_predicted = [], []
    for i, row in enumerate(reader):
        if i > 0:
            true_ = row[1]
            true = [x.strip().lower() for x in true_.split(',')]
            predicted_ = row[2]
            predicted = preprocess_predicted(predicted_, stemmer, lemmatizer)
            if stemmer:
                true = [map(STEMMER, x) for x in true]
            if lemmatizer:
                true = [map(LEMMATIZER, x) for x in true]
            all_true.append(true)
            all_predicted.append(predicted)
    return all_true, all_predicted

def precision(true, predicted):
    correct, total = 0, 0
    for true_ls, predicted_ls in zip(true, predicted):
        for p in predicted_ls:
            total += 1
            if p in true_ls:
                correct += 1
    print(f'Precision: {correct} / {total} = {correct/total:.2f}')

def recall(true, predicted):
    correct, total = 0, 0
    for true_ls, predicted_ls in zip(true, predicted):
        for t in true_ls:
            total += 1
            if t in predicted_ls:
                correct += 1
    print(f'Recall: {correct} / {total} = {correct/total:.2f}')

def validate_gpt_analysis():
    all_true, all_predicted = get_data(
        file='src/explanations_materials/explanations_validation.csv',
        stemmer=True,
        lemmatizer=False
    )
    precision(all_true, all_predicted)
    recall(all_true, all_predicted)

    print(json.dumps({k: list(v) for k,v in TERM_MAP.items()}, indent=2))


PROMPT_TEMPLATE = """I have received feedback from human judges explaining their preference for a certain dialogue response from the options displayed to them. For each of the following explanations, please list the positive aspects identified. Aspects should be one word only, so please summarize the positive traits identified into one word if needed. Examples of aspects that could be mentioned are empathy, engagement, curiosity, acknowledgement, support, naturalness, and more. 

Output a list of aspects for each explanation below.

{feedback10}"""

def predict_better_aspects():
    gpt = GPT(
            account, 
            model='gpt-3.5-turbo-0125', 
            temperature=1.0
    )

    csvreader = csv.DictReader(open('src/explanations_materials/explanation_outputs.csv'))
    rows = [x for x in csvreader]
    chunked_input = chunks(rows, 10)
    for chunk in tqdm(chunked_input, desc='Predicting better aspects'):
        explanations = [f"{i+1}) {e['explanation_transformed']}" for i, e in enumerate(chunk)]
        explanations10 = '\n'.join(explanations)
        prompt_str = PROMPT_TEMPLATE.format(feedback10=explanations10)
        prompt = Prompt(gpt, template=prompt_str, store='gptcache42624')
        predicted_aspects = prompt()
        predicted_aspects = [x for x in predicted_aspects.split('\n') if x.strip() != '']
        for c, p in zip(chunk, predicted_aspects):
            c['predicted_aspects'] = p
    csvwriter = csv.DictWriter(
        open('src/explanations_materials/explanation_outputs_aspects_3.csv', 'w'),
        fieldnames=['predicted_aspects', 'explanation_transformed', 'explanation', 'dialogue_id', 'worker_id', 'modelpair', 'winning_model', 'winning_model_finegrained']
    )
    csvwriter.writeheader()
    csvwriter.writerows(rows)

CONCEPT_TO_TERM = {
    'specific': ['specif', 'uniqu',  'person', 'express', 'attent', 'follow-up quest', 'follow-up', 'inform', 'relevant quest', 'acknowledg', 'address'],
    'engaging': ['engag', 'meaning', 'interest', 'impact', 'thought-provok', 'compel', 'interesting discuss'],
    'support': ['understand', 'support', 'concern', 'encourag', 'appreci', 'gratitud', 'reassur', 'thought', 'care', 'comfort', 'kind', 'valid', 'consider', 'genuine interest', 'connect', 'genuin interest', 'genuine concern', 'sincer', 'genuin'],
    'empathy': ['empathi', 'empathet',  'emot', 'feel', 'sympathi', 'sympath'],
    'natural': ['natur', 'flow', 'human', 'smooth', 'relat', 'human-lik', 'humanlik', 'coher'],
    'positivity': ['posit', 'enthusiasm', 'excit', 'enthusiast', 'enjoy', 'positive ton', 'friendli', 'welcom', 'invit', 'warmth', 'uplift', 'positive reinforc', 'heartfelt'],
    'helpful': ['construct', 'practic', 'assist', 'solut', 'advic', 'suggest', 'help', 'practical advic', 'collabor', 'solution-ori', 'problem-solv', 'resolut', 'practical help', 'potential solut'],
    'relevant': ['relev', 'continu', 'appropri', 'relat'],
    'detailed': ['depth', 'detail', 'comprehens', 'deeper', 'deep'],
    'proactive': ['proactiv', 'interact', 'confid', 'active listen', 'curios'],
    'etiquette': ['polit', 'respect', 'courteou'],
    'humor': ['humor', 'humour'],
}

TERM_TO_CONCEPT = {
    v: k 
    for k, v_ls in CONCEPT_TO_TERM.items()
    for v in v_ls
}

def construct_concepts_distribution(agg_aspects, figname, do_print=True, do_figure=True):
    sorted_agg_aspects = {k:v for k,v in sorted(agg_aspects.items(), key=lambda x: x[1], reverse=True)}

    if do_print:
        print('ASPECTS')
        for aspect, count in sorted_agg_aspects.items():
            print(f"{aspect} [{','.join(TERM_MAP[aspect])}] => {count}")

    concept_counts = {}
    for aspect, count in sorted_agg_aspects.items():
        concept = TERM_TO_CONCEPT.get(aspect, aspect)
        if concept not in concept_counts:
            concept_counts[concept] = 0
        concept_counts[concept] += count

    sorted_agg_concepts = {k:v for k,v in sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)}
    
    if do_print:
        print()
        print()
        print('CONCEPTS')
        for concept, count in sorted_agg_concepts.items():
            print(f"{concept} => {count}")

    if do_figure:
        plot(sorted_agg_concepts, figname=figname)

    return sorted_agg_concepts

def plot(sorted_agg_concepts, figname):
    concepts = [k for k,v in list(sorted_agg_concepts.items()) if k in CONCEPT_TO_TERM]
    counts = [v for k,v in list(sorted_agg_concepts.items()) if k in CONCEPT_TO_TERM]
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.bar(concepts, counts, color='gray')
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(fontsize=24)
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig(f'fig/{figname}.pdf')

def process_outputs():
    csvreader = csv.DictReader(open('src/explanations_materials/explanation_outputs_aspects.csv'))

    agg_aspects = {}
    for item in csvreader:
        predicted_aspects = preprocess_predicted(item['predicted_aspects'], stemmer=True)
        for a in predicted_aspects:
            winning_model = MODEL_MAP.get(item['winning_model'], item['winning_model'])
            if winning_model not in agg_aspects:
                agg_aspects[winning_model] = {}
            if a not in agg_aspects[winning_model]:
                agg_aspects[winning_model][a] = 0
            agg_aspects[winning_model][a] += 1


    # overall aspect counts (stemmed)

    all_agg_aspects = {}
    for model, model_aspects in agg_aspects.items():
        for aspect, count in model_aspects.items():
            if aspect not in all_agg_aspects:
                all_agg_aspects[aspect] = 0
            all_agg_aspects[aspect] += count
    
    print('OVERALL')
    construct_concepts_distribution(all_agg_aspects, 'all_aspects')

    # get explanation counts per model
    csvreader = csv.DictReader(open('src/explanations_materials/explanation_outputs_aspects.csv'))

    model_explanation_counts = {}
    for item in csvreader:
        winning_model = MODEL_MAP.get(item['winning_model'], item['winning_model'])
        if winning_model not in model_explanation_counts:
            model_explanation_counts[winning_model] = 0
        model_explanation_counts[winning_model] += 1

    # overall aspect counts per model (stemmed)

    print()
    print()
    print('MODEL-WISE')

    order = None
    aspect_percentages = {}
    for model, model_aspects in agg_aspects.items():
        concepts_distribution = construct_concepts_distribution(model_aspects, f'{model}_aspects', do_print=False, do_figure=False)
        # total = sum(concepts_distribution.values())
        aspect_percentages[model] = {
            k: v/model_explanation_counts[model] for k,v in concepts_distribution.items()
        }

    # get aspect order based on sorted ConvoSense-E
    order = sorted(aspect_percentages['ConvoSense-E'].items(), key=lambda x: x[1], reverse=True)
    order = [k for k,_ in order if k in CONCEPT_TO_TERM]
    order += [k for k in CONCEPT_TO_TERM if k not in order]

    # do grouped barplot with percentages
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'mediumpurple']
    plt.figure(figsize=(20, 6))  # Adjust the figure size as needed
    plt.margins(x=0.02)
    bar_width = 1.0
    for i, model in enumerate(MODELS):
        percentages = aspect_percentages[model]
        concepts = [k for k in order]
        percs = [percentages.get(k, 0) for k in order]
        r = np.arange(len(concepts))*6
        plt.bar(r + i * bar_width, percs, color=colors[i], width=bar_width, label=model if model != 'ChatGPT' else 'GPT')
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(
        [r + bar_width * (len(aspect_percentages) - 1) / 2 for r in [x*6 for x in range(len(concepts))]], 
        concepts,
        rotation=0, ha='center', fontsize=22
    )
    plt.yticks([0, 0.10, 0.20, 0.30, 0.40], fontsize=24)
    plt.legend(fontsize=22)
    plt.title('')
    plt.tight_layout()
    figname = 'modelwise_aspects'
    plt.savefig(f'fig/{figname}.pdf')



    # model-pair aspect counts (stemmed)

    csvreader = csv.DictReader(open('src/explanations_materials/explanation_outputs_aspects.csv'))

    agg_modelpair_aspects = {}
    modelpair_explanation_counts = {}
    for item in csvreader:
        predicted_aspects = preprocess_predicted(item['predicted_aspects'], stemmer=True)
        modelpair = item['modelpair']
        for orig_modelname, map_modelname in MODEL_MAP.items():
            modelpair = modelpair.replace(orig_modelname, map_modelname)
        winning_model = MODEL_MAP.get(item['winning_model'], item['winning_model'])
        if modelpair not in modelpair_explanation_counts:
            modelpair_explanation_counts[modelpair] = {}
        if winning_model not in modelpair_explanation_counts[modelpair]:
            modelpair_explanation_counts[modelpair][winning_model] = 0
        modelpair_explanation_counts[modelpair][winning_model] += 1
        if modelpair not in agg_modelpair_aspects:
            agg_modelpair_aspects[modelpair] = {}
        if winning_model not in agg_modelpair_aspects[modelpair]:
            agg_modelpair_aspects[modelpair][winning_model] = {}
        for a in predicted_aspects:
            if a not in agg_modelpair_aspects[modelpair][winning_model]:
                agg_modelpair_aspects[modelpair][winning_model][a] = 0
            agg_modelpair_aspects[modelpair][winning_model][a] += 1
    



    colors = {
        'ConvoSense-E': 'skyblue', 
        'ConvoSense-I': 'lightcoral', 
        'Doctor': 'lightgreen', 
        'ChatGPT': 'mediumpurple'
    }
    modelpair_aspect_percentages = {}
    for modelpair, modelwise_aspects in agg_modelpair_aspects.items():
        modelpair_aspect_percentages[modelpair] = {}
        for model, model_aspects in modelwise_aspects.items():
            concepts_distribution = construct_concepts_distribution(model_aspects, f'{model}_aspects', do_print=False, do_figure=False)
            # total = sum(concepts_distribution.values())
            modelpair_aspect_percentages[modelpair][model] = {
                k: v/modelpair_explanation_counts[modelpair][model] for k,v in concepts_distribution.items()
            }
        order = CONCEPT_TO_TERM.keys()
        plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
        bar_width = 0.5
        for i, model in enumerate(modelpair_aspect_percentages[modelpair]):
            model = MODEL_MAP.get(model, model)
            percentages = modelpair_aspect_percentages[modelpair][model]
            concepts = [k for k in order]
            percs = [percentages.get(k, 0) for k in order]
            r = np.arange(len(concepts))*2
            plt.bar(r + i * bar_width, percs, color=colors[model], width=bar_width, label=model)
        plt.xlabel('')
        plt.ylabel('')
        plt.xticks(
            [r + bar_width * (len(modelpair_aspect_percentages[modelpair]) - 1) / 2 for r in [x*2 for x in range(len(concepts))]], 
            concepts,
            rotation=45, ha='right', fontsize=20
        )
        plt.yticks([0, 0.05, 0.10, 0.15, 0.20], fontsize=24)
        plt.legend(fontsize=20)
        plt.title('')
        plt.tight_layout()
        figname = f'{modelpair}_aspects'
        plt.savefig(f'fig/{figname}.pdf')

if __name__ == '__main__':
    # validate_gpt_analysis()
    process_outputs()
    # predict_better_aspects()