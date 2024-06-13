import os
import csv
import random
random.seed(125)
import modeling.data.load as data_loader
import pandas as pd
from pandas import DataFrame
import json
from collections import Counter
from modeling.utils import krippendorffs_alpha
from matplotlib import pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

MODELNAMES = {
    'chatgpt-35-curated-compositional-single-0125': 'ConvoSense-E',
    'chatgpt-35-curated-compositional-triple-0125': 'triple',
    'chatgpt-35-vanilla-0125': 'GPT',
    'doctor-0125': 'Doctor',
    'chatgpt-35-latentselection-0125': 'ConvoSense-I',
    'chatgpt-35-chaevanilla-0125': 'GPT$_{chae}$'
}
def get_name(name):
    return MODELNAMES.get(name, name)

def load_results(outputsdir, csvdirs):
    files = [
        f'data/{outputsdir}/{csvdir}/{x}'
        for csvdir in csvdirs 
        for x in os.listdir(f'data/{outputsdir}/{csvdir}') 
        if x.endswith('.csv') and 'test' not in x
    ]
    records = []
    for file in files:
        for result_dict in csv.DictReader(open(file)):
            dialogue_id = result_dict["Input.dialogue_id"]
            dialogue = result_dict["Input.dialogue_context"]
            worker_id = result_dict["WorkerId"]
            model_a = get_name(result_dict["Input.model_id_a"])
            model_b = get_name(result_dict["Input.model_id_b"])
            modelpair = ' vs '.join(sorted([model_a, model_b]))
            response_a = result_dict["Input.response_a"]
            response_b = result_dict["Input.response_b"]
            metadata = [dialogue_id, worker_id, dialogue, modelpair, model_a, response_a, model_b, response_b]
            answer_dict =json.loads(result_dict["Answer.taskAnswers"])[0]
            df = pd.DataFrame(columns=["value"])
            for key, value in answer_dict.items():
                if isinstance(value, dict):
                    if len(value) > 1:
                        records.append([
                            *metadata,
                            'pairwise' if 'aspect' not in key else 'categorical',
                            key,
                            [k for k, v in value.items() if v][0]
                        ])
                    else:
                        records.append([
                            *metadata,
                            'binary',
                            key,
                            list(value.values())[0]
                        ])
                else:
                    records.append([
                        *metadata, 'text', key, value
                    ])
    df = pd.DataFrame.from_records(records, columns=['dialogue_id', 'worker_id', 'dialogue_context', 'modelpair', 'model_a', 'response_a', 'model_b', 'response_b', 'type', 'label', 'value'])
    df['winning_model'] = df.apply(winning_model, axis=1)
    df['winning_model_finegrained'] = df.apply(winning_model_finegrained, axis=1)
    df['winning_option'] = df.apply(winning_option, axis=1)
    df = df[df['label'] != 'att']
    return df

def agreement(df: DataFrame):
    pairwise_df = df.dropna(axis=0, how='any') # only pairwise questions have no NaNs (winning_model column)
    pivot_pairwise_df = pairwise_df.pivot_table(index=['dialogue_id', 'modelpair', 'label'], columns='worker_id', values='winning_option', aggfunc='first')
    for row in pivot_pairwise_df.iterrows():
        print(f"{row[0][0][:10]} {row[0][1]:10} {json.dumps(Counter([x for x in row[1] if isinstance(x, str)]))}")
    per_label_pairwise_df = pivot_pairwise_df.groupby(by=['label'])
    agreement_df = per_label_pairwise_df.apply(krippendorffs_alpha, ci=False, to_string=True, level_of_measurement='nominal')
    print(agreement_df)

def calculate_majority(df):
    counts = Counter(df['winning_model']).items()
    max_model = max(counts, key=lambda x: x[1])
    return max_model[0]

def majority_vote(df: DataFrame):
    pairwise_df = df.dropna(axis=0, how='any') # only pairwise questions have no NaNs (winning_model column)
    group_by_task = pairwise_df.groupby(by=['dialogue_id', 'modelpair', 'label'])
    majority_vote_results = pd.DataFrame(group_by_task.apply(calculate_majority))
    majority_vote_results.columns = ['winning_model']
    majority_vote_results.reset_index(inplace=True)
    return majority_vote_results

def winning_model(row):
    if row['type'] == 'pairwise':
        if 'B' in row['value']:
            winning_model = row['model_b']
        elif 'A' in row['value']:
            winning_model = row['model_a']
        else:
            print(f'WARNING: value {row["value"]} does not contain A or B!')
        return winning_model
    return None

def winning_model_finegrained(row):
    if row['type'] == 'pairwise':
        selection = row['value']
        if 'B' in selection:
            winning_model = row['model_b']
        elif 'A' in selection:
            winning_model = row['model_a']
        else:
            print(f'WARNING: value {row["value"]} does not contain A or B!')
        if 'slight' in selection:
            winning_model = 'slight-' + winning_model
        else:
            winning_model = 'def-' + winning_model
        return winning_model
    return None

def winning_option(row):
    if row['type'] == 'pairwise':
        if 'B' in row['value']:
            winning_option = 'B'
        elif 'A' in row['value']:
            winning_option = 'A'
        else:
            print(f'WARNING: value {row["value"]} does not contain A or B!')
        return winning_option
    return None

def pairwise_perc(df: DataFrame, column):
    counts = Counter(df[column].to_list())
    total = sum(counts.values())
    percs = {m: counts[m] / total for m in counts}
    new_item = pd.DataFrame(percs, index=[0])
    return new_item

def pairwise_results(df: DataFrame, finegrained=False, plot_pie=False):
    pairwise_df = df.dropna(axis=0, how='any') # only pairwise questions have no NaNs (winning_model column)
    modelpair_label_dfs = pairwise_df.groupby(by=['modelpair', 'label'])
    modelpair_percs = modelpair_label_dfs.apply(pairwise_perc, 'winning_model')
    modelpair_percs = modelpair_percs.fillna(0)
    print(modelpair_percs)
    if finegrained:
        modelpair_percs_finegrained = modelpair_label_dfs.apply(pairwise_perc, 'winning_model_finegrained')
        modelpair_percs_finegrained = modelpair_percs_finegrained.fillna(0)
        print()
        print(modelpair_percs_finegrained)

        # Creating the combined string

        for modelpair, grouped_data in modelpair_percs_finegrained.groupby('modelpair'):
            combined_string = ""
            grouped_data = grouped_data.reset_index()
            model_a, model_b = modelpair.split(' vs ')
            def_a = f"def-{model_a}"
            slight_a = f"slight-{model_a}"
            def_b = f"def-{model_b}"
            slight_b = f"slight-{model_b}"
            header = """\\begin{table*}[htb]
    \\centering
    \\begin{tabular}{c|rrrr|rr}
        \\toprule \n"""
            header_1 = "        " + "& \\multicolumn{2}{c}{\\textbf{" + model_a + "}} & \\multicolumn{2}{c|}{\\textbf{" + model_b + "}} & \\multirow{2}{*}{\\textbf{" + model_a + "}} & \\multirow{2}{*}{\\textbf{" + model_b + "}} \\\\ \n"
            header_2 = "        " + "& Definitely & Slightly & Slightly & Definitely & & \\\\"
            header_3 = "        " + '\\midrule \n'
            combined_string += header + header_1 + header_2 + header_3
            for label in ['natural', 'engaging', 'specific', 'quality']:
                for row in grouped_data.iterrows():
                    row = row[1]
                    if row['label'] == label:
                        row = "        " + f"{row['label']} & {row[def_a]*100:.1f}\\% & {row[slight_a]*100:.1f}\\% & {row[slight_b]*100:.1f}\\% & {row[def_b]*100:.1f}\\% & {(row[def_a]+row[slight_a])*100:.1f}\\% & {(row[def_b] + row[slight_b])*100:.1f}\\% \\\\ \n"
                        combined_string += row
            combined_string += "        " + '\\bottomrule \n'
            caption = f"Pairwise evaluation results for {model_a} vs {model_b} showing the fine-grained preference selections on the left and the total win percentages on the right."
            tab_label = f"tab:{model_a}-vs-{model_b}"
            combined_string += "    \end{tabular} \n"
            combined_string += "    \caption{" + caption + "} \n"
            combined_string += "    \label{" + tab_label + "} \n"
            combined_string += "\end{table*}"
            print()
            print(combined_string)
            print()

    if plot_pie:
        # Create pie chart for each group
        for modelpair, group_data in modelpair_percs_finegrained.groupby('modelpair'):
            for label, data in group_data.groupby('label'):
                sizes = data.values[0].tolist()
                chunks = data.keys().to_list()
                plt.figure(figsize=(6, 6))
                plt.pie(sizes, labels=chunks, autopct='%1.1f%%')
                plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                plt.savefig(f'fig/piechart_{modelpair}_{label}.pdf')
            
def view_results(df: DataFrame, label):
    pairwise_df = df.dropna(axis=0, how='any') # only pairwise questions have no NaNs (winning_model column)
    grouped = pairwise_df.groupby(by=['dialogue_context', 'label', 'modelpair'])
    for group_id, group in grouped:
        if group_id[1] == label:
            item = list(group[['dialogue_context', 'worker_id', 'model_a', 'response_a', 'model_b', 'response_b', 'winning_option']].iterrows())[0][1]
            print(item['dialogue_context'].replace('<br>', '\n'))
            print(f"[A {item['model_a']}] {item['response_a']}")
            print(f"[B {item['model_b']}] {item['response_b']}")   
            print(Counter(group['winning_option']))
            explanations = df[(df['label'] == 'quality-textbox') & (df['dialogue_context'] == item['dialogue_context']) & (df['model_a'] == item['model_a']) & (df['model_b'] == item['model_b'])]
            for row in explanations.iterrows():
                print(f" - {row[1]['value']}")
            print()
            print()

def influential_aspect_results(df: DataFrame):
    match = 0
    total_count = 0
    influential_aspect_results = df[df['type'] == 'categorical']
    for row in influential_aspect_results.iterrows():
        inf_aspect = row[1]['value'].replace('ness', '').replace('ity', '')
        if inf_aspect != 'other':
            total_count += 1
            aspect_row = df[(df['dialogue_id'] == row[1]['dialogue_id']) & (df['worker_id'] == row[1]['worker_id']) & (df['modelpair'] == row[1]['modelpair']) & (df['label'] == inf_aspect)]
            select_by_aspect_row = aspect_row['value'].to_list()[0]
            if 'A' in select_by_aspect_row:
                better_model_by_aspect_row = aspect_row['model_a'].to_list()[0]
            elif 'B' in select_by_aspect_row:
                better_model_by_aspect_row = aspect_row['model_b'].to_list()[0]
            else:
                print(f'WARNING: aspect row value {select_by_aspect_row} does not contain A or B!')
            quality_row = df[(df['dialogue_id'] == row[1]['dialogue_id']) & (df['worker_id'] == row[1]['worker_id']) & (df['modelpair'] == row[1]['modelpair']) & (df['label'] == 'quality')]
            winning_model = quality_row['winning_model'].to_list()[0]
            match += better_model_by_aspect_row == winning_model
    match_perc = match / total_count * 100
    print(f'Influential aspect winner matches overall winner: {match_perc:.1f}')
    aspect_counts = Counter(influential_aspect_results['value'])
    aspect_percentages = {k: f"{v/len(influential_aspect_results)*100:.1f}" for k,v in aspect_counts.items()}
    print(json.dumps(aspect_percentages, indent=2))

def explanation_analysis(df: DataFrame):
    explanation_to_worker_id = {}
    explanation_results = df[df['label'] == 'quality-textbox']
    for row in explanation_results.iterrows():
        if row[1]['value'] not in explanation_to_worker_id:
            explanation_to_worker_id[row[1]['value']] = set()
        explanation_to_worker_id[row[1]['value']].add(row[1]['worker_id'])
    counts = Counter(explanation_results['value'])
    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    for explanation, count in sorted_counts.items():
        workers = explanation_to_worker_id[explanation]
        print(explanation)
        print(f'\t{count}')
        print(f'\t {len(workers)} workers = {workers}')
        print()
    explanations = []
    csvwriter = csv.writer(open('src/explanations_materials/explanation_outputs.csv', 'w'))
    for row in explanation_results.iterrows():
        explanation = row[1]['value']
        explanation_t = explanation.lower().strip().replace('\n', ' ').replace('response a', 'response x').replace('response b', 'response x').replace('responce a', 'response x').replace('responce b', 'response x').replace('reponse a', 'response x').replace('reponse b', 'response x')
        winning_model_row = df[(df['dialogue_id'] == row[1]['dialogue_id']) & (df['worker_id'] == row[1]['worker_id']) & (df['modelpair'] == row[1]['modelpair']) & (df['label'] == 'quality')]
        assert len(winning_model_row) == 1
        winning_model = winning_model_row['winning_model'].to_list()[0]
        winning_model_finegrained = winning_model_row['winning_model_finegrained'].to_list()[0]
        explanations.append([
            explanation_t, explanation, row[1]['dialogue_id'], row[1]['worker_id'], row[1]['modelpair'], winning_model, winning_model_finegrained
        ])
    random.shuffle(explanations)
    explanations = [['explanation_transformed', 'explanation', 'dialogue_id', 'worker_id', 'modelpair', 'winning_model', 'winning_model_finegrained']] + explanations
    csvwriter.writerows(explanations)

def other_textbox_analysis(df: DataFrame):
    other_textbox_results = df[df['label'] == 'quality-aspect-other-textbox']
    counts = Counter(other_textbox_results['value'])
    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    print(json.dumps(sorted_counts, indent=2))

    print()
    print('#'*50)
    print('ANALYSIS')
    print('#'*50)
    influential_aspect_results(df)
    explanation_analysis(df)
    other_textbox_analysis(df)

type_sentence_prefix = {
    "Cause": "I think it is possible the previous dialogue turn was caused by",
    "React$_o$": "The Listener (You) feels",
    "React": "I think the Speaker (Other) feels",
    "Subsequent": "Next, I predict",
    "Motivation": "I think the Speaker (Other) is motivated",
    "Attribute": "I think the Speaker (Other) is",
    "Desire$_o$": "The Listener (You) wants",
    "Constituent": "I think it is possible the previous dialogue turn depends on",
    "Prerequisite": "I think it is possible the previous dialogue turn requires",
    "Desire": "I think the Speaker (Other) wants"
}
prefix_to_type = {v:k for k,v in type_sentence_prefix.items()}

colors = {
    'ConvoSense-E': 'skyblue', 
    'ConvoSense-I': 'lightcoral', 
    'Doctor': 'lightgreen', 
    'GPT': 'mediumpurple'
}

def results_by_cstype(df):
    # add ConvoSense-E selected cs type to dataframe
    response_output_data = data_loader.load_data(
        dir='CommonsenseDialogues/response_outputs/test',
        file='chatgpt-35-curated-compositional-single-0125-0-100.json'
    )
    selected_cs_type_col = []
    for row in df.iterrows():
        row = row[1]
        dialogue_id = row['dialogue_id']
        if 'ConvoSense-E' not in row['modelpair']:
            selected_cs_type_col.append('NONE')
        else:
            dialogue = [d for d in response_output_data if d.dialogue_id == dialogue_id][0]
            selected_cs = list(dialogue.turns[-1].response.values())[0].selected_cs[0]
            for prefix, cstype in prefix_to_type.items():
                if selected_cs.startswith(f'* {prefix}') or selected_cs.startswith(f'{prefix}'):
                    selected_cs_type_col.append(cstype)
                    break
            else:
                print(f'WARNING! Could not identify type of selected cs "{selected_cs}".')
    df['cstype'] = selected_cs_type_col
    print(json.dumps(Counter(selected_cs_type_col), indent=2))
    # groupby include selected cs type
    pairwise_df = df.dropna(axis=0, how='any') # only pairwise questions have no NaNs (winning_model column)
    modelpair_label_dfs = pairwise_df.groupby(by=['modelpair', 'cstype', 'label'])
    modelpair_percs = modelpair_label_dfs.apply(pairwise_perc, 'winning_model')
    modelpair_percs = modelpair_percs.fillna(0)
    print(modelpair_percs)
    # make 1 grouped barplot per metric of ConvoSense-E win against other model with x-axis selected cs type
    winprop = modelpair_percs[['ConvoSense-E']]
    winprop = winprop.drop(index=winprop.index[winprop.index.get_level_values('modelpair') == 'ChatGPT vs Doctor'])
    winprop = winprop.reset_index()
    for label in winprop['label'].unique():
        winprop_for_label = winprop[winprop['label'] == label]
        winprop_for_label['ConvoSense-E'] *= 100
        pivoted = winprop_for_label.pivot(index='cstype', columns='modelpair', values='ConvoSense-E')
        pivoted.columns = [c.replace('ConvoSense-E vs ', '').replace(' vs ConvoSense-E', '') for c in pivoted.columns]
        pivoted = pivoted[['ConvoSense-I', 'Doctor', 'GPT']]
        fig = plt.figure(figsize=(20, 5))  # Adjust the figure size as needed
        ax = fig.add_subplot(1, 1, 1)
        pivoted.plot(
            kind='bar', 
            stacked=False,
            color=[colors[x] for x in pivoted.columns],
            ax=ax
        )
        plt.title('')
        plt.ylabel('')
        plt.xlabel('')
        plt.yticks(fontsize=24)
        plt.xticks(rotation=0, ha='center', fontsize=22)
        plt.legend(fontsize=22, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.25))
        plt.tight_layout()
        figname = f'{label}_by_cstype'
        plt.savefig(f'fig/{figname}.pdf')
        plt.close()

if __name__ == '__main__':
    outputsdir = 'CommonsenseDialogues/response_outputs/test'
    csvdirs = [
        'mturk-csv-0-50-csigsrsingle-doctor',
        'mturk-csv-0-50-csigsrsingle-native',
        'mturk-csv-0-50-csigsrsingle-latentselection',
        'mturk-csv-50-100-csigsrsingle-latentselection',
        'mturk-csv-50-100-csigsrsingle-doctor',
        'mturk-csv-50-100-csigsrsingle-native',
        # 'mturk-csv-0-50-doctor-latentselection',
        # 'mturk-csv-0-50-native-latentselection',
        # 'mturk-csv-50-100-doctor-latentselection',
        # 'mturk-csv-50-100-native-latentselection',
        # 'mturk-csv-0-50-doctor-native',
        # 'mturk-csv-50-100-doctor-native',
        # 'mturk-csv-0-50-native-chaenative',
        # 'mturk-csv-50-100-native-chaenative'
    ]
    df = load_results(outputsdir, csvdirs)

    # view_results(df, label='quality')
    # explanation_analysis(df)

    # print()
    # print('#'*50)
    # print('AGREEMENT')
    # print('#'*50)
    # agreement(df)

    # print()
    # print('#'*50)
    # print('RAW DATA')
    # print('#'*50)
    # pairwise_results(df, finegrained=True, plot_pie=False)

    # print()
    # print('#'*50)
    # print('AFTER MAJORITY VOTE')
    # print('#'*50)
    # agg_df = majority_vote(df)
    # pairwise_results(agg_df)

    print()
    print('#'*50)
    print('SPLIT BY CS TYPE')
    print('#'*50)
    results_by_cstype(df)







        