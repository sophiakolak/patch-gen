import sys
import os
import json
import pandas as pd
import nltk

def check_args(dir):
    try:
        language = dir.split('-')[1]
        if language == "py":
            print("Running for Quixbugs, python")
        if language == "java":
            print("Running for Quixbugs, java")
    except:
        print("Incorrect file path")
        exit()

def extract_results(paths):
    model_files = {}
    for path in paths:
        model_path = path[0]
        model = path[1]
        files = os.listdir(model_path)
        model_files[model] = files
    return model_files

def extract_bugs_and_patches():
    path = 'py-lines'
    files = os.listdir(path)
    file_bug, file_patch = {}, {}
    for file in files:
        with open(path + '/' + file, 'r') as f:
            data = json.loads(f.read())
            bug = data['buggy_line']
            patch = data['correct_line']
            file_bug[file] = bug
            file_patch[file] = patch
    return file_bug, file_patch

def extract_candidates(base_path, model, files):
    all_candidates = {}
    for file in files:
        rank = 0
        candidates = {}
        with open(base_path+file, 'r') as f:
            for line in f:
                data = json.loads(line)
                text = data['text']
                candidates[rank] = text
                rank+=1
        all_candidates[file] = candidates
    return all_candidates

def calc_bleu_score(candidate, solution):
    candidate_tokenized = nltk.word_tokenize(candidate)
    solution_tokenized = nltk.word_tokenize(solution)
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([solution_tokenized], candidate_tokenized, weights = (0.5, 0.5))
    return BLEUscore

def add_bleu_score_patch(df, model, file_patch):
    files_present = df.columns.tolist()[:-1]
    ranks = df['rank'].tolist()
    df_patch_bs = pd.DataFrame(columns=['model', 'file', 'rank', 'patch_bleu_score'])
    for file in files_present:
        if file not in file_patch:
            continue
        patch = file_patch[file]
        for rank in ranks:
            candidate = df[file][rank-1]
            bleu_score = calc_bleu_score(candidate, patch)
            data = (model, file, rank, bleu_score)
            df_patch_bs.loc[len(df_patch_bs)] = data
    return df_patch_bs

def add_bleu_score_bug(df, model, file_bug):
    files_present = df.columns.tolist()[:-1]
    ranks = df['rank'].tolist()
    df_bug_bs = pd.DataFrame(columns=['model', 'file', 'rank', 'bug_bleu_score'])
    for file in files_present:
        if file not in file_bug:
            continue
        bug = file_bug[file]
        for rank in ranks:
            candidate = df[file][rank-1]
            bleu_score = calc_bleu_score(candidate, bug)
            data = (model, file, rank, bleu_score)
            df_bug_bs.loc[len(df_bug_bs)] = data
    return df_bug_bs
   

def main():
    '''
    Calculate BLEU score for the results from each of the four models
    Takes in the name of the directory where the results are stored

    Usage: python3 bleu_score.py 'quix-py-output' -> Quixbugs-python results
           python3 bleu_score.py 'quix-java-output'-> Quibugs-java results

    '''
    args = sys.argv
    if len(args) != 2:
        print("Usage: python bleu_score.py <reference_file>")
        exit()

    dir =  str(args[1])
    check_args(dir)

    small_path = dir + "/output-small"
    medium_path = dir + "/output-medium"
    large_path = dir + "/output-large"
    codex_path = dir + "/output-codex"

    paths = [(small_path, "small"), (medium_path, "medium"), 
                (large_path,"large"), (codex_path, "codex")]

    model_files = extract_results(paths)

    file_bug, file_patch = extract_bugs_and_patches()

    all_data = []
    for model in model_files:
        base_path = dir+'/output-'+model+'/'
        candidates = extract_candidates(base_path, model, model_files[model])
        df = pd.DataFrame(data=candidates)
        df['rank'] = df.index+1
        all_data.append(df)
    
    models = ['small', 'medium', 'large', 'codex']
    df_all_models = pd.DataFrame(columns=['model', 'file', 'rank', 'patch_bleu_score', 'bug_bleu_score'])
    for model_df in zip(models, all_data):
        model = model_df[0]
        df = model_df[1]
        df_patch_bs = add_bleu_score_patch(df, model, file_patch)
        df_bug_bs = add_bleu_score_bug(df, model, file_bug)
        df_final = pd.merge(df_patch_bs, df_bug_bs, on=['model', 'file', 'rank'])
        df_all_models = pd.concat([df_all_models, df_final])
    print(df_all_models)
                

main()

