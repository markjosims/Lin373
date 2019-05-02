# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:31:15 2019

@author: Mark_S
"""

import pandas as pd
import numpy as np 
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

in_file = 'snli_1.0_dev.csv'

def main():
    df = pd.read_csv(in_file)
    conditions = [(df['gold_label'] == 'entailment'),
                  (df['gold_label'] == 'neutral'),
                  (df['gold_label'] == 'contradiction')]
    choices = [1, 0, -1]
    df['label'] = np.select(conditions, choices)
    df = df.drop(labels = ['sentence1_binary_parse', 'sentence2_binary_parse', 
                           'sentence1_parse', 'sentence2_parse', 'captionID', 
                           'pairID', 'label1', 'label2', 'label3',
                           'label4', 'label5', 'gold_label'], axis=1)
    
    sentences = zip(df['sentence1'], df['sentence2'])
    scores = [get_bleu_score(a, b) for a, b in sentences]
    df['scores'] = scores
    
    cntr = df[df['label'] == -1]
    c_avg = sum(cntr['scores'])/len(cntr)
    print('Average bleu scores for contradiction:', c_avg)
        
    neut = df[df['label'] == 0]
    n_avg = sum(neut['scores'])/len(neut)
    print('Average bleu scores for neutral:', n_avg)

    ent = df[df['label'] == 1]
    e_avg = sum(ent['scores'])/len(ent)
    print('Average bleu scores for entailment:', e_avg)
    
    other = df[df['label'] != 1]
    o_avg = sum(other['scores'])/len(other)
    
    ratio = e_avg/o_avg
    print("Avg entailment score over non-entailment score:", ratio)
    
    twentieth = np.quantile(ent['scores'], .40)
    eightieth = np.quantile(other['scores'], .55)
    print(twentieth, eightieth,'\n')
    
    conditions = [(df['scores'] >= 0.0045),
                  (df['scores'] < 0.0045)]
    choices = [1, -1]
    
    df['predicted_label'] = np.select(conditions, choices)
    labels = list(zip(df['label'], df['predicted_label']))
    true_pos = [label == 1 and predicted_label == 1 for label, predicted_label\
                in labels]
    false_pos = [label != 1 and predicted_label == 1 for label, predicted_label\
                 in labels]
    
    print("Percent of entailments correctly identified",\
          true_pos.count(True)/len(ent))
    
    print("Percent of non-entailments incorrectly labeled (as entailment)",\
          false_pos.count(True)/len(other))
    
    false_pos_neu = [label == 0 and predicted_label == 1 for label, predicted_label\
                     in labels]
    false_pos_cntr = [label == -1 and predicted_label == 1 for label, predicted_label\
                      in labels]
    
    print("Percent of contradictions incorrectly labeled (as entailment)",\
          false_pos_cntr.count(True)/len(cntr))
    print("Percent of neutral statements incorrectly labeled (as entailment)",\
          false_pos_neu.count(True)/len(neut))
    
def get_bleu_score(ref, hyp):
    return sentence_bleu([ref.split()], hyp.split(), weights = [1,0.75,0.3,0.1],\
                          smoothing_function=SmoothingFunction().method1)

main()