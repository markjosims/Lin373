# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:40:19 2019

@author: Mark_S
"""

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import KeyedVectors
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet as wn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.data import find
import numpy as np
import pandas as pd


in_file = 'snli_1.0_dev.csv'
out_file = 'syntax_labels.txt'
ppos = {'n':'NN',
        'v':'VB',
        'adj': 'JJ',
        'adv':'RB'}
stemmer = SnowballStemmer('english')
path_scores = {}
syn_scores = {}

def main():
    df = pd.read_csv(in_file)
    conditions = [(df['gold_label'] == 'entailment'),
                  (df['gold_label'] == 'neutral'),
                  (df['gold_label'] == 'contradiction')]
    choices = [1, 0, -1]
    df['label'] = np.select(conditions, choices)
    df = df.drop(labels = ['sentence1_binary_parse', 'sentence2_binary_parse',
                           'captionID', 'pairID', 'label1', 'label2', 'label3',
                           'label4', 'label5', 'gold_label'], axis=1)
    
    sent_bleu_scores = {pos:[] for pos in ppos.keys()}
    bigram_scores = {pos:[] for pos in ppos.keys()}
    object_sim_scores = []
    pp_object_sim_scores = []
    def_scores = []
    indef_scores = []
    
    stops = [len(df)*((i+1)/4) for i in range(4)]
    for i, row in enumerate(df.iterrows()):
        if i in stops:
            print(i/len(df))
        row = row[1]
        text1, text2 = row['sentence1'], row['sentence2']
        p1, p2 = row['sentence1_parse'], row['sentence2_parse']
        ob_1, ob_2 = get_object(p1), get_object(p2)
        ppob_1, ppob_2 = get_object(p1, head='PP'), get_object(p2, head='PP')
        
        get_sent_bleu_score(p1, p2, sent_bleu_scores)
        #get_sent_bigram_scores(p1, p2, bigram_scores)
        object_sim_scores.append(weighted_avg(get_syn_score(ob_1, ob_2, 'n')))
        pp_object_sim_scores.append(weighted_avg(get_syn_score(ppob_1, ppob_2, 'n')))
        def_scores.append(get_definite_count(p1) - get_definite_count(p2))
        indef_scores.append(get_indefinite_count(p1) - get_indefinite_count(p2))

    df['n_scores'], df['v_scores'] = sent_bleu_scores['n'], sent_bleu_scores['v']
    df['adj_scores'], df['adv_scores'] = sent_bleu_scores['adj'], sent_bleu_scores['adv']
    df['object_scores'] = object_sim_scores
    df['pp_object_scores'] = pp_object_sim_scores
    df['def_scores'] = def_scores
    df['indef_scores'] = indef_scores
    #df['n_bigram_scores'], df['v_bigram_scores'] = bigram_scores['n'], bigram_scores['v']
    #df['adj_bigram_scores'], df['adv_bigram_scores'] = bigram_scores['adj'], bigram_scores['adv']

def get_sent_bigram_scores(parse_1, parse_2, scores):
	bigrams_1, bigrams_2 = get_bigrams(parse_1), get_bigrams(parse_2)
	pos_1, pos_2 = get_word_and_pos(parse_1), get_word_and_pos(parse_2)
	for pos in ppos.keys():
		bg_1 = get_bigrams_same_pos(bigrams_1, pos_1, pos)
		bg_2 = get_bigrams_same_pos(bigrams_2, pos_2, pos)
		scores[pos].append(weighted_avg(score_bigrams(bg_1, bg_2)))

def get_bigrams(sent):
	result = []
	sent = sent.split(sep='(')
	try:
		next_tok = sent[1].split()[1]
	except IndexError:
		return ''
	for i, tok in enumerate(sent[:-1]):
		tok = tok.split()
		if len(tok) < 2:
			continue
		tok = tok[1]
		result.append((tok, next_tok))
		next_tok = sent[i+1]
	return result

def get_bigrams_same_pos(bigrams, sent_pos, pos):
	result = []
	for bg in bigrams:
		if sent_pos[bg[1]] == pos:
			result.append(bg)

	return result

def score_bigrams(bigrams_1, bigrams_2):
	result = []
	for bg in bigrams_1:
		bg_score = []
		for comp_bg in bigrams_2:
			bg_score.append(get_syn_score(bg, comp_bg))
		result.append(max(bg_score))
	return result
    
def get_sent_bleu_score(p1, p2, scores):
    for pos in ppos.keys():
        words_1, words_2 = get_pos(p1, pos, stemmer), get_pos(p2, pos, stemmer)
        scores[pos].append(get_bleu_score(words_1, words_2))
    


def get_object(parse, stemmer=SnowballStemmer('english'), head='VP'):
    result = []
    parse = parse.split(sep="("+head)
    for clause in parse:
        x = get_pos(clause, 'n')
        if x:
            result.append(x[0])
    return result

def get_word_and_pos(parse):
	result = {}
	for lexeme in parse.split(sep='('):
		lexeme = lexeme.split()
		if len(lexeme) < 2:
			continue
		pos_tag = lexeme[0][:2]
		lemma = stemmer.stem(lexeme[1])
		if pos_tag in ppos:
			result[lemma] = pos_tag
		else:
			result[lemma] = '??'
	return result

def get_pos(parse, pos, stemmer=SnowballStemmer('english')):
    result = []
    parse = parse.replace(')','')
    parse = parse.split(sep='(')
    for lexeme in parse:
        lexeme = lexeme.split()
        if len(lexeme) < 2:
            continue
        if lexeme[0][:2] == ppos[pos]:
            lemma = lexeme[1]
            result.append(stemmer.stem(lemma))
            
    return(result)
            
def get_synonyms(word, pos=None):
    result = []
    syns = wn.synsets(word, pos)
    for s in syns:
        for l in s.lemmas():
            lemma_form = l.name()
            if lemma_form not in result:
                result.append(lemma_form)
    
    return result

def get_syns_of_syns(word, pos=None):
    result = get_synonyms(word, pos)
    for x in result[:]:
        these_syns = get_synonyms(x)
        for syn in these_syns:
            if syn not in result:
                result.append(syn)
    
    return result

def get_antonyms(word, pos=None):
    result = []
    ants = wn.synsets(word, pos)
    for a in ants:
        for l in a.lemmas():
            these_ants = l.antonyms()
            for this_a in these_ants:
                lemma_form = this_a.name()
                if lemma_form not in result:
                    result.append(lemma_form)
    
    return result

def get_ants_of_syns(word, pos=None):
    syns = get_synonyms(word, pos)
    result = get_antonyms(word, pos)
    for x in syns:
        these_ants = get_antonyms(x)
        for a in these_ants:
            if a not in result:
                result.append(a)
    
    return result

def get_syns_of_ants(word, pos=None):
    result = get_antonyms(word, pos)
    for x in result[:]:
        these_syns = get_synonyms(x)
        for syn in these_syns:
            if syn not in result:
                result.append(syn)
                
    return result

def get_ants_of_ants(word, pos=None):
    result = []
    ants = get_antonyms(word, pos)
    for x in ants:
        these_ants = get_antonyms(x)
        for a in these_ants:
            if a not in result:
                result.append(a)
                
    return result

def is_synonym(word_1, word_2, pos=None):
    return word_1 in get_synonyms(word_2, pos) \
    or word_2 in get_synonyms(word_1, pos)

def is_antonym(word_1, word_2, pos=None):
    return word_1 in get_antonyms(word_2, pos) \
    or word_2 in get_antonyms(word_1, pos)
    
def is_remote_synonym(word_1, word_2, pos=None):
    return word_1 in get_syns_of_syns(word_2, pos)\
    or word_1 in get_ants_of_ants(word_2, pos)\
    or word_2 in get_syns_of_syns(word_1, pos)\
    or word_2 in get_ants_of_ants(word_1, pos)
    
def is_remote_antonym(word_1, word_2, pos=None):
    return word_1 in get_syns_of_ants(word_2, pos)\
    or word_1 in get_ants_of_syns(word_2, pos)\
    or word_2 in get_syns_of_ants(word_1, pos)\
    or word_2 in get_ants_of_syns(word_1, pos)\

def get_syn_score(sent_1, sent_2, pos):
    result = []
    for word in sent_1:
        w_s = []
        for comp_word in sent_2:
            w_s.append(get_syn_score_word(word, comp_word, pos))
        if w_s:
            result.append(min(w_s))
            
    return result

def get_syn_score_word(word_1, word_2, pos):
    sent_1 = sent_1.split() if type(sent_1) is str else sent_1
    sent_2 = sent_2.split() if type(sent_2) is str else sent_2

    if (word_1, word_2) in syn_scores.keys():
        return syn_scores[(word_1, word_2)]
    if is_synonym(word_1, word_2, pos):
        syn_scores[(word_1, word_2)] = 1
        return 1
    elif is_antonym(word_1, word_2, pos):
        syn_scores[(word_1, word_2)] = -1
        return -1.5
    elif is_remote_synonym(word_1, word_2, pos):
        syn_scores[(word_1, word_2)] = 0.5
        return 0.75
    elif is_remote_antonym(word_1, word_2, pos):
        syn_scores[(word_1, word_2)] = -0.5
        return -1
    else:
        syn_scores[(word_1, word_2)] = 0
        return 0

def get_path_score(sent_1, sent_2, pos):
    sent_1 = sent_1.split() if type(sent_1) is str else sent_1
    sent_2 = sent_2.split() if type(sent_2) is str else sent_2
    
    sent_scores = []
    
    for word in sent_1:
        w_s = []
        synsets = wn.synsets(word, pos)
        if not synsets:
            continue
        for comp_word in sent_2:
            if (word, comp_word) in path_scores.keys():
                w_s.append(path_scores[(word, comp_word)])
                continue
            comp_synsets = wn.synsets(comp_word, pos)
            if not comp_synsets:
                continue
            for ss in synsets:
                for c_ss in comp_synsets:
                    score = ss.wup_similarity(c_ss)
                    if score:
                        path_scores[(word, comp_word)] = score
                        w_s.append(score)
        if w_s:
            w_s = max(w_s)
            sent_scores.append(w_s)
    return sent_scores
    
def get_bleu_score(ref, hyp):
    ref = ref.split() if type(ref) is str else ref
    hyp = hyp.split() if type(hyp) is str else hyp
    return sentence_bleu([ref], hyp, weights = [1,0.75,0.3,0.1],\
                          smoothing_function=SmoothingFunction().method1)
    
def get_definite_count(s):
    s = s.lower().replace(')', '')
    try:
        return s.split().count('the')**2 / s.count('np')
    except ZeroDivisionError:
        return 0

def get_indefinite_count(s):
    s = s.lower().replace(')','')
    try:
        return (s.split().count('a') + s.split().count('an'))**2 / s.count('np')
    except ZeroDivisionError:
        return 0

def avg(l):
    try:
        return sum(l) / len(l)
    except ZeroDivisionError:
        return 0
    
def weighted_avg(l):
    if not l: # avoids some potential errors, namely empty lists
        return 0
    return(avg(l) + avg([x for x in l if x != 0]))
    
def safe_max(l):
    try:
        return max(l)
    except ValueError:
        return 0

def safe_min(l):
    try:
        return min(l)
    except ValueError:
        return 0

def safe_median(l):
    try:
        return l[len(l)//2]
    except IndexError:
        return 0

main()
#woman = wn.synset('woman.n.01')
#sister = wn.synset('sister.n.01')
#print(is_remote_synonym('sibling', 'brother'))