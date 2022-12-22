from __future__ import annotations

from collections.abc import Iterable
import multiprocessing
from typing import Callable, TypeVar, Tuple
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

import time
import os
import csv

import inspect
from multiprocessing import Pool, Queue, Manager
import functools
import itertools
import random
import operator
import pkg_resources

import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, wordnet
from nltk import FreqDist
from tqdm import tqdm
from textblob import TextBlob
from textblob.en import Spelling
from symspellpy import SymSpell, Verbosity

import warnings
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from nltk.tokenize import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import torch
from scipy.special import softmax
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np

directoryString = ""

sourceFiles = {
    "iTunes": {
            "Replika": '1 - Replika/1 - Replika_iTunes_Output_FIXED.xlsx',
            "Wysa" : '2 - Wysa/2 - Wysa_iTunes_Output.xlsx',
            "Woebot" : '3 - Woebot/3 - Woebot_iTunes_Output.xlsx',
            "Daylio" : '4 - Daylio/4 - Daylio_iTunes_Output.xlsx',
            "MoodMeter" : '5 - Mood Meter/5 - Mood Meter_iTunes_Output.xlsx',
            "iMoodJournal" : '6 - iMood Journal/6 - iMood Journal_iTunes_Output.xlsx',
            "eMoodTracker" : '7 - eMood Tracker/7 - eMood Tracker_iTunes_Output.xlsx',
    },
    "Google": {
            "Replika": '1 - Replika/1 - Replika_Google_Play_Output_FIXED.xlsx',
            "Wysa": '2 - Wysa/2 - Wysa_Google_Play_Output.xlsx',
            "Woebot": '3 - Woebot/3 - Woebot_Google_Play_Output.xlsx',
            "Daylio": '4 - Daylio/4 - Daylio_Google_Play_Output.xlsx',
            "MoodMeter": '5 - Mood Meter/5 - Mood Meter_Google_Play_Output.xlsx',
            "iMoodJournal": '6 - iMood Journal/6 - iMood Journal_Google_Play_Output.xlsx',
            "eMoodTracker": '7 - eMood Tracker/7 - eMood Tracker_Google_Play_Output.xlsx'
    }
}


# Create a dictionary to convert TreeBank POS_Tags to WordNet POS_Tags
penn_treebank_to_wordnet = {
    'CC': None, # conjunction, coordinating
    'CD': None, # numeral, cardinal
    'DT': None, # determiner
    'EX': None, # existential there
    'IN': None, # preposition or conjunction, subordinating
    'JJ': wordnet.ADJ, # adjective or numeral, ordinal
    'JJR': wordnet.ADJ, # adjective, comparative
    'JJS': wordnet.ADJ, # adjective, superlative
    'LS': None, # list item marker
    'MD': None, # modal auxiliary
    'NN': wordnet.NOUN, # noun, common, singular or mass
    'NNP': wordnet.NOUN, # noun, proper, singular
    'NNS': wordnet.NOUN, # noun, common, plural
    'PDT': None, # pre-determiner
    'POS': None, # genitive marker
    'PRP': wordnet.NOUN, # pronoun, personal
    'PRP$': wordnet.NOUN, # pronoun, possessive
    'RB': wordnet.ADV, # adverb
    'RBR': wordnet.ADV, # adverb, comparative
    'RBS': wordnet.ADV, # adverb, superlative
    'RP': wordnet.ADV, # particle
    'TO': None, # "to" as preposition or infinitive marker
    'UH': None, # interjection
    'VB': wordnet.VERB, # verb, base form
    'VBD': wordnet.VERB, # verb, past tense
    'VBG': wordnet.VERB, # verb, present participle or gerund
    'VBN': wordnet.VERB, # verb, past participle
    'VBP': wordnet.VERB, # verb, present tense, not 3rd person singular
    'VBZ': wordnet.VERB, # verb, present tense, 3rd person singular
    'WDT': None, # WH-determiner
    'WP': None, #WH-pronoun
    'WRB': None, #Wh-adverb
}

sym_spell_standard = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
standard_dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
standard_bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
sym_spell_standard.load_dictionary(standard_dictionary_path, term_index=0, count_index=1)
sym_spell_standard.load_bigram_dictionary(standard_bigram_path, term_index=0, count_index=2)

sym_spell_custom = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
custom_dictionary_path = directoryString + 'symspell_dictionary/custom_frequency_dictionary_en_82_765.txt'
sym_spell_custom.load_dictionary(custom_dictionary_path, term_index=0, count_index=1)
custom_bigram_path = standard_dictionary_path
sym_spell_custom.load_bigram_dictionary(custom_bigram_path, term_index=0, count_index=2)

tt = TweetTokenizer()

path = directoryString + "urban_dictionary/en-spelling-with-urban.txt"
assert os.path.isfile(path)
# # Load the new dictionary
customSpelling = Spelling(path=path)

wnl = nltk.WordNetLemmatizer()

stop_words = set(
    ['.', ',', '!', '(', ')', '"', '’', '/', "\\", '-', '?', '...', '..', '&', ':', ';', "\'", '*', '“', '”', '_',
    '%', '=', '‘', '[', ']', '—', '–', '{', '}', '~'])
stop_words.update(stopwords.words('english'))

### Surpress inevitable warnings about missing "Glyphs"
warnings.filterwarnings("ignore", category=UserWarning)

### GLOBAL VARIABLES ###

# Set up Roberta sentiment analysis
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Create a detokenizer
twd = TreebankWordDetokenizer()

sia = SentimentIntensityAnalyzer()

from enum  import Enum
class Spellcheck(Enum):
    NO_SPELLCHECK = 0
    SYMSPELL = 1
    SYMSPELL_COMPOUND = 2
    TEXTBLOB = 3
    SYMSPELL_URBAN = 4
    SYMSPELL_COMPOUND_URBAN = 5
    TEXTBLOB_URBAN = 6


def load(file_name):
    print("Loading", file_name)
    return pd.DataFrame(pd.read_excel(directoryString + "data/" + file_name), columns=['Review'])['Review'].astype(str)

def load_corpus(limit: int|None = None) -> list[Tuple[str, str]]:
    gen = ((F'{store}_{app}_{row}', review)
        for store, apps in sourceFiles.items() 
        for app, file_name in apps.items() 
        for row, review in zip(
            itertools.count(2, 1),
            load(file_name)
        ))
    return list(gen) if limit == None else list(itertools.islice(gen, limit))


def lemmatise(word, pos_tag):
    """Should not be called directly, use `preProcessAll` instead.
    Perform lemmatization on a `word` with a given 'Part of Speech' tag (`pos_tag`).
    Takes the TreeBank POS tag format, and converts it to the required WordNet format.

    Args:
        word (str): The word to be lemmatized
        pos_tag (str): The 'Part of Speech' tag in TreeBank format

    Returns:
        str: The lemma of the `word` for the given `pos_tag`.
    """
    wn_tag = penn_treebank_to_wordnet[pos_tag] if pos_tag in penn_treebank_to_wordnet else None
    return wnl.lemmatize(word,wn_tag) if wn_tag != None else word

def roberta(list_text):

    result = []
    for text in list_text:
        encoded_text = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        model2 = model.to(device)
        output = model2(**encoded_text)
        scores = output[0][0].cpu().detach().numpy()
        scores = softmax(scores)
        # scors_dict = {
        #     'neg' : scores[0],
        #     'neu' : scores[1],
        #     'pos' : scores[2]
        # }
        comp_score = scores[0] + scores[2]
        result.append(comp_score)
    
    return result

    encoded_text = tokenizer(list_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    model2 = model.to(device)
    output = model2(encoded_text['input_ids'])
    output
    scores = torch.nn.functional.softmax(output.cpu().detach(), dim=1)
    return list(scores[:,0] + scores[:,2])

def vader(text):
    return abs(sia.polarity_scores(text)["compound"])

def process_part(review, preprocess: bool, spellcheck: Spellcheck):

    if spellcheck == Spellcheck.SYMSPELL_COMPOUND:
        review = sym_spell_standard.lookup_compound(review, max_edit_distance=2, ignore_non_words=True, ignore_term_with_digits=True)[0].term
    elif spellcheck == Spellcheck.SYMSPELL_COMPOUND_URBAN:
        review = sym_spell_custom.lookup_compound(review, max_edit_distance=2, ignore_non_words=True, ignore_term_with_digits=True)[0].term
    elif spellcheck == Spellcheck.TEXTBLOB:
        review = str(TextBlob(review).correct())

    review = tt.tokenize(review)

    if spellcheck == Spellcheck.SYMSPELL:
        review = [sym_spell_standard.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True, ignore_token=r"(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])(.*)")[0].term for word in review]
    elif spellcheck == Spellcheck.SYMSPELL_URBAN:
        review = [sym_spell_custom.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True, ignore_token=r"(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])(.*)")[0].term for word in review]
    elif spellcheck == Spellcheck.TEXTBLOB_URBAN:
        review = [customSpelling.suggest(word)[0][0] for word in review]

    if preprocess:
        # review = [(lemmatise(t, p), p) for t, p in nltk.pos_tag(review)]
        # review = [(token, pos) for token, pos in review if token.lower() not in stop_words]
        review = [token for token in [lemmatise(t, p) for t, p in nltk.pos_tag(review)] if token.lower() not in stop_words]
    else:
        review = [token for token in review if token.lower() not in stop_words]

    detoken = twd.detokenize(review)

    to_process = list(itertools.chain([('review', detoken)], (('bigram'," ".join(ngram)) for ngram in nltk.ngrams(review, 2)), (('trigram', " ".join(ngram)) for ngram in nltk.ngrams(review, 3))))
    roberta_results = list(zip((x for x, _ in to_process), roberta([y for _, y in to_process])))
    roberta_review = np.average([x for src, x in roberta_results if src == 'review'])
    roberta_bigram = np.average([x for src, x in roberta_results if src == 'bigram'])
    roberta_trigram = np.average([x for src, x in roberta_results if src == 'trigram'])

    vader_review = vader(detoken)
    vader_bigram = np.average([vader(" ".join(bigram)) for bigram in nltk.ngrams(review, 2)])
    vader_trigram = np.average([vader(" ".join(bigram)) for bigram in nltk.ngrams(review, 3)])

    return roberta_review, roberta_bigram, roberta_trigram, vader_review, vader_bigram, vader_trigram
    # res = (('rr', roberta_review), ('rb', roberta_bigram), ('rt', roberta_trigram), ('vr', vader_review), ('vb', vader_bigram), ('vt', vader_trigram))
    # print(preprocess, spellcheck, res)
    # return res

process_params = list(itertools.product([True, False], Spellcheck))

def process(src, review, writer):
    start = time.perf_counter()
    print(start, "Starting", src)

    res = functools.reduce(operator.add, (process_part(review, *args) for args in process_params))
    writer.writerow(list(itertools.chain([src], res)))

    end = time.perf_counter()
    print(end, "Finished", src, "Process time:", F"{end-start:.2f}s")

def main():
    print("### On GPU ###" if device == "cuda:0" else "### ON CPU ###")

    app_start = time.perf_counter()

    corpra = load_corpus(limit = 2)


    f = open('results.csv', 'w', newline='')
    # create the csv writer
    writer = csv.writer(f)

    writer.writerow(list(itertools.chain(["Review"], (F"{m}_{p}" for p, m in itertools.product([F"{'WPP' if preproccess else 'NPP'}_{spellcheck}" for preproccess, spellcheck in process_params], ["Roberta", "Roberta_Bigram", "Roberta_Trigram", "Vader", "Vader_Bigram", "Vader_Trigram"])))))
    f.flush()
    for src, review in corpra:
        process(src, review, writer)
        f.flush()

    f.close()

    app_end = time.perf_counter()
    print("Total time:", F"{app_end-app_start:.2f}s")

if __name__ == "__main__":
    ## Globals

    ## Main
    main()
