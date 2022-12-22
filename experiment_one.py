# This script conducts experiment one of my research into pre-processing
# of data from the internet for the purposes of Natural Language Processing (NLP).
# This script takes in review data from 7 mental health apps. Data comes from
# both the iTunes store and Google Play store. This script will output data
# from different combinations of processing and spell checking the data, and I
# will then be able to demonstrate which processes are effective, in order to 
# produce good data to perform NLP tasks.
#
# "Pre-Processing" or "No Pre-Procesing" (Tokenisation, POS Tagging, Lemma, etc)
# "No Spell Check", "Standard Spell Check" or "Custom Spell Check"
# "Bi-Gram Production" and "Tri-Gram Production"
# "Bi-Gram Production" and "Tri-Gram Production" Compared to "Urban Dictionary Corpus"
#
# Frequency Distributions of these will then demonstrate the quality of the text being
# output, and which methods produce the best results.
#
# James Kavanagh (jk620)
# University of Kent, School of Computing
# 29th July 2022

""" Imports """

from concurrent.futures import ProcessPoolExecutor as Executor
import multiprocessing
import os
import pickle
import time
from tqdm import tqdm
import pandas as pd
import nltk
from textblob import TextBlob
from textblob.en import Spelling
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import FreqDist
import pkg_resources
from symspellpy import SymSpell, Verbosity

"""Global variables"""
# Set the main directory string for easier loading/writing of data
if multiprocessing.cpu_count() > 10:
    directoryString = ""
else:
    directoryString = "C:/Users/Student/OneDrive - University of Kent/URSS/Experiment 1 - Spell or No Spell/"

# Create dictionary directory of the raw data source files
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

# Set the path of the custom 'TextBlob' dictionary
path = directoryString + "urban_dictionary/en-spelling-with-urban.txt"
assert os.path.isfile(path)
# # Load the new dictionary
customSpelling = Spelling(path=path)

# Create SymSpell resources
sym_spell_standard = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
standard_dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
standard_bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
sym_spell_standard.load_dictionary(standard_dictionary_path, term_index=0, count_index=1)
sym_spell_standard.load_bigram_dictionary(standard_bigram_path, term_index=0, count_index=2)

sym_spell_custom = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
custom_dictionary_path = directoryString + 'symspell_dictionary/custom_frequency_dictionary_en_82_765.txt'
    # I have left the bigram dictionary the same, to prevent bias in n-gram analysis against the urban dictionary later
    # This line can be uncommented if you wish to use the custom bigram dictionary
# custom_bigram_path = directoryString + 'symspell_dictionary/custom_frequency_bigramdictionary_en_243_342.txt'
custom_bigram_path = standard_dictionary_path
sym_spell_custom.load_dictionary(custom_dictionary_path, term_index=0, count_index=1)
sym_spell_custom.load_bigram_dictionary(custom_bigram_path, term_index=0, count_index=2)

# Create an instance of the Tokenizer
tt = TweetTokenizer()

# Create an instance of the Lemmatizer
wnl = nltk.WordNetLemmatizer()

""" FLAGS """
# Set this flag to 'True' if you want to decapitalise the tokens
decapitalise = True # False
# Set this flag to 'True' if you want to remove stop words
removeStopWordsBool = True
# Set this flag to 'True' if you want to perform Lemmatisation
lemmaFlag = True

"""Methods"""

# def parallelReduce(func, lst):
#     """Takes a specific function that requires a single list parameter which is to be
#     reduced to a single value.

#     Args:
#         func (function): The function to be applied over a list.
#         lst (list): The list for the function to be applied over.

#     Returns:
#         Any: Returns a single value that has been reduced
#     """
#     # Takes the provided list and splits into as many lists (chunks) as there are cores of the CPU
#     chnks = chunks(lst, multiprocessing.cpu_count())
#     # Applies the function to each chunk, split over the cores
#     with multiprocessing.Pool() as pool:
#         X = pool.map(func, chnks)
#     # Applies function one more time to flatten the multiple returns from each core into single result
#     Y = func(X)
#     return Y

def mapAllValues(doubleDict, func):
    """Takes a double dictionary and applies a function over it.

    Args:
        doubleDict (dictionary): A dictionary containing another dictionary.
        func (lambda or function): A lambda or a function.

    Returns:
        dictionary: returns a dictionary that the function has been applied over.
    """
    return {storeName: {appName: func(appFile) for appName, appFile in tqdm(storeDict.items())} for storeName, storeDict in doubleDict.items()}

def mapAllValues_NoProgressBar(doubleDict, func):
    """Takes a double dictionary and applies a function over it.

    Args:
        doubleDict (dictionary): A dictionary containing another dictionary.
        func (lambda or function): A lambda or a function.

    Returns:
        dictionary: returns a dictionary that the function has been applied over.
    """
    return {storeName: {appName: func(appFile) for appName, appFile in storeDict.items()} for storeName, storeDict in doubleDict.items()}

# def chunks(lst, n):
#     """Yield successive `n`-sized chunks from a list.

#     Args:
#         lst (list): The list to be split up.
#         n (int): The amount of chunks to split the list into.

#     Yields:
#         list: `n` lists derived from the original list.
#     """ 
#     for i in range(0, len(lst), n):
#         yield lst[i:i + n]

def loadReviews(file_name):
    """Takes a file name of an Excel document and loads it from the 'directoryString', bringing in
    only the 'Review' column.

    Args:
        file_name (string): The file name of the document to be loaded.

    Returns:
        panda.Series (str): Returns a pandas DataFrame as string.
    """
    df = pd.DataFrame(pd.read_excel(directoryString + "review_data/" + file_name), columns=['Review'])
    df.head()
    return df['Review'].astype(str)

def posTag_Lemmatise(tokenisedReview):
    """Should not be called directly, use `preProcessAll` instead.
    Performs 'Part of Speech' tagging and lemmatization of a tokenised review (list of string).

    Args:
        tokenisedReview (list of str): A list of strings (tokens) to be pre-processed.

    Returns:
        list of tuple: A list of tuples (token, pos_tag)
    """
    pos = nltk.pos_tag(tokenisedReview)
    if lemmaFlag:
        return [(lemmatise(t,p),p) for (t, p) in pos]
    else:
        return pos

def preProcessingHelper(store, app, tokenisedReview):
    return store, app, posTag_Lemmatise(tokenisedReview)

# def parallel_Pos_Lemma(store, app, tokenisedReviews):
#     """Performs 'Part of Speech' tagging and lemmatization of a tokenised review (list of string).
#     Performs this on a list of list, i.e., all reviews for a single app.
#     If the global flag for lemmatisation is set to 'True', it will perform both POS_Tag and Lemma. Otherwise,
#     it will only perform POS_Tag.

#     Args:
#         store (str): the store name (iTunes or Google)
#         app (str): the name of the app to be processed
#         tokenisedReviews (list of list of str): A list of of list of strings (tokens) to be pre-processed

#     Returns:
#         list of list of tuple: A list of list of tuples
        
#         `[[(term, POS_Tag), (term, POS_Tag), ...,], [(term, POS_Tag), (term, POS_Tag), ...,], ...]`
#     """
#     toReturn = []
#     for tokenisedReview in tqdm(tokenisedReviews, desc=store + " " + app):
#         pos = nltk.pos_tag(tokenisedReview)
#         if lemmaFlag:
#             toReturn.append([(lemmatise(t,p),p) for (t, p) in pos])
#         else:
#             toReturn.append(pos)
#     return app, toReturn

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

def parallel_tokenise(store, app, review):
    if decapitalise:
        review = review.lower()
    return store, app, tt.tokenize(review)

def parallelMapPerReview(review_func, source, label = "Processing", chunksize = 200):
        # Do the multiprocessing...
    with multiprocessing.Pool(max(round(multiprocessing.cpu_count() * 0.75), 1)) as pool:
            # Create new dictionary as final output
        output = {}
            # bigList is list[(store, app, review)]
        bigList = [(store, app, review) for store, store_reviews in source.items() for app, reviews in store_reviews.items() for review in reviews]
            # For each single review, pass it to `parallel_tokenise`, all in parallel. `chunksize` will dictate how often the progress bar will update
        for store, app, review in pool.starmap(review_func, tqdm(bigList, desc=label), chunksize=chunksize):
                # Rebuild the dictionary structure.
                # We set the root as each `store` with `setDefault(store,{})`.
                # `setDeault(app,[])` assigns each review to a single `app`, rather than having duplicated app names for EACH review.
                # Then we append each review to the appropriate list.
            output.setdefault(store, {}).setdefault(app, []).append(review)
        
            # Convert the dictionary back into a Pandas Series so we can perform functions on the data as required later.
        return mapAllValues_NoProgressBar(output, pd.Series)

def parallelMapGroupReviews(reviews_func, source, label = "Processing", chunksize = 10):
        # Do the multiprocessing...
    with multiprocessing.Pool(max(round(multiprocessing.cpu_count() * 0.75), 1)) as pool:
            # Create new dictionary as final output
        output = {}
            # bigList is list[(store, app, review)]
        bigList = [(store, app, reviews) for store, store_reviews in source.items() for app, reviews in store_reviews.items()]
            # For each single review, pass it to `parallel_tokenise`, all in parallel. `chunksize` will dictate how often the progress bar will update
        for store, app, review in pool.starmap(reviews_func, tqdm(bigList, desc=label), chunksize=chunksize):
                # Rebuild the dictionary structure.
                # We set the root as each `store` with `setDefault(store,{})`.
                # `setDeault(app,[])` assigns each review to a single `app`, rather than having duplicated app names for EACH review.
                # Then we append each review to the appropriate list.
            output.setdefault(store, {}).setdefault(app, []).append(review)
        
            # Convert the dictionary back into a Pandas Series so we can perform functions on the data as required later.
        return mapAllValues_NoProgressBar(output, pd.Series)
    
def removeStopwordsFunction(dictOfReviews, posTagged):
    stop_words = set(['.', ',', '!', '(', ')', '"', '’', '/', "\\", '-', '?', '...', '..', '&', ':', ';', "\'", '*', '“', '”', '_', '%', '=', '‘', '[', ']', '—', '–', '{', '}', '~'])
        # If `removeStopWords`` is set to 'True', then add the stop words from the nltk library to the `stop_words` set.
    if removeStopWordsBool:
        stop_words.update(stopwords.words('english'))
                
        # Stopwords removal
    if posTagged:
        return mapAllValues(dictOfReviews, lambda reviews: reviews.apply(lambda review: [(token,pos) for (token,pos) in review if token.lower() not in stop_words]))
    else:
        return mapAllValues(dictOfReviews, lambda reviews: reviews.apply(lambda review: [token for token in review if token.lower() not in stop_words]))
    
def spellCorrectReview(store, app, listOfTokens):
    return store, app, [sym_spell_standard.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True, ignore_token=r"(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])(.*)")[0].term for word in listOfTokens]
    
    # toReturn = []
    # for word in listOfTokens:
    #     correctedWords = sym_spell_standard.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True, ignore_token=r"(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])(.*)")
    #     print(word, [(s.term, s.distance, s.count) for s in correctedWords])
    #     # for eachWord in correctedWords:
    #     #     print(eachWord)
    #     #     toReturn.append(eachWord.term)
    # return store, app, toReturn

def customSpellCorrectReview(store, app, listOfTokens):
    return store, app, [sym_spell_custom.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True, ignore_token=r"(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])(.*)")[0].term for word in listOfTokens]
    
    # toReturn = []
    # for word in listOfTokens:
    #     correctedWords = sym_spell_custom.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True, ignore_token=r"(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])(.*)")
    #     for eachWord in correctedWords:
    #         toReturn.append(eachWord.term)
    # return store, app, toReturn
    
def spellCorrectCompoundStandard(store, app, reviewString):
    try:
        return store, app, sym_spell_standard.lookup_compound(reviewString, max_edit_distance=2, ignore_non_words=True, ignore_term_with_digits=True)[0].term
    except IndexError:
        return store, app, reviewString

def spellCorrectCompoundCustom(store, app, reviewString):
    try:
        return store, app, sym_spell_custom.lookup_compound(reviewString, max_edit_distance=2, ignore_non_words=True, ignore_term_with_digits=True)[0].term
    except IndexError:
        return store, app, reviewString
    
def standardTextBlobSpell(store, app, reviewString):
    # print('Start', store, app, hex(hash(reviewString) & (256*256-1)))
    tb1 = TextBlob(reviewString)
    tb2 = tb1.correct()
    del tb1
    s = str(tb2)
    del tb2
    # print('  End', store, app, hex(hash(reviewString) & (256*256-1)))
    return store, app, s
    
def customTextBlobSpell(store, app, listOfTokens):
    return store, app, [customSpelling.suggest(word)[0][0] for word in listOfTokens]
    
def main():
    print(r"""
       __   __  ___    __    ___     ___      .__   __.  __      .______                                                           
      |  | |  |/  /   / /   |__ \   / _ \     |  \ |  | |  |     |   _  \                                                          
      |  | |  '  /   / /_      ) | | | | |    |   \|  | |  |     |  |_)  |                                                         
.--.  |  | |    <   | '_ \    / /  | | | |    |  . `  | |  |     |   ___/                                                          
|  `--'  | |  .  \  | (_) |  / /_  | |_| |    |  |\   | |  `----.|  |                                                              
 \______/  |__|\__\  \___/  |____|  \___/     |__| \__| |_______|| _|                                                              
                                                                                                                                   
 __________   ___ .______    _______ .______       __  .___  ___.  _______ .__   __. .___________.     ______   .__   __.  _______ 
|   ____\  \ /  / |   _  \  |   ____||   _  \     |  | |   \/   | |   ____||  \ |  | |           |    /  __  \  |  \ |  | |   ____|
|  |__   \  V  /  |  |_)  | |  |__   |  |_)  |    |  | |  \  /  | |  |__   |   \|  | `---|  |----`   |  |  |  | |   \|  | |  |__   
|   __|   >   <   |   ___/  |   __|  |      /     |  | |  |\/|  | |   __|  |  . `  |     |  |        |  |  |  | |  . `  | |   __|  
|  |____ /  .  \  |  |      |  |____ |  |\  \----.|  | |  |  |  | |  |____ |  |\   |     |  |        |  `--'  | |  |\   | |  |____ 
|_______/__/ \__\ | _|      |_______|| _| `._____||__| |__|  |__| |_______||__| \__|     |__|         \______/  |__| \__| |_______|
    """)
    ##### RAW DATA IMPORT PHASE #####        
        # Load all the raw data from Excel sheets
    print("Loading raw data from CSV's.")
    print("This may take some time.")
    t = time.time() # Start the timer
    # all_reviews_raw = mapAllValues(sourceFiles, loadReviews) # Dictionary
        # Export that as a pickle
    # pickle.dump( all_reviews_raw, open(directoryString + 'pickles/' + 'all_reviews_raw.p', "wb" ) )
        # Enable this line to load the pickle and not have to create it
    all_reviews_raw = pickle.load( open( directoryString + 'pickles/' + 'all_reviews_raw.p', "rb" ) )
        # Check the data was loaded correctly. Hardcoded value for my specific dataset.
    # if (all_reviews_raw.values().__sizeof__() != 24):
    #     print("There was an error loading the data. Please try again.")
    #     exit()
    # else:
    #     print(f"Data was loaded successfuly in {time.time() - t} seconds")
    #     print()
        
        ### COMPOUND SPELL CHECK ###
        ### Must be compelted BEFORE tokenising ###
    print("Compound spell correcting all reviews")
    t = time.time()
        # Perform the spell correction per review
    # compoundSpellCorrected_notTokenised_store_app_reviews = parallelMapPerReview(spellCorrectCompoundStandard, all_reviews_raw, label="Compound Spell Correction")
    # customCompoundSpellCorrected_notTokenised_store_app_reviews = parallelMapPerReview(spellCorrectCompoundCustom, all_reviews_raw, label="Custom Compound Spell Correction")
        # Export results as a pickle
    # pickle.dump( compoundSpellCorrected_notTokenised_store_app_reviews, open( directoryString + 'pickles/' + 'compoundSpellCorrected_notTokenised_store_app_reviews.p', "wb" ) )
    # pickle.dump( customCompoundSpellCorrected_notTokenised_store_app_reviews, open( directoryString + 'pickles/' + 'customCompoundSpellCorrected_notTokenised_store_app_reviews.p', "wb" ) )
        # Enable this line to load the pickle and not have to create it
    compoundSpellCorrected_notTokenised_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'compoundSpellCorrected_notTokenised_store_app_reviews.p', "rb" ) )
    customCompoundSpellCorrected_notTokenised_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'customCompoundSpellCorrected_notTokenised_store_app_reviews.p', "rb" ) )
    print(f"Compound Spelling correction completed in {time.time() - t} seconds")
    print()
    
        ### TextBlob Standard Spell Check ###
        ### Must be completed BEFORE tokenising ###
    print("Standard TextBlob spell correcting all reviews")
    t = time.time()
        # Perform the spell correction per review
    standardTextBlobCorrected_notTokenised_store_app_reviews = parallelMapPerReview(standardTextBlobSpell, all_reviews_raw, label="Standard TextBlob Spell Correction", chunksize=20)
        # Export results as a pickle
    pickle.dump( standardTextBlobCorrected_notTokenised_store_app_reviews, open( directoryString + 'pickles/' + 'standardTextBlobCorrected_notTokenised_store_app_reviews.p', "wb" ) )
        # Enable this line to load the pickle and not have to create it
    # standardTextBlobCorrected_notTokenised_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'standardTextBlobCorrected_notTokenised_store_app_reviews.p', "rb" ) )
    print(f"Standard TextBlob Spelling correction completed in {time.time() - t} seconds")
    print()
    
    ##### PROCESS DATA PHASE #####
        ### TOKENIZE ###
    print("Tokenising all reviews")
    t = time.time() # Start the timer
        # Perform the tokenising via Parallel Function
    tokenised_store_app_reviews = parallelMapPerReview(parallel_tokenise, all_reviews_raw, label="Tokenising Raw Reviews")
    compoundSpellCorrected_Tokenised_store_app_reviews = parallelMapPerReview(parallel_tokenise, compoundSpellCorrected_notTokenised_store_app_reviews, label="Tokenising Compound Corrected")
    customCompoundSpellCorrected_Tokenised_store_app_reviews = parallelMapPerReview(parallel_tokenise, customCompoundSpellCorrected_notTokenised_store_app_reviews, label="Tokenising Custom Compound Corrected")
    standardTextBlobCorrected_Tokenised_store_app_reviews = parallelMapPerReview(parallel_tokenise, standardTextBlobCorrected_notTokenised_store_app_reviews, label="Tokenising Standard TextBlob Corrected")
        # Export that as a pickle
    pickle.dump( tokenised_store_app_reviews, open(directoryString + 'pickles/' + 'tokenised_store_app_reviews.p', "wb" ) )
    pickle.dump( compoundSpellCorrected_Tokenised_store_app_reviews, open(directoryString + 'pickles/' + 'compoundSpellCorrected_Tokenised_store_app_reviews.p', "wb" ) )
    pickle.dump( customCompoundSpellCorrected_Tokenised_store_app_reviews, open(directoryString + 'pickles/' + 'customCompoundSpellCorrected_Tokenised_store_app_reviews.p', "wb" ) )
    pickle.dump( standardTextBlobCorrected_Tokenised_store_app_reviews, open(directoryString + 'pickles/' + 'standardTextBlobCorrected_Tokenised_store_app_reviews.p', "wb" ) )
        # Enable this line to load the pickle and not have to create it
    # tokenised_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'tokenised_store_app_reviews.p', "rb" ) )
    # compoundSpellCorrected_Tokenised_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'compoundSpellCorrected_Tokenised_store_app_reviews.p', "rb" ) )
    # customCompoundSpellCorrected_Tokenised_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'customCompoundSpellCorrected_Tokenised_store_app_reviews.p', "rb" ) )
    # standardTextBlobCorrected_Tokenised_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'standardTextBlobCorrected_Tokenised_store_app_reviews.p', "rb" ) )
    print(f"Tokenising completed in {time.time() - t} seconds")
    print()
    
        ### SPELL CORRECT PHASE ###
    print("Spelling correction for all tokenised reviews")
    t = time.time() 
        # Perform the spell correction per review
    spellCorrected_tokenised_store_app_reviews = parallelMapPerReview(spellCorrectReview, tokenised_store_app_reviews, label="Standard Spell Correcting")
    customSpellCorrected_tokenised_store_app_reviews = parallelMapPerReview(customSpellCorrectReview, tokenised_store_app_reviews, label="Custom Spell Correcting")
    customTextBlobSpellCorrected_tokenised_store_app_reviews = parallelMapPerReview(customTextBlobSpell, tokenised_store_app_reviews, label="Custom TextBlob Spell Correcting")
        # Export the results as a pickle
    pickle.dump( spellCorrected_tokenised_store_app_reviews, open( directoryString + 'pickles/' + 'spellCorrected_tokenised_store_app_reviews.p', "wb" ) )
    pickle.dump( customSpellCorrected_tokenised_store_app_reviews, open( directoryString + 'pickles/' + 'customSpellCorrected_tokenised_store_app_reviews.p', "wb" ) )
    pickle.dump( customTextBlobSpellCorrected_tokenised_store_app_reviews, open( directoryString + 'pickles/' + 'customTextBlobSpellCorrected_tokenised_store_app_reviews.p', "wb" ) )
        # Enable this line to load the pickle and not have to create it
    # spellCorrected_tokenised_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'spellCorrected_tokenised_store_app_reviews.p', "rb" ) )
    # customSpellCorrected_tokenised_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'customSpellCorrected_tokenised_store_app_reviews.p', "rb" ) )
    # customTextBlobSpellCorrected_tokenised_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'customTextBlobSpellCorrected_tokenised_store_app_reviews.p', "rb" ) )
    print(f"Spelling correction completed in {time.time() - t} seconds")
    print()
    
        ### POS TAGGING & LEMMATIZATION ###
    print("'Part of speech tagging' (POS) and Lemmatisation" if lemmaFlag else "'Part of speech tagging' (POS)")
    t = time.time()    
        # Perform the POS tagging and Lemmatization via Parallel Function
    lemmad_store_app_reviews = parallelMapPerReview(preProcessingHelper, tokenised_store_app_reviews, label="Lemmatising")
    spellCorrected_lemmad_store_app_reviews = parallelMapPerReview(preProcessingHelper, spellCorrected_tokenised_store_app_reviews, label="Lemmatising Spell Corrected")
    customSpellCorrected_lemmad_store_app_reviews = parallelMapPerReview(preProcessingHelper, customSpellCorrected_tokenised_store_app_reviews, label="Lemmatising Custom Spell Corrected")
    compoundSpellCorrected_lemmad_store_app_reviews = parallelMapPerReview(preProcessingHelper, compoundSpellCorrected_Tokenised_store_app_reviews, label="Lemmatising Compound Spell Corrected")
    compoundCustomSpellCorrected_lemmad_store_app_reviews = parallelMapPerReview(preProcessingHelper, customCompoundSpellCorrected_Tokenised_store_app_reviews, label="Lemmatising Custom Compound Spell Corrected")
    standardTextBlobspellCorrected_lemmad_store_app_reviews = parallelMapPerReview(preProcessingHelper, standardTextBlobCorrected_Tokenised_store_app_reviews, label="Lemmatising Standard TextBlob Spell Corrected")
    customTextBlobSpellCorrected_lemmad_store_app_reviews = parallelMapPerReview(preProcessingHelper, customTextBlobSpellCorrected_tokenised_store_app_reviews, label="Lemmatising Custom TextBlob Spell Corrected")
        # Export the results as a pickle
    if lemmaFlag:
        pickle.dump( lemmad_store_app_reviews, open(directoryString + 'pickles/' + 'lemmad_store_app_reviews.p', "wb" ) )
        pickle.dump( spellCorrected_lemmad_store_app_reviews, open(directoryString + 'pickles/' + 'spellCorrected_lemmad_store_app_reviews.p', "wb" ) )
        pickle.dump( customSpellCorrected_lemmad_store_app_reviews, open(directoryString + 'pickles/' + 'customSpellCorrected_lemmad_store_app_reviews.p', "wb" ) )
        pickle.dump( compoundSpellCorrected_lemmad_store_app_reviews, open(directoryString + 'pickles/' + 'compoundSpellCorrected_lemmad_store_app_reviews.p', "wb" ) )
        pickle.dump( compoundCustomSpellCorrected_lemmad_store_app_reviews, open(directoryString + 'pickles/' + 'compoundCustomSpellCorrected_lemmad_store_app_reviews.p', "wb" ) )
        pickle.dump( standardTextBlobspellCorrected_lemmad_store_app_reviews, open(directoryString + 'pickles/' + 'standardTextBlobspellCorrected_lemmad_store_app_reviews.p', "wb" ) )
        pickle.dump( customTextBlobSpellCorrected_lemmad_store_app_reviews, open(directoryString + 'pickles/' + 'customTextBlobSpellCorrected_lemmad_store_app_reviews.p', "wb" ) )
    else:
        pickle.dump( lemmad_store_app_reviews, open(directoryString + 'pickles/' + 'POS_Only_store_app_reviews.p', "wb" ) )
        pickle.dump( spellCorrected_lemmad_store_app_reviews, open(directoryString + 'pickles/' + 'spellCorrected_POS_Only_store_app_reviews.p', "wb" ) )
        pickle.dump( customSpellCorrected_lemmad_store_app_reviews, open(directoryString + 'pickles/' + 'customSpellCorrected_POS_Only_store_app_reviews.p', "wb" ) )
        pickle.dump( compoundSpellCorrected_lemmad_store_app_reviews, open(directoryString + 'pickles/' + 'compoundSpellCorrected_POS_Only_store_app_reviews.p', "wb" ) )
        pickle.dump( compoundCustomSpellCorrected_lemmad_store_app_reviews, open(directoryString + 'pickles/' + 'compoundCustomSpellCorrected_POS_Only_store_app_reviews.p', "wb" ) )
        pickle.dump( standardTextBlobspellCorrected_lemmad_store_app_reviews, open(directoryString + 'pickles/' + 'standardTextBlobspellCorrected_POS_Only_store_app_reviews.p', "wb" ) )
        pickle.dump( customTextBlobSpellCorrected_lemmad_store_app_reviews, open(directoryString + 'pickles/' + 'customTextBlobSpellCorrected_POS_Only_store_app_reviews.p', "wb" ) )
        # Enable this line to load the pickle and not have to create it
    # lemmad_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'lemmad_store_app_reviews.p', "rb" ) )
    # spellCorrected_lemmad_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'spellCorrected_lemmad_store_app_reviews.p', "rb" ) )
    # customSpellCorrected_lemmad_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'customSpellCorrected_lemmad_store_app_reviews.p', "rb" ) )
    # compoundSpellCorrected_lemmad_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'compoundSpellCorrected_lemmad_store_app_reviews.p', "rb" ) )
    # compoundCustomSpellCorrected_lemmad_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'compoundCustomSpellCorrected_lemmad_store_app_reviews.p', "rb" ) )
    # standardTextBlobspellCorrected_lemmad_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'standardTextBlobspellCorrected_lemmad_store_app_reviews.p', "rb" ) )
    # customTextBlobSpellCorrected_lemmad_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'customTextBlobSpellCorrected_lemmad_store_app_reviews.p', "rb" ) )
    ### OR ###
    # lemmad_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'POS_Only_store_app_reviews.p', "rb" ) )
    # spellCorrected_lemmad_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'spellCorrected_POS_Only_store_app_reviews.p', "rb" ) )
    # customSpellCorrected_lemmad_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'customSpellCorrected_POS_Only_store_app_reviews.p', "rb" ) )
    # compoundSpellCorrected_lemmad_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'compoundSpellCorrected_POS_Only_store_app_reviews.p', "rb" ) )
    # compoundCustomSpellCorrected_lemmad_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'compoundCustomSpellCorrected_POS_Only_store_app_reviews.p', "rb" ) )
    # standardTextBlobspellCorrected_lemmad_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'standardTextBlobspellCorrected_POS_Only_store_app_reviews.p', "rb" ) )
    # customTextBlobSpellCorrected_lemmad_store_app_reviews = pickle.load( open( directoryString + 'pickles/' + 'customTextBlobSpellCorrected_POS_Only_store_app_reviews.p', "rb" ) )
    
    print(f"POS tagging & Lemmatisation completed in {time.time() - t} seconds")
    print()
    
        ### STOPWORD REMOVAL ###
    print("Stopword removal and punctuation cleaning" if removeStopWordsBool else "Removing singular punctuation and noise")
    t = time.time()
        # Create a set of all the punctuation characters that should be removed if they appear on their own. They will remain if they're part of another token though.
    stop_words = set(['.', ',', '!', '(', ')', '"', '’', '/', "\\", '-', '?', '...', '..', '&', ':', ';', "\'", '*', '“', '”', '_', '%', '=', '‘', '[', ']', '—', '–', '{', '}', '~'])
        # If `removeStopWords`` is set to 'True', then add the stop words from the nltk library to the `stop_words` set.
    if removeStopWordsBool:
        stop_words.update(stopwords.words('english'))
                
        # Stopwords removal
    all_reviews_preProcessed = removeStopwordsFunction(lemmad_store_app_reviews, posTagged=True)
    all_reviews_preProcessed_spellCorrected = removeStopwordsFunction(spellCorrected_lemmad_store_app_reviews, posTagged=True)
    all_reviews_preProcessed_customSpellCorrected = removeStopwordsFunction(customSpellCorrected_lemmad_store_app_reviews, posTagged=True)
    all_reviews_preProcessed_compoundSpellCorrected = removeStopwordsFunction(compoundSpellCorrected_lemmad_store_app_reviews, posTagged=True)
    all_reviews_preProcessed_compoundCustomSpellCorrected = removeStopwordsFunction(compoundCustomSpellCorrected_lemmad_store_app_reviews, posTagged=True)
    all_reviews_preProcessed_standardTextBlobSpellCorrected = removeStopwordsFunction(standardTextBlobspellCorrected_lemmad_store_app_reviews, posTagged=True)
    all_reviews_preProcessed_customTextBlobSpellCorrected = removeStopwordsFunction(customTextBlobSpellCorrected_lemmad_store_app_reviews, posTagged=True)
        # Export the results as a pickle
    if removeStopWordsBool:
        pickle.dump( all_reviews_preProcessed, open(directoryString + 'pickles/' + 'all_reviews_preProcessed.p', "wb" ) )
        pickle.dump( all_reviews_preProcessed_spellCorrected, open(directoryString + 'pickles/' + 'all_reviews_preProcessed_spellCorrected.p', "wb" ) )
        pickle.dump( all_reviews_preProcessed_customSpellCorrected, open(directoryString + 'pickles/' + 'all_reviews_preProcessed_customSpellCorrected.p', "wb" ) )
        pickle.dump( all_reviews_preProcessed_compoundSpellCorrected, open(directoryString + 'pickles/' + 'all_reviews_preProcessed_compoundSpellCorrected.p', "wb" ) )
        pickle.dump( all_reviews_preProcessed_compoundCustomSpellCorrected, open(directoryString + 'pickles/' + 'all_reviews_preProcessed_compoundCustomSpellCorrected.p', "wb" ) )
        pickle.dump( all_reviews_preProcessed_standardTextBlobSpellCorrected, open(directoryString + 'pickles/' + 'all_reviews_preProcessed_standardTextBlobSpellCorrected.p', "wb" ) )
        pickle.dump( all_reviews_preProcessed_customTextBlobSpellCorrected, open(directoryString + 'pickles/' + 'all_reviews_preProcessed_customTextBlobSpellCorrected.p', "wb" ) )
    else:
        pickle.dump( all_reviews_preProcessed, open(directoryString + 'pickles/' + 'all_reviews_preProcessed_with_stopwords.p', "wb" ) )
        pickle.dump( all_reviews_preProcessed_spellCorrected, open(directoryString + 'pickles/' + 'all_reviews_preProcessed_spellCorrected_with_stopwords.p', "wb" ) )
        pickle.dump( all_reviews_preProcessed_customSpellCorrected, open(directoryString + 'pickles/' + 'all_reviews_preProcessed_customSpellCorrected_with_stopwords.p', "wb" ) )
        pickle.dump( all_reviews_preProcessed_compoundSpellCorrected, open(directoryString + 'pickles/' + 'all_reviews_preProcessed_compoundSpellCorrected_with_stopwords.p', "wb" ) )
        pickle.dump( all_reviews_preProcessed_compoundCustomSpellCorrected, open(directoryString + 'pickles/' + 'all_reviews_preProcessed_compoundCustomSpellCorrected_with_stopwords.p', "wb" ) )
        pickle.dump( all_reviews_preProcessed_standardTextBlobSpellCorrected, open(directoryString + 'pickles/' + 'all_reviews_preProcessed_standardTextBlobSpellCorrected_with_stopwords.p', "wb" ) )
        pickle.dump( all_reviews_preProcessed_customTextBlobSpellCorrected, open(directoryString + 'pickles/' + 'all_reviews_preProcessed_customTextBlobSpellCorrected_with_stopwords.p', "wb" ) )
        # Enable this line to load the pickle and not have to create it
    # all_reviews_preProcessed = pickle.load( open( directoryString + 'pickles/' + 'all_reviews_preProcessed.p', "rb" ) )
    # all_reviews_preProcessed_spellCorrected = pickle.load( open( directoryString + 'pickles/' + 'all_reviews_preProcessed_spellCorrected.p', "rb" ) )
    # all_reviews_preProcessed_customSpellCorrected = pickle.load( open( directoryString + 'pickles/' + 'all_reviews_preProcessed_customSpellCorrected.p', "rb" ) )
    # all_reviews_preProcessed_compoundSpellCorrected = pickle.load( open( directoryString + 'pickles/' + 'all_reviews_preProcessed_compoundSpellCorrected.p', "rb" ) )
    # all_reviews_preProcessed_compoundCustomSpellCorrected = pickle.load( open( directoryString + 'pickles/' + 'all_reviews_preProcessed_compoundCustomSpellCorrected.p', "rb" ) )
    # all_reviews_preProcessed_standardTextBlobSpellCorrected = pickle.load( open( directoryString + 'pickles/' + 'all_reviews_preProcessed_standardTextBlobSpellCorrected.p', "rb" ) )
    # all_reviews_preProcessed_customTextBlobSpellCorrected = pickle.load( open( directoryString + 'pickles/' + 'all_reviews_preProcessed_customTextBlobSpellCorrected.p', "rb" ) )
    ### OR ###
    # all_reviews_preProcessed = pickle.load( open( directoryString + 'pickles/' + 'all_reviews_preProcessed_with_stopwords.p', "rb" ) )
    # all_reviews_preProcessed_spellCorrected = pickle.load( open( directoryString + 'pickles/' + 'all_reviews_preProcessed_spellCorrected_with_stopwords.p', "rb" ) )
    # all_reviews_preProcessed_customSpellCorrected = pickle.load( open( directoryString + 'pickles/' + 'all_reviews_preProcessed_customSpellCorrected_with_stopwords.p', "rb" ) )
    # all_reviews_preProcessed_compoundSpellCorrected = pickle.load( open( directoryString + 'pickles/' + 'all_reviews_preProcessed_compoundSpellCorrected_with_stopwords.p', "rb" ) )
    # all_reviews_preProcessed_compoundCustomSpellCorrected = pickle.load( open( directoryString + 'pickles/' + 'all_reviews_preProcessed_compoundCustomSpellCorrected_with_stopwords.p', "rb" ) )
    # all_reviews_preProcessed_standardTextBlobSpellCorrected = pickle.load( open( directoryString + 'pickles/' + 'all_reviews_preProcessed_standardTextBlobSpellCorrected_with_stopwords.p', "rb" ) )
    # all_reviews_preProcessed_customTextBlobSpellCorrected = pickle.load( open( directoryString + 'pickles/' + 'all_reviews_preProcessed_customTextBlobSpellCorrected_with_stopwords.p', "rb" ) )
    print(f"Stopword removal and punctuation cleaning completed in {time.time() - t} seconds" if removeStopWordsBool else f"Singular punctuation and noise removal completed in {time.time() - t} seconds")
    print()
    
    #####  CREATE REQUIRED OUTPUTS  #####
    #####           AND             #####
    #####       CLEAR MEMORY        #####
    print("Splitting data out into required lists")
    t = time.time()
        ### No Pre Processing (only tokenised and stopword removed) ###
        ### NO SPELL CHECKING ###
        # Take the raw tokenised reviews and remove the stopwords. Leaving us with a dataset that hasn't been pre-processed
    raw_tokenised_stopwordRemoved = removeStopwordsFunction(tokenised_store_app_reviews, posTagged=False)
        # Create the lists
    NPP_all_reviews =    [review for app_dict in raw_tokenised_stopwordRemoved.values() for reviews in app_dict.values() for review in reviews]
    NPP_apple =          [review for reviews in raw_tokenised_stopwordRemoved['iTunes'].values() for review in reviews]
    NPP_Google =         [review for reviews in raw_tokenised_stopwordRemoved['Google'].values() for review in reviews]
    NPP_Replika =        [review for app_dict in raw_tokenised_stopwordRemoved.values() for review in app_dict['Replika']]
    NPP_Wysa =           [review for app_dict in raw_tokenised_stopwordRemoved.values() for review in app_dict['Wysa']]
    NPP_Woebot =         [review for app_dict in raw_tokenised_stopwordRemoved.values() for review in app_dict['Woebot']]
    NPP_Daylio =         [review for app_dict in raw_tokenised_stopwordRemoved.values() for review in app_dict['Daylio']]
    NPP_MoodMeter =      [review for app_dict in raw_tokenised_stopwordRemoved.values() for review in app_dict['MoodMeter']]
    NPP_iMoodJournal =   [review for app_dict in raw_tokenised_stopwordRemoved.values() for review in app_dict['iMoodJournal']]
    NPP_eMoodTracker =   [review for app_dict in raw_tokenised_stopwordRemoved.values() for review in app_dict['eMoodTracker']]    
            
        ### No Pre Processing (only tokenised and stopword removed) ###
        ### WITH SPELL CHECKING ###
        # Take the standard spell checked tokenised reviews and remove the stopwords. Leaving us with a dataset that hasn't been pre-processed
    spellCorrected_tokenised_stopwordRemoved = removeStopwordsFunction(spellCorrected_tokenised_store_app_reviews, posTagged=False)
        # Create the lists
    NPP_all_reviews_spellCorrected =    [review for app_dict in spellCorrected_tokenised_stopwordRemoved.values() for reviews in app_dict.values() for review in reviews]
    NPP_apple_spellCorrected =          [review for reviews in spellCorrected_tokenised_stopwordRemoved['iTunes'].values() for review in reviews]
    NPP_Google_spellCorrected =         [review for reviews in spellCorrected_tokenised_stopwordRemoved['Google'].values() for review in reviews]
    NPP_Replika_spellCorrected =        [review for app_dict in spellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Replika']]
    NPP_Wysa_spellCorrected =           [review for app_dict in spellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Wysa']]
    NPP_Woebot_spellCorrected =         [review for app_dict in spellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Woebot']]
    NPP_Daylio_spellCorrected =         [review for app_dict in spellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Daylio']]
    NPP_MoodMeter_spellCorrected =      [review for app_dict in spellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['MoodMeter']]
    NPP_iMoodJournal_spellCorrected =   [review for app_dict in spellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['iMoodJournal']]
    NPP_eMoodTracker_spellCorrected =   [review for app_dict in spellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['eMoodTracker']]
    
        ### No Pre Processing (only tokenised and stopword removed) ###
        ### WITH COMPOUND SPELL CHECKING ###
        # Take the standard compound spell checked tokenised reviews and remove the stopwords. Leaving us with a dataset that hasn't been pre-processed
    compoundSpellCorrected_tokenised_stopwordRemoved = removeStopwordsFunction(compoundSpellCorrected_Tokenised_store_app_reviews, posTagged=False)
        # Create the lists
    NPP_all_reviews_compoundSpellCorrected =    [review for app_dict in compoundSpellCorrected_tokenised_stopwordRemoved.values() for reviews in app_dict.values() for review in reviews]
    NPP_apple_compoundSpellCorrected =          [review for reviews in compoundSpellCorrected_tokenised_stopwordRemoved['iTunes'].values() for review in reviews]
    NPP_Google_compoundSpellCorrected =         [review for reviews in compoundSpellCorrected_tokenised_stopwordRemoved['Google'].values() for review in reviews]
    NPP_Replika_compoundSpellCorrected =        [review for app_dict in compoundSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Replika']]
    NPP_Wysa_compoundSpellCorrected =           [review for app_dict in compoundSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Wysa']]
    NPP_Woebot_compoundSpellCorrected =         [review for app_dict in compoundSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Woebot']]
    NPP_Daylio_compoundSpellCorrected =         [review for app_dict in compoundSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Daylio']]
    NPP_MoodMeter_compoundSpellCorrected =      [review for app_dict in compoundSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['MoodMeter']]
    NPP_iMoodJournal_compoundSpellCorrected =   [review for app_dict in compoundSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['iMoodJournal']]
    NPP_eMoodTracker_compoundSpellCorrected =   [review for app_dict in compoundSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['eMoodTracker']]
    
        ### No Pre Processing (only tokenised and stopword removed) ###
        ### WITH STANDARD TEXTBLOB SPELL CHECKING ###
        # Take the standard TextBlob spell checked tokenised reviews and remove the stopwords. Leaving us with a dataset that hasn't been pre-processed
    standardTextBlobSpellCorrected_tokenised_stopwordRemoved = removeStopwordsFunction(standardTextBlobCorrected_Tokenised_store_app_reviews, posTagged=False)
        # Create the lists
    NPP_all_reviews_standardTextBlobSpellCorrected =    [review for app_dict in standardTextBlobSpellCorrected_tokenised_stopwordRemoved.values() for reviews in app_dict.values() for review in reviews]
    NPP_apple_standardTextBlobSpellCorrected =          [review for reviews in standardTextBlobSpellCorrected_tokenised_stopwordRemoved['iTunes'].values() for review in reviews]
    NPP_Google_standardTextBlobSpellCorrected =         [review for reviews in standardTextBlobSpellCorrected_tokenised_stopwordRemoved['Google'].values() for review in reviews]
    NPP_Replika_standardTextBlobSpellCorrected =        [review for app_dict in standardTextBlobSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Replika']]
    NPP_Wysa_standardTextBlobSpellCorrected =           [review for app_dict in standardTextBlobSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Wysa']]
    NPP_Woebot_standardTextBlobSpellCorrected =         [review for app_dict in standardTextBlobSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Woebot']]
    NPP_Daylio_standardTextBlobSpellCorrected =         [review for app_dict in standardTextBlobSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Daylio']]
    NPP_MoodMeter_standardTextBlobSpellCorrected =      [review for app_dict in standardTextBlobSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['MoodMeter']]
    NPP_iMoodJournal_standardTextBlobSpellCorrected =   [review for app_dict in standardTextBlobSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['iMoodJournal']]
    NPP_eMoodTracker_standardTextBlobSpellCorrected =   [review for app_dict in standardTextBlobSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['eMoodTracker']]
    
        ### No Pre Processing (only tokenised and stopword removed) ###
        ### WITH CUSTOM SPELL CHECKING ###
        # Take the custom spell checked tokenised reviews and remove the stopwords. Leaving us with a dataset that hasn't been pre-processed
    customSpellCorrected_tokenised_stopwordRemoved = removeStopwordsFunction(customSpellCorrected_tokenised_store_app_reviews, posTagged=False)
        # Create the lists
    NPP_all_reviews_customSpellCorrected =    [review for app_dict in customSpellCorrected_tokenised_stopwordRemoved.values() for reviews in app_dict.values() for review in reviews]
    NPP_apple_customSpellCorrected =          [review for reviews in customSpellCorrected_tokenised_stopwordRemoved['iTunes'].values() for review in reviews]
    NPP_Google_customSpellCorrected =         [review for reviews in customSpellCorrected_tokenised_stopwordRemoved['Google'].values() for review in reviews]
    NPP_Replika_customSpellCorrected =        [review for app_dict in customSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Replika']]
    NPP_Wysa_customSpellCorrected =           [review for app_dict in customSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Wysa']]
    NPP_Woebot_customSpellCorrected =         [review for app_dict in customSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Woebot']]
    NPP_Daylio_customSpellCorrected =         [review for app_dict in customSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Daylio']]
    NPP_MoodMeter_customSpellCorrected =      [review for app_dict in customSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['MoodMeter']]
    NPP_iMoodJournal_customSpellCorrected =   [review for app_dict in customSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['iMoodJournal']]
    NPP_eMoodTracker_customSpellCorrected =   [review for app_dict in customSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['eMoodTracker']]
    
        ### No Pre Processing (only tokenised and stopword removed) ###
        ### WITH COMPOUND CUSTOM SPELL CHECKING ###
        # Take the custom compound spell checked tokenised reviews and remove the stopwords. Leaving us with a dataset that hasn't been pre-processed
    compoundCustomSpellCorrected_tokenised_stopwordRemoved = removeStopwordsFunction(customCompoundSpellCorrected_Tokenised_store_app_reviews, posTagged=False)
        # Create the lists
    NPP_all_reviews_compoundCustomSpellCorrected =    [review for app_dict in compoundCustomSpellCorrected_tokenised_stopwordRemoved.values() for reviews in app_dict.values() for review in reviews]
    NPP_apple_compoundCustomSpellCorrected =          [review for reviews in compoundCustomSpellCorrected_tokenised_stopwordRemoved['iTunes'].values() for review in reviews]
    NPP_Google_compoundCustomSpellCorrected =         [review for reviews in compoundCustomSpellCorrected_tokenised_stopwordRemoved['Google'].values() for review in reviews]
    NPP_Replika_compoundCustomSpellCorrected =        [review for app_dict in compoundCustomSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Replika']]
    NPP_Wysa_compoundCustomSpellCorrected =           [review for app_dict in compoundCustomSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Wysa']]
    NPP_Woebot_compoundCustomSpellCorrected =         [review for app_dict in compoundCustomSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Woebot']]
    NPP_Daylio_compoundCustomSpellCorrected =         [review for app_dict in compoundCustomSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Daylio']]
    NPP_MoodMeter_compoundCustomSpellCorrected =      [review for app_dict in compoundCustomSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['MoodMeter']]
    NPP_iMoodJournal_compoundCustomSpellCorrected =   [review for app_dict in compoundCustomSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['iMoodJournal']]
    NPP_eMoodTracker_compoundCustomSpellCorrected =   [review for app_dict in compoundCustomSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['eMoodTracker']]
    
        ### No Pre Processing (only tokenised and stopword removed) ###
        ### WITH CUSTOM TEXTBLOB SPELL CHECKING ###
        # Take the custom TextBlob spell checked tokenised reviews and remove the stopwords. Leaving us with a dataset that hasn't been pre-processed
    customTextBlobSpellCorrected_tokenised_stopwordRemoved = removeStopwordsFunction(customTextBlobSpellCorrected_tokenised_store_app_reviews, posTagged=False)
        # Create the lists
    NPP_all_reviews_customTextBlobSpellCorrected =    [review for app_dict in customTextBlobSpellCorrected_tokenised_stopwordRemoved.values() for reviews in app_dict.values() for review in reviews]
    NPP_apple_customTextBlobSpellCorrected =          [review for reviews in customTextBlobSpellCorrected_tokenised_stopwordRemoved['iTunes'].values() for review in reviews]
    NPP_Google_customTextBlobSpellCorrected =         [review for reviews in customTextBlobSpellCorrected_tokenised_stopwordRemoved['Google'].values() for review in reviews]
    NPP_Replika_customTextBlobSpellCorrected =        [review for app_dict in customTextBlobSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Replika']]
    NPP_Wysa_custoTextBlobmSpellCorrected =           [review for app_dict in customTextBlobSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Wysa']]
    NPP_Woebot_customTextBlobSpellCorrected =         [review for app_dict in customTextBlobSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Woebot']]
    NPP_Daylio_customTextBlobSpellCorrected =         [review for app_dict in customTextBlobSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['Daylio']]
    NPP_MoodMeter_customTextBlobSpellCorrected =      [review for app_dict in customTextBlobSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['MoodMeter']]
    NPP_iMoodJournal_customTextBlobSpellCorrected =   [review for app_dict in customTextBlobSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['iMoodJournal']]
    NPP_eMoodTracker_customTextBlobSpellCorrected =   [review for app_dict in customTextBlobSpellCorrected_tokenised_stopwordRemoved.values() for review in app_dict['eMoodTracker']]
    
        ### With Pre Processing (inlcudes lemmatization and POS Tagging) ###
        ### NO SPELL CHECKING ###
        # Create the lists
    WPP_all_reviews =    [review for app_dict in all_reviews_preProcessed.values() for reviews in app_dict.values() for review in reviews]
    WPP_apple =          [review for reviews in all_reviews_preProcessed['iTunes'].values() for review in reviews]
    WPP_Google =         [review for reviews in all_reviews_preProcessed['Google'].values() for review in reviews]
    WPP_Replika =        [review for app_dict in all_reviews_preProcessed.values() for review in app_dict['Replika']]
    WPP_Wysa =           [review for app_dict in all_reviews_preProcessed.values() for review in app_dict['Wysa']]
    WPP_Woebot =         [review for app_dict in all_reviews_preProcessed.values() for review in app_dict['Woebot']]
    WPP_Daylio =         [review for app_dict in all_reviews_preProcessed.values() for review in app_dict['Daylio']]
    WPP_MoodMeter =      [review for app_dict in all_reviews_preProcessed.values() for review in app_dict['MoodMeter']]
    WPP_iMoodJournal =   [review for app_dict in all_reviews_preProcessed.values() for review in app_dict['iMoodJournal']]
    WPP_eMoodTracker =   [review for app_dict in all_reviews_preProcessed.values() for review in app_dict['eMoodTracker']]
    
        ### With Pre Processing (inlcudes lemmatization and POS Tagging) ###
        ### WITH SPELL CHECKING ###
    WPP_all_reviews_spellCorrected =    [review for app_dict in all_reviews_preProcessed_spellCorrected.values() for reviews in app_dict.values() for review in reviews]
    WPP_apple_spellCorrected =          [review for reviews in all_reviews_preProcessed_spellCorrected['iTunes'].values() for review in reviews]
    WPP_Google_spellCorrected =         [review for reviews in all_reviews_preProcessed_spellCorrected['Google'].values() for review in reviews]
    WPP_Replika_spellCorrected =        [review for app_dict in all_reviews_preProcessed_spellCorrected.values() for review in app_dict['Replika']]
    WPP_Wysa_spellCorrected =           [review for app_dict in all_reviews_preProcessed_spellCorrected.values() for review in app_dict['Wysa']]
    WPP_Woebot_spellCorrected =         [review for app_dict in all_reviews_preProcessed_spellCorrected.values() for review in app_dict['Woebot']]
    WPP_Daylio_spellCorrected =         [review for app_dict in all_reviews_preProcessed_spellCorrected.values() for review in app_dict['Daylio']]
    WPP_MoodMeter_spellCorrected =      [review for app_dict in all_reviews_preProcessed_spellCorrected.values() for review in app_dict['MoodMeter']]
    WPP_iMoodJournal_spellCorrected =   [review for app_dict in all_reviews_preProcessed_spellCorrected.values() for review in app_dict['iMoodJournal']]
    WPP_eMoodTracker_spellCorrected =   [review for app_dict in all_reviews_preProcessed_spellCorrected.values() for review in app_dict['eMoodTracker']]
    
        ### With Pre Processing (inlcudes lemmatization and POS Tagging) ###
        ### WITH COMPOUND SPELL CHECKING ###
    WPP_all_reviews_compoundSpellCorrected =    [review for app_dict in all_reviews_preProcessed_compoundSpellCorrected.values() for reviews in app_dict.values() for review in reviews]
    WPP_apple_compoundSpellCorrected =          [review for reviews in all_reviews_preProcessed_compoundSpellCorrected['iTunes'].values() for review in reviews]
    WPP_Google_compoundSpellCorrected =         [review for reviews in all_reviews_preProcessed_compoundSpellCorrected['Google'].values() for review in reviews]
    WPP_Replika_compoundSpellCorrected =        [review for app_dict in all_reviews_preProcessed_compoundSpellCorrected.values() for review in app_dict['Replika']]
    WPP_Wysa_compoundSpellCorrected =           [review for app_dict in all_reviews_preProcessed_compoundSpellCorrected.values() for review in app_dict['Wysa']]
    WPP_Woebot_compoundSpellCorrected =         [review for app_dict in all_reviews_preProcessed_compoundSpellCorrected.values() for review in app_dict['Woebot']]
    WPP_Daylio_compoundSpellCorrected =         [review for app_dict in all_reviews_preProcessed_compoundSpellCorrected.values() for review in app_dict['Daylio']]
    WPP_MoodMeter_compoundSpellCorrected =      [review for app_dict in all_reviews_preProcessed_compoundSpellCorrected.values() for review in app_dict['MoodMeter']]
    WPP_iMoodJournal_compoundSpellCorrected =   [review for app_dict in all_reviews_preProcessed_compoundSpellCorrected.values() for review in app_dict['iMoodJournal']]
    WPP_eMoodTracker_compoundSpellCorrected =   [review for app_dict in all_reviews_preProcessed_compoundSpellCorrected.values() for review in app_dict['eMoodTracker']]
    
        ### With Pre Processing (inlcudes lemmatization and POS Tagging) ###
        ### WITH STANDARD TEXTBLOB SPELL CHECKING ###
    WPP_all_reviews_standardTextBlobSpellCorrected =    [review for app_dict in all_reviews_preProcessed_standardTextBlobSpellCorrected.values() for reviews in app_dict.values() for review in reviews]
    WPP_apple_standardTextBlobSpellCorrected =          [review for reviews in all_reviews_preProcessed_standardTextBlobSpellCorrected['iTunes'].values() for review in reviews]
    WPP_Google_standardTextBlobSpellCorrected =         [review for reviews in all_reviews_preProcessed_standardTextBlobSpellCorrected['Google'].values() for review in reviews]
    WPP_Replika_standardTextBlobSpellCorrected =        [review for app_dict in all_reviews_preProcessed_standardTextBlobSpellCorrected.values() for review in app_dict['Replika']]
    WPP_Wysa_standardTextBlobSpellCorrected =           [review for app_dict in all_reviews_preProcessed_standardTextBlobSpellCorrected.values() for review in app_dict['Wysa']]
    WPP_Woebot_standardTextBlobSpellCorrected =         [review for app_dict in all_reviews_preProcessed_standardTextBlobSpellCorrected.values() for review in app_dict['Woebot']]
    WPP_Daylio_standardTextBlobSpellCorrected =         [review for app_dict in all_reviews_preProcessed_standardTextBlobSpellCorrected.values() for review in app_dict['Daylio']]
    WPP_MoodMeter_standardTextBlobSpellCorrected =      [review for app_dict in all_reviews_preProcessed_standardTextBlobSpellCorrected.values() for review in app_dict['MoodMeter']]
    WPP_iMoodJournal_standardTextBlobSpellCorrected =   [review for app_dict in all_reviews_preProcessed_standardTextBlobSpellCorrected.values() for review in app_dict['iMoodJournal']]
    WPP_eMoodTracker_standardTextBlobSpellCorrected =   [review for app_dict in all_reviews_preProcessed_standardTextBlobSpellCorrected.values() for review in app_dict['eMoodTracker']]
    
        ### With Pre Processing (inlcudes lemmatization and POS Tagging) ###
        ### WITH CUSTOM SPELL CHECKING ###
    WPP_all_reviews_customSpellCorrected =    [review for app_dict in all_reviews_preProcessed_customSpellCorrected.values() for reviews in app_dict.values() for review in reviews]
    WPP_apple_customSpellCorrected =          [review for reviews in all_reviews_preProcessed_customSpellCorrected['iTunes'].values() for review in reviews]
    WPP_Google_customSpellCorrected =         [review for reviews in all_reviews_preProcessed_customSpellCorrected['Google'].values() for review in reviews]
    WPP_Replika_customSpellCorrected =        [review for app_dict in all_reviews_preProcessed_customSpellCorrected.values() for review in app_dict['Replika']]
    WPP_Wysa_customSpellCorrected =           [review for app_dict in all_reviews_preProcessed_customSpellCorrected.values() for review in app_dict['Wysa']]
    WPP_Woebot_customSpellCorrected =         [review for app_dict in all_reviews_preProcessed_customSpellCorrected.values() for review in app_dict['Woebot']]
    WPP_Daylio_customSpellCorrected =         [review for app_dict in all_reviews_preProcessed_customSpellCorrected.values() for review in app_dict['Daylio']]
    WPP_MoodMeter_customSpellCorrected =      [review for app_dict in all_reviews_preProcessed_customSpellCorrected.values() for review in app_dict['MoodMeter']]
    WPP_iMoodJournal_customSpellCorrected =   [review for app_dict in all_reviews_preProcessed_customSpellCorrected.values() for review in app_dict['iMoodJournal']]
    WPP_eMoodTracker_customSpellCorrected =   [review for app_dict in all_reviews_preProcessed_customSpellCorrected.values() for review in app_dict['eMoodTracker']]
    
        ### With Pre Processing (inlcudes lemmatization and POS Tagging) ###
        ### WITH COMPOUND CUSTOM SPELL CHECKING ###
    WPP_all_reviews_compoundCustomSpellCorrected =    [review for app_dict in all_reviews_preProcessed_compoundCustomSpellCorrected.values() for reviews in app_dict.values() for review in reviews]
    WPP_apple_compoundCustomSpellCorrected =          [review for reviews in all_reviews_preProcessed_compoundCustomSpellCorrected['iTunes'].values() for review in reviews]
    WPP_Google_compoundCustomSpellCorrected =         [review for reviews in all_reviews_preProcessed_compoundCustomSpellCorrected['Google'].values() for review in reviews]
    WPP_Replika_compoundCustomSpellCorrected =        [review for app_dict in all_reviews_preProcessed_compoundCustomSpellCorrected.values() for review in app_dict['Replika']]
    WPP_Wysa_compoundCustomSpellCorrected =           [review for app_dict in all_reviews_preProcessed_compoundCustomSpellCorrected.values() for review in app_dict['Wysa']]
    WPP_Woebot_compoundCustomSpellCorrected =         [review for app_dict in all_reviews_preProcessed_compoundCustomSpellCorrected.values() for review in app_dict['Woebot']]
    WPP_Daylio_compoundCustomSpellCorrected =         [review for app_dict in all_reviews_preProcessed_compoundCustomSpellCorrected.values() for review in app_dict['Daylio']]
    WPP_MoodMeter_compoundCustomSpellCorrected =      [review for app_dict in all_reviews_preProcessed_compoundCustomSpellCorrected.values() for review in app_dict['MoodMeter']]
    WPP_iMoodJournal_compoundCustomSpellCorrected =   [review for app_dict in all_reviews_preProcessed_compoundCustomSpellCorrected.values() for review in app_dict['iMoodJournal']]
    WPP_eMoodTracker_compoundCustomSpellCorrected =   [review for app_dict in all_reviews_preProcessed_compoundCustomSpellCorrected.values() for review in app_dict['eMoodTracker']]
    
        ### With Pre Processing (inlcudes lemmatization and POS Tagging) ###
        ### WITH CUSTOM TEXTBLOB SPELL CHECKING ###
    WPP_all_reviews_customTextBlobSpellCorrected =    [review for app_dict in all_reviews_preProcessed_customTextBlobSpellCorrected.values() for reviews in app_dict.values() for review in reviews]
    WPP_apple_customTextBlobSpellCorrected =          [review for reviews in all_reviews_preProcessed_customTextBlobSpellCorrected['iTunes'].values() for review in reviews]
    WPP_Google_customTextBlobSpellCorrected =         [review for reviews in all_reviews_preProcessed_customTextBlobSpellCorrected['Google'].values() for review in reviews]
    WPP_Replika_customTextBlobSpellCorrected =        [review for app_dict in all_reviews_preProcessed_customTextBlobSpellCorrected.values() for review in app_dict['Replika']]
    WPP_Wysa_customTextBlobSpellCorrected =           [review for app_dict in all_reviews_preProcessed_customTextBlobSpellCorrected.values() for review in app_dict['Wysa']]
    WPP_Woebot_customTextBlobSpellCorrected =         [review for app_dict in all_reviews_preProcessed_customTextBlobSpellCorrected.values() for review in app_dict['Woebot']]
    WPP_Daylio_customTextBlobSpellCorrected =         [review for app_dict in all_reviews_preProcessed_customTextBlobSpellCorrected.values() for review in app_dict['Daylio']]
    WPP_MoodMeter_customTextBlobSpellCorrected =      [review for app_dict in all_reviews_preProcessed_customTextBlobSpellCorrected.values() for review in app_dict['MoodMeter']]
    WPP_iMoodJournal_customTextBlobSpellCorrected =   [review for app_dict in all_reviews_preProcessed_customTextBlobSpellCorrected.values() for review in app_dict['iMoodJournal']]
    WPP_eMoodTracker_customTextBlobSpellCorrected =   [review for app_dict in all_reviews_preProcessed_customTextBlobSpellCorrected.values() for review in app_dict['eMoodTracker']]
    
    print(f"Lists creation completed in {time.time() - t} seconds")
    print()
        
        ### Dump all lists out to pickles ###
    print("Pickling newly created lists")
    t = time.time()
    for varname in tqdm(['NPP_all_reviews', 'NPP_apple', 'NPP_Google', 'NPP_Replika', 'NPP_Wysa', 'NPP_Woebot', 'NPP_Daylio', 'NPP_MoodMeter', 'NPP_iMoodJournal', 'NPP_eMoodTracker',
     'NPP_all_reviews_spellCorrected', 'NPP_apple_spellCorrected', 'NPP_Google_spellCorrected', 'NPP_Replika_spellCorrected', 'NPP_Wysa_spellCorrected', 'NPP_Woebot_spellCorrected', 'NPP_Daylio_spellCorrected', 'NPP_MoodMeter_spellCorrected', 'NPP_iMoodJournal_spellCorrected', 'NPP_eMoodTracker_spellCorrected',
     'NPP_all_reviews_customSpellCorrected', 'NPP_apple_customSpellCorrected', 'NPP_Google_customSpellCorrected', 'NPP_Replika_customSpellCorrected', 'NPP_Wysa_customSpellCorrected', 'NPP_Woebot_customSpellCorrected', 'NPP_Daylio_customSpellCorrected', 'NPP_MoodMeter_customSpellCorrected', 'NPP_iMoodJournal_customSpellCorrected', 'NPP_eMoodTracker_customSpellCorrected',
     'WPP_all_reviews', 'WPP_apple', 'WPP_Google', 'WPP_Replika', 'WPP_Wysa', 'WPP_Woebot', 'WPP_Daylio', 'WPP_MoodMeter', 'WPP_iMoodJournal', 'WPP_eMoodTracker',
     'WPP_all_reviews_spellCorrected', 'WPP_apple_spellCorrected', 'WPP_Google_spellCorrected', 'WPP_Replika_spellCorrected', 'WPP_Wysa_spellCorrected', 'WPP_Woebot_spellCorrected', 'WPP_Daylio_spellCorrected', 'WPP_MoodMeter_spellCorrected', 'WPP_iMoodJournal_spellCorrected', 'WPP_eMoodTracker_spellCorrected',
     'WPP_all_reviews_customSpellCorrected', 'WPP_apple_customSpellCorrected', 'WPP_Google_customSpellCorrected', 'WPP_Replika_customSpellCorrected', 'WPP_Wysa_customSpellCorrected', 'WPP_Woebot_customSpellCorrected', 'WPP_Daylio_customSpellCorrected', 'WPP_MoodMeter_customSpellCorrected', 'WPP_iMoodJournal_customSpellCorrected', 'WPP_eMoodTracker_customSpellCorrected',
     'NPP_all_reviews_compoundSpellCorrected', 'NPP_apple_compoundSpellCorrected', 'NPP_Google_compoundSpellCorrected', 'NPP_Replika_compoundSpellCorrected', 'NPP_Wysa_compoundSpellCorrected', 'NPP_Woebot_compoundSpellCorrected', 'NPP_Daylio_compoundSpellCorrected', 'NPP_MoodMeter_compoundSpellCorrected', 'NPP_iMoodJournal_compoundSpellCorrected', 'NPP_eMoodTracker_compoundSpellCorrected',
     'NPP_all_reviews_compoundCustomSpellCorrected', 'NPP_apple_compoundCustomSpellCorrected', 'NPP_Google_compoundCustomSpellCorrected', 'NPP_Replika_compoundCustomSpellCorrected', 'NPP_Wysa_compoundCustomSpellCorrected', 'NPP_Woebot_compoundCustomSpellCorrected', 'NPP_Daylio_compoundCustomSpellCorrected', 'NPP_MoodMeter_compoundCustomSpellCorrected', 'NPP_iMoodJournal_compoundCustomSpellCorrected', 'NPP_eMoodTracker_compoundCustomSpellCorrected',
     'WPP_all_reviews_compoundSpellCorrected', 'WPP_apple_compoundSpellCorrected', 'WPP_Google_compoundSpellCorrected', 'WPP_Replika_compoundSpellCorrected', 'WPP_Wysa_compoundSpellCorrected', 'WPP_Woebot_compoundSpellCorrected', 'WPP_Daylio_compoundSpellCorrected', 'WPP_MoodMeter_compoundSpellCorrected', 'WPP_iMoodJournal_compoundSpellCorrected', 'WPP_eMoodTracker_compoundSpellCorrected',
     'NPP_all_reviews_standardTextBlobSpellCorrected', 'NPP_apple_standardTextBlobSpellCorrected', 'NPP_Google_standardTextBlobSpellCorrected', 'NPP_Replika_standardTextBlobSpellCorrected', 'NPP_Wysa_standardTextBlobSpellCorrected', 'NPP_Woebot_standardTextBlobSpellCorrected', 'NPP_Daylio_standardTextBlobSpellCorrected', 'NPP_MoodMeter_standardTextBlobSpellCorrected', 'NPP_iMoodJournal_standardTextBlobSpellCorrected', 'NPP_eMoodTracker_standardTextBlobSpellCorrected',
     'NPP_all_reviews_customTextBlobSpellCorrected', 'NPP_apple_customTextBlobSpellCorrected', 'NPP_Google_customTextBlobSpellCorrected', 'NPP_Replika_customTextBlobSpellCorrected', 'NPP_Wysa_custoTextBlobmSpellCorrected', 'NPP_Woebot_customTextBlobSpellCorrected', 'NPP_Daylio_customTextBlobSpellCorrected', 'NPP_MoodMeter_customTextBlobSpellCorrected', 'NPP_iMoodJournal_customTextBlobSpellCorrected', 'NPP_eMoodTracker_customTextBlobSpellCorrected',
     'WPP_all_reviews_standardTextBlobSpellCorrected', 'WPP_apple_standardTextBlobSpellCorrected', 'WPP_Google_standardTextBlobSpellCorrected', 'WPP_Replika_standardTextBlobSpellCorrected', 'WPP_Wysa_standardTextBlobSpellCorrected', 'WPP_Woebot_standardTextBlobSpellCorrected', 'WPP_Daylio_standardTextBlobSpellCorrected', 'WPP_MoodMeter_standardTextBlobSpellCorrected', 'WPP_iMoodJournal_standardTextBlobSpellCorrected', 'WPP_eMoodTracker_standardTextBlobSpellCorrected',
     'WPP_all_reviews_customTextBlobSpellCorrected', 'WPP_apple_customTextBlobSpellCorrected', 'WPP_Google_customTextBlobSpellCorrected', 'WPP_Replika_customTextBlobSpellCorrected', 'WPP_Wysa_customTextBlobSpellCorrected', 'WPP_Woebot_customTextBlobSpellCorrected', 'WPP_Daylio_customTextBlobSpellCorrected', 'WPP_MoodMeter_customTextBlobSpellCorrected', 'WPP_iMoodJournal_customTextBlobSpellCorrected', 'WPP_eMoodTracker_customTextBlobSpellCorrected'], 
                        desc="Pickling"):
        pickle.dump( eval(varname), open(directoryString + 'pickles/' + varname + '.p', "wb" ) )
    print(f"Lists pickling completed in {time.time() - t} seconds")
        ### Clear all unneeded variables (as they are HUGE), to clear some space in memory ###
    # del all_reviews_preProcessed, all_reviews_preProcessed_spellCorrected, all_reviews_raw, lemmad_store_app_reviews, raw_tokenised_stopwordRemoved, spellCorrected_lemmad_store_app_reviews, spellCorrected_tokenised_store_app_reviews, tokenised_store_app_reviews
    
    print("     ### No Pre-Processing - No Spell Checking ###")
    print(NPP_eMoodTracker[560])
    print()
    print("     ### No Pre-Processing - Standard Spell Checking ###")
    print(NPP_eMoodTracker_spellCorrected[560])
    print()
    print("     ### No Pre-Processing - Compound Standard Spell Checking ###")
    print(NPP_eMoodTracker_compoundSpellCorrected[560])
    print()
    print("     ### No Pre-Processing - Custom Spell Checking ###")
    print(NPP_eMoodTracker_customSpellCorrected[560])
    print()
    print("     ### No Pre-Processing - Compound Custom Spell Checking ###")
    print(NPP_eMoodTracker_compoundCustomSpellCorrected[560])
    print()
    print("     ### No Pre-Processing - Standard TextBlob Spell Checking ###")
    print(NPP_eMoodTracker_standardTextBlobSpellCorrected[560])
    print()
    print("     ### No Pre-Processing - Custom TextBlob Spell Checking ###")
    print(NPP_eMoodTracker_customTextBlobSpellCorrected[560])
    print()
    print("     ### With Pre-Processing - No Spell Checking ###")
    print(WPP_eMoodTracker[560])
    print()
    print("     ### With Pre-Processing - Standard Spell Checking ###")
    print(WPP_eMoodTracker_spellCorrected[560])
    print()
    print("     ### With Pre-Processing - Compound Standard Spell Checking ###")
    print(WPP_eMoodTracker_compoundSpellCorrected[560])
    print()    
    print("     ### With Pre-Processing - Custom Spell Checking ###")
    print(WPP_eMoodTracker_customSpellCorrected[560])
    print()    
    print("     ### With Pre-Processing - Compound Custom Spell Checking ###")
    print(WPP_eMoodTracker_compoundCustomSpellCorrected[560])
    print()
    print("     ### With Pre-Processing - Standard TextBlob Spell Checking ###")
    print(WPP_eMoodTracker_standardTextBlobSpellCorrected[560])
    print()
    print("     ### With Pre-Processing - Custom TextBlob Spell Checking ###")
    print(WPP_eMoodTracker_customTextBlobSpellCorrected[560])    
    
    ##### SPELL CHECKING PHASE #####
    # input_terms = []
    # for word in WPP_all_reviews:
    #         tokens = [x[0] for x in word]
    #         # string = " ".join(tokens)
    #         input_terms.append(tokens)
    # print(input_terms[350500])
    
    # print(spellCorrectReview(input_terms[350500]))
    # for word in input_terms[350500]:
    #     suggestions = sym_spell_standard.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True, ignore_token=r"(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])(.*)")
    #     for suggestion in suggestions:
    #         # print(suggestion, end=" ")
    #         print(suggestion.term, end=" ")
            
    # print()
    # print(" - - - - - - - - - - ")
    # print()
    
    # print(spellCorrectReview(NPP_all_reviews[350500]))
    # print(NPP_all_reviews[350500])
    # for word in NPP_all_reviews[350500]:
    #     suggestions = sym_spell_standard.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True, ignore_token=r"(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])(.*)")
    #     for suggestion in suggestions:
    #         # print(suggestion, end=" ")
    #         print(suggestion.term, end=" ")
    

"""Main"""
if __name__=="__main__":
    # This must be called as part of the parrelization requirements
    multiprocessing.freeze_support()
    # Execute the program
    main()