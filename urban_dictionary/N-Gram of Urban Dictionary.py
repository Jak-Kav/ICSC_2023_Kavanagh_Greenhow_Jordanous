import ast
import multiprocessing
import pickle
import time
import timeit
from textblob import TextBlob
from textblob.en import Spelling
import os
import pandas as pd
from tqdm import tqdm
from functools import reduce

# Set the path of the new custom dictionary
path = "C:/Python310/Lib/site-packages/textblob/en/urban-spelling.txt"
assert os.path.isfile(path)
# Load the new dictionary
spelling = Spelling(path=path)

directoryString = "C:/Users/Student/OneDrive - University of Kent/URSS/Urban Dictionary Data/"

# Load pickles
List_of_All_UrbanDict_Bigrams = pickle.load( open( directoryString + 'pickles/' + 'List_of_All_UrbanDict_Bigrams.p', "rb" ) )
List_of_All_UrbanDict_Trigrams = pickle.load( open( directoryString + 'pickles/' + 'List_of_All_UrbanDict_Trigrams.p', "rb" ) )
List_of_All_UrbanDict_Quadgrams = pickle.load( open( directoryString + 'pickles/' + 'List_of_All_UrbanDict_Quadgrams.p', "rb" ) )
List_of_All_UrbanDict_Quintgrams = pickle.load( open( directoryString + 'pickles/' + 'List_of_All_UrbanDict_Quintgrams.p', "rb" ) )
All_Reviews_List_Of_Strings = pickle.load( open( directoryString + 'pickles/' + 'All_Reviews_List_Of_Strings.p', "rb" ) )
All_Reviews_List_Of_TextBlob = pickle.load( open( directoryString + 'pickles/' + 'All_Reviews_List_Of_TextBlob.p', "rb" ) )
All_Reviews_List_Of_Bigrams = pickle.load( open( directoryString + 'pickles/' + 'All_Reviews_List_Of_Bigrams.p', "rb" ) )
All_Reviews_List_Of_Trigrams = pickle.load( open( directoryString + 'pickles/' + 'All_Reviews_List_Of_Trigrams.p', "rb" ) )
All_Reviews_List_Of_Quadgrams = pickle.load( open( directoryString + 'pickles/' + 'All_Reviews_List_Of_Quadgrams.p', "rb" ) )
All_Reviews_List_Of_Quintgrams = pickle.load( open( directoryString + 'pickles/' + 'All_Reviews_List_Of_Quintgrams.p', "rb" ) )
bigram_list_1d = [val for sublist in All_Reviews_List_Of_Bigrams for val in sublist]
trigram_list_1d = [val for sublist in All_Reviews_List_Of_Trigrams for val in sublist]
quadgram_list_1d = [val for sublist in All_Reviews_List_Of_Quadgrams for val in sublist]
quintgram_list_1d = [val for sublist in All_Reviews_List_Of_Quintgrams for val in sublist]

# Check n-grams against Urban Dictionary N-Gram list
def urbanDict_ngram_to_pickle():
    # Import the bi-grams csv
    start = timeit.default_timer()
    print("Importing Urban Dictionary Bi-Grams")
    urban_bi_grams_df = pd.DataFrame(pd.read_csv(directoryString = "Processed Urban Dictionary Bi-Grams.csv"), columns=['word'])
    # Convert the DataFrame to a list of bi-grams
    urban_bi_grams = urban_bi_grams_df['word'].tolist()
    end = timeit.default_timer()
    print(end - start, "seconds")
    print("Exporting list of Urban Dictionary Bi-Grams to pickle")
    start = timeit.default_timer()
    pickle.dump( urban_bi_grams, open(directoryString + 'pickles/' + 'List_of_All_UrbanDict_Bigrams.p', "wb" ) )
    end = timeit.default_timer()
    print(end - start, "seconds")

    # Import the tri-grams csv
    start = timeit.default_timer()
    print("Importing Urban Dictionary Tri-Grams")
    urban_tri_grams_df = pd.DataFrame(pd.read_csv(directoryString + "Processed Urban Dictionary Tri-Grams.csv"), columns=['word'])
    # Convert the DataFrame to a list of Tri-grams
    urban_tri_grams = urban_tri_grams_df['word'].tolist()
    end = timeit.default_timer()
    print(end - start, "seconds")
    print("Exporting list of Urban Dictionary Tri-Grams to pickle")
    start = timeit.default_timer()
    pickle.dump( urban_tri_grams, open(directoryString + 'pickles/' + 'List_of_All_UrbanDict_Trigrams.p', "wb" ) )
    end = timeit.default_timer()
    print(end - start, "seconds")

    # Import the Quad-grams csv
    start = timeit.default_timer()
    print("Importing Urban Dictionary Quad-Grams")
    urban_quad_grams_df = pd.DataFrame(pd.read_csv(directoryString + "Processed Urban Dictionary Quad-Grams.csv"), columns=['word'])
    # Convert the DataFrame to a list of Quad-grams
    urban_quad_grams = urban_quad_grams_df['word'].tolist()
    end = timeit.default_timer()
    print(end - start, "seconds")
    print("Exporting list of Urban Dictionary Quad-Grams to pickle")
    start = timeit.default_timer()
    pickle.dump( urban_quad_grams, open(directoryString + 'pickles/' + 'List_of_All_UrbanDict_Quadgrams.p', "wb" ) )
    end = timeit.default_timer()
    print(end - start, "seconds")

    # Import the Quint-grams csv
    start = timeit.default_timer()
    print("Importing Urban Dictionary Quint-Grams")
    urban_quint_grams_df = pd.DataFrame(pd.read_csv(directoryString + "Processed Urban Dictionary Quint-Grams.csv"), columns=['word'])
    # Convert the DataFrame to a list of Quint-grams
    urban_quint_grams = urban_quint_grams_df['word'].tolist()
    end = timeit.default_timer()
    print(end - start, "seconds")
    print("Exporting list of Urban Dictionary Quint-Grams to pickle")
    start = timeit.default_timer()
    pickle.dump( urban_quint_grams, open(directoryString + 'pickles/' + 'List_of_All_UrbanDict_Quintgrams.p', "wb" ) )
    end = timeit.default_timer()
    print(end - start, "seconds")

# Import the pickled review data
def reviewData_convert_and_pickle():
    start = timeit.default_timer()
    print("Importing pickled review data")
    directoryString2 = "C:/Users/Student/OneDrive - University of Kent/URSS/GitLab Repo/urss-moody-reviews/Data Scraping/DATA/"
    exportString = directoryString2 + "LEMMA_FreqDist/"
    all_replika_reviews = pickle.load( open( exportString + 'pickles/' + 'All_Reviews.p', "rb" ) )
    end = timeit.default_timer()
    print(end - start, "seconds")

    # Go through the pickled data and convert each 'review' to a list of strings (words)
    start = timeit.default_timer()
    print("Converting pickled data into a list of individual reviews")
    each_review = [] # Create empty list to store reviews to
    # Work through the pickled data dictionary structure and convert reviews as a list of individual reviews
    for store, app in tqdm(all_replika_reviews.items(), position=0, desc="Store", leave=False, miniters=0):
        for app, review in tqdm(app.items(), position=1, desc="App", leave=False, miniters=0):
            for review, word in tqdm(review.items(), position=3, desc="Review", leave=False, miniters=0):
                tokens = [x[0] for x in word]
                string = " ".join(tokens)
                each_review.append(string)
                #time.sleep(0.1)
    end = timeit.default_timer()
    print(end - start, "seconds")
    print("Exporting list of strings to pickle")
    start = timeit.default_timer()
    pickle.dump( each_review, open(directoryString + 'pickles/' + 'All_Reviews_List_Of_Strings.p', "wb" ) )
    end = timeit.default_timer()
    print(end - start, "seconds")

    # Turn the sentance into bi-grams via TextBlob functions
    start = timeit.default_timer()
    print("Creating a TextBlob object of each review")
    review_TextBlob_list = [] # Create empty list to store TextBlob objects to
    for review in tqdm(each_review, miniters=0):
        review_TextBlob = TextBlob(review)
        review_TextBlob_list.append(review_TextBlob)
    end = timeit.default_timer()
    print(end - start, "seconds")
    print("Exporting list of TextBlob objects to pickle")
    start = timeit.default_timer()
    pickle.dump( review_TextBlob_list, open(directoryString + 'pickles/' + 'All_Reviews_List_Of_TextBlob.p', "wb" ) )
    end = timeit.default_timer()
    print(end - start, "seconds")
    
def textBlob_spelling_correction():
    spellCorrected_review_TextBlob_list = []
    start = timeit.default_timer()
    chnks = chunks(All_Reviews_List_Of_TextBlob, multiprocessing.cpu_count())
    with multiprocessing.Pool() as pool:
        for result in pool.imap(parallelSpellCorrect, chnks):
            spellCorrected_review_TextBlob_list.extend(result)            
    end = timeit.default_timer()    
    print("TextBlob Spell Correction")
    print("Time:", (end - start))
    print("Exporting list of spell corrected TextBlob objects to pickle")
    start = timeit.default_timer()
    pickle.dump( spellCorrected_review_TextBlob_list, open(directoryString + 'pickles/' + 'All_Reviews_Spell-Corrected_List_Of_TextBlob.p', "wb" ) )
    end = timeit.default_timer()
    print(end - start, "seconds")
    
def textBlob_to_ngrams(ngram_num):
    start = timeit.default_timer()
    print("Creating a list of N-Grams from each TextBlob object")
    review_ngram_list = []
    for tb_review in tqdm(All_Reviews_List_Of_TextBlob, miniters=0):
        review_bigram = tb_review.ngrams(ngram_num)
        result = [' '.join(i) for i in review_bigram]
        review_ngram_list.append(result)
    end = timeit.default_timer()
    print(end - start, "seconds")
    print("Exporting list of N-Grams to pickle")
    start = timeit.default_timer()
    if ngram_num == 2:
        pickle.dump( review_ngram_list, open(directoryString + 'pickles/' + 'All_Reviews_List_Of_Bigrams.p', "wb" ) )
    elif ngram_num == 3:
        pickle.dump( review_ngram_list, open(directoryString + 'pickles/' + 'All_Reviews_List_Of_Trigrams.p', "wb" ) )
    elif ngram_num == 4:
        pickle.dump( review_ngram_list, open(directoryString + 'pickles/' + 'All_Reviews_List_Of_Quadgrams.p', "wb" ) )
    elif ngram_num == 5:
        pickle.dump( review_ngram_list, open(directoryString + 'pickles/' + 'All_Reviews_List_Of_Quintgrams.p', "wb" ) )
    else:
        pickle.dump( review_ngram_list, open(directoryString + 'pickles/' + 'All_Reviews_List_Of_?grams.p', "wb" ) )
    end = timeit.default_timer()
    print(end - start, "seconds")

# # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # #

def chunks(lst, n):
    """Yield successive `n`-sized chunks from a list.

    Args:
        lst (list): The list to be split up.
        n (int): The amount of chunks to split the list into.

    Yields:
        list: `n` lists derived from the original list.
    """ 
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
            
def listCompare(ngrams_chunk, urbanDictList):
    toReturn = []
    for ngram in ngrams_chunk:
        if ngram in urbanDictList:
            toReturn.append(ngram)
    return toReturn

def parallelSpellCorrect(list_chunk):
    toReturn = []
    for tb_review in list_chunk:
        corrected = TextBlob.correct(tb_review)
        toReturn.append(corrected)
    return toReturn

# def listCompare2(ngrams_chunk):
#     toReturn = []
#     for ngram in ngrams_chunk:
#         if ngram in List_of_All_UrbanDict_Trigrams:
#             toReturn.append(ngram)
#     return toReturn

def main():
    # Find any matching bi-grams
    
    # # # # # Iterativley # # # # #
    # out1 = []
    # start = timeit.default_timer()
    # for bigram in tqdm(bigram_list_1d[:5000], position=0, desc="Bi-Gram", leave=False, miniters=0):
    #     if bigram in List_of_All_UrbanDict_Bigrams:
    #         out1.append(bigram)
    # end = timeit.default_timer()
    # print("First 5000 items")
    # print("Iterative:", out1)
    # print("Time:", (end - start))
    # print("- - - - - - - - - -")
    # print(len(bigram_list_1d))
    
    # # # # # PARALLEL # # # # #
    # Create an empty list to output to
    # out = []
    # start = timeit.default_timer()
    # chnks = chunks(trigram_list_1d, multiprocessing.cpu_count())
    # with multiprocessing.Pool() as pool:
    #     for resultB in pool.map(listCompare2, chnks):
    #         out.extend(resultB)            
    # end = timeit.default_timer()    
    # print("All 4788131 items")
    # print("Parallel:", out)
    # print("Time:", (end - start))
    
    # # # # # FINAL COMPLETE SEQUENCE # # # # #
    final_bigram_output_list = []
    final_trigram_output_list = []
    final_quadgram_output_list = []
    final_quintgram_output_list = []
    chnksA = chunks(bigram_list_1d, multiprocessing.cpu_count())
    chnksB = chunks(trigram_list_1d, multiprocessing.cpu_count())
    chnksC = chunks(quadgram_list_1d, multiprocessing.cpu_count())
    chnksD = chunks(quintgram_list_1d, multiprocessing.cpu_count())
    with multiprocessing.Pool() as pool:
        print("Starting Bi-Grams")
        for resultA in pool.starmap(listCompare, [(a, List_of_All_UrbanDict_Bigrams) for a in chnksA]):
            final_bigram_output_list.extend(resultA)
        print("Bi-Grams complete")
        print("Starting Tri-Grams")
        for resultB in pool.starmap(listCompare, [(b, List_of_All_UrbanDict_Trigrams) for b in chnksB]):
            final_trigram_output_list.extend(resultB)
        print("Tri-Grams complete")
        print("Starting Quad-Grams")
        for resultC in pool.starmap(listCompare, [(c, List_of_All_UrbanDict_Quadgrams) for c in chnksC]):
            final_quadgram_output_list.extend(resultC)
        print("Quad-Grams complete")
        print("Starting Quint-Grams")
        for resultD in pool.starmap(listCompare, [(d, List_of_All_UrbanDict_Quintgrams) for d in chnksD]):
            final_quintgram_output_list.extend(resultD)
        print("Quint-Grams Complete")
                
    # Export the pickles and export the outputs as CSV
    print("Exporting pickles and CSV's")
    pickle.dump( final_bigram_output_list, open(directoryString + 'pickles/' + 'Final_Bigram_Output_List.p', "wb" ) )
    pickle.dump( final_trigram_output_list, open(directoryString + 'pickles/' + 'Final_Trigram_Output_List.p', "wb" ) )
    pickle.dump( final_quadgram_output_list, open(directoryString + 'pickles/' + 'Final_Quadigram_Output_List.p', "wb" ) )
    pickle.dump( final_quintgram_output_list, open(directoryString + 'pickles/' + 'Final_Quintgram_Output_List.p', "wb" ) )
    bigram_df = pd.DataFrame(final_bigram_output_list, columns=["Bi-Gram"])
    trigram_df = pd.DataFrame(final_trigram_output_list, columns=["Tri-Gram"])
    quadgram_df = pd.DataFrame(final_quadgram_output_list, columns=["Quad-Gram"])
    quintgram_df = pd.DataFrame(final_quintgram_output_list, columns=["Quint-Gram"])
    bigram_df.to_csv(directoryString + 'pickles/' + 'Final_Bigram_Output_List.csv', index=False)
    trigram_df.to_csv(directoryString + 'pickles/' + 'Final_Trigram_Output_List.csv', index=False)
    quadgram_df.to_csv(directoryString + 'pickles/' + 'Final_Quadgram_Output_List.csv', index=False)
    quintgram_df.to_csv(directoryString + 'pickles/' + 'Final_Quintgram_Output_List.csv', index=False)
    print("Exports complete")
    

""" ENTRY POINT FOR THE SCRIPT """
if __name__=="__main__":
    # This must be called as part of the parrelization requirements
    multiprocessing.freeze_support()
    # Execute the program
    main()