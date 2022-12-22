import os
import pandas as pd
from tqdm import tqdm

answer = input("Have you got the correct headers in the CSV file? Press 'y' or 'n'")
if answer == 'n':
    print("Then go do that first!")
    quit()
answer = input("Have you combined both the iTunes and Google datasets? Press 'y' or 'n'")
if answer == 'n':
    print("Then go do that first!")
    quit()
answer = input("Have you removed all the '-' symbols from the data? Press 'y' or 'n'")
if answer == 'n':
    print("Then go do that first!")
    quit()

### Global Variables ###
directoryString = "C:/Users/Student/OneDrive - University of Kent/URSS/Experiment 1 - Spell or No Spell/"
importedDataFrame = pd.read_csv("C:/Users/Student/OneDrive - University of Kent/URSS/Experiment 1 - Spell or No Spell/Final_All_Results_Combined.csv")

# Create the Pandas DataFrame to export
columnNames = ['Review', 'Pre_Processed', 'Spell_Checked', 'Roberta_Result', 'VADER_Result', 'Roberta_Bigram_Result', 
               'VADER_Bigram_Result', 'Roberta_Trigram_Result', 'VADER_Trigram_Result']
resultsDataFrame = pd.DataFrame(columns=columnNames)

tmp = [resultsDataFrame]
for index, row in tqdm(importedDataFrame.iterrows(), total=importedDataFrame.shape[0], desc="Creating new converted rows"):
    data = [
        {'Review':row['Review'], 'Pre_Processed':1, 'Spell_Checked':"None", 'Roberta_Result':row['Roberta_WPP_Spellcheck.NO_SPELLCHECK'], 'VADER_Result':row['Vader_WPP_Spellcheck.NO_SPELLCHECK'], 
            'Roberta_Bigram_Result':row['Roberta_Bigram_WPP_Spellcheck.NO_SPELLCHECK'], 'VADER_Bigram_Result':row['Vader_Bigram_WPP_Spellcheck.NO_SPELLCHECK'], 
            'Roberta_Trigram_Result':row['Roberta_Trigram_WPP_Spellcheck.NO_SPELLCHECK'], 'VADER_Trigram_Result':row['Vader_Trigram_WPP_Spellcheck.NO_SPELLCHECK']},
        {'Review':row['Review'], 'Pre_Processed':1, 'Spell_Checked':"Symspell", 'Roberta_Result':row['Roberta_WPP_Spellcheck.SYMSPELL'], 'VADER_Result':row['Vader_WPP_Spellcheck.SYMSPELL'], 
            'Roberta_Bigram_Result':row['Roberta_Bigram_WPP_Spellcheck.SYMSPELL'], 'VADER_Bigram_Result':row['Vader_Bigram_WPP_Spellcheck.SYMSPELL'], 
            'Roberta_Trigram_Result':row['Roberta_Trigram_WPP_Spellcheck.SYMSPELL'], 'VADER_Trigram_Result':row['Vader_Trigram_WPP_Spellcheck.SYMSPELL']},
        {'Review':row['Review'], 'Pre_Processed':1, 'Spell_Checked':"Symspell_Compound", 'Roberta_Result':row['Roberta_WPP_Spellcheck.SYMSPELL_COMPOUND'], 'VADER_Result':row['Vader_WPP_Spellcheck.SYMSPELL_COMPOUND'], 
            'Roberta_Bigram_Result':row['Roberta_Bigram_WPP_Spellcheck.SYMSPELL_COMPOUND'], 'VADER_Bigram_Result':row['Vader_Bigram_WPP_Spellcheck.SYMSPELL_COMPOUND'], 
            'Roberta_Trigram_Result':row['Roberta_Trigram_WPP_Spellcheck.SYMSPELL_COMPOUND'], 'VADER_Trigram_Result':row['Vader_Trigram_WPP_Spellcheck.SYMSPELL_COMPOUND']},
        {'Review':row['Review'], 'Pre_Processed':1, 'Spell_Checked':"TextBlob", 'Roberta_Result':row['Roberta_WPP_Spellcheck.TEXTBLOB'], 'VADER_Result':row['Vader_WPP_Spellcheck.TEXTBLOB'], 
            'Roberta_Bigram_Result':row['Roberta_Bigram_WPP_Spellcheck.TEXTBLOB'], 'VADER_Bigram_Result':row['Vader_Bigram_WPP_Spellcheck.TEXTBLOB'], 
            'Roberta_Trigram_Result':row['Roberta_Trigram_WPP_Spellcheck.TEXTBLOB'], 'VADER_Trigram_Result':row['Vader_Trigram_WPP_Spellcheck.TEXTBLOB']},
        {'Review':row['Review'], 'Pre_Processed':1, 'Spell_Checked':"Symspell_Urban", 'Roberta_Result':row['Roberta_WPP_Spellcheck.SYMSPELL_URBAN'], 'VADER_Result':row['Vader_WPP_Spellcheck.SYMSPELL_URBAN'], 
            'Roberta_Bigram_Result':row['Roberta_Bigram_WPP_Spellcheck.SYMSPELL_URBAN'], 'VADER_Bigram_Result':row['Vader_Bigram_WPP_Spellcheck.SYMSPELL_URBAN'], 
            'Roberta_Trigram_Result':row['Roberta_Trigram_WPP_Spellcheck.SYMSPELL_URBAN'], 'VADER_Trigram_Result':row['Vader_Trigram_WPP_Spellcheck.SYMSPELL_URBAN']},
        {'Review':row['Review'], 'Pre_Processed':1, 'Spell_Checked':"Symspell_Compound_Urban", 'Roberta_Result':row['Roberta_WPP_Spellcheck.SYMSPELL_COMPOUND_URBAN'], 'VADER_Result':row['Vader_WPP_Spellcheck.SYMSPELL_COMPOUND_URBAN'], 
            'Roberta_Bigram_Result':row['Roberta_Bigram_WPP_Spellcheck.SYMSPELL_COMPOUND_URBAN'], 'VADER_Bigram_Result':row['Vader_Bigram_WPP_Spellcheck.SYMSPELL_COMPOUND_URBAN'], 
            'Roberta_Trigram_Result':row['Roberta_Trigram_WPP_Spellcheck.SYMSPELL_COMPOUND_URBAN'], 'VADER_Trigram_Result':row['Vader_Trigram_WPP_Spellcheck.SYMSPELL_COMPOUND_URBAN']},
        {'Review':row['Review'], 'Pre_Processed':1, 'Spell_Checked':"TextBlob_Urban", 'Roberta_Result':row['Roberta_WPP_Spellcheck.TEXTBLOB_URBAN'], 'VADER_Result':row['Vader_WPP_Spellcheck.TEXTBLOB_URBAN'], 
            'Roberta_Bigram_Result':row['Roberta_Bigram_WPP_Spellcheck.TEXTBLOB_URBAN'], 'VADER_Bigram_Result':row['Vader_Bigram_WPP_Spellcheck.TEXTBLOB_URBAN'], 
            'Roberta_Trigram_Result':row['Roberta_Trigram_WPP_Spellcheck.TEXTBLOB_URBAN'], 'VADER_Trigram_Result':row['Vader_Trigram_WPP_Spellcheck.TEXTBLOB_URBAN']},
        {'Review':row['Review'], 'Pre_Processed':0, 'Spell_Checked':"None", 'Roberta_Result':row['Roberta_NPP_Spellcheck.NO_SPELLCHECK'], 'VADER_Result':row['Vader_NPP_Spellcheck.NO_SPELLCHECK'], 
            'Roberta_Bigram_Result':row['Roberta_Bigram_NPP_Spellcheck.NO_SPELLCHECK'], 'VADER_Bigram_Result':row['Vader_Bigram_NPP_Spellcheck.NO_SPELLCHECK'], 
            'Roberta_Trigram_Result':row['Roberta_Trigram_NPP_Spellcheck.NO_SPELLCHECK'], 'VADER_Trigram_Result':row['Vader_Trigram_NPP_Spellcheck.NO_SPELLCHECK']},
        {'Review':row['Review'], 'Pre_Processed':0, 'Spell_Checked':"Symspell", 'Roberta_Result':row['Roberta_NPP_Spellcheck.SYMSPELL'], 'VADER_Result':row['Vader_NPP_Spellcheck.SYMSPELL'], 
            'Roberta_Bigram_Result':row['Roberta_Bigram_NPP_Spellcheck.SYMSPELL'], 'VADER_Bigram_Result':row['Vader_Bigram_NPP_Spellcheck.SYMSPELL'], 
            'Roberta_Trigram_Result':row['Roberta_Trigram_NPP_Spellcheck.SYMSPELL'], 'VADER_Trigram_Result':row['Vader_Trigram_NPP_Spellcheck.SYMSPELL']},
        {'Review':row['Review'], 'Pre_Processed':0, 'Spell_Checked':"Symspell_Compound", 'Roberta_Result':row['Roberta_NPP_Spellcheck.SYMSPELL_COMPOUND'], 'VADER_Result':row['Vader_NPP_Spellcheck.SYMSPELL_COMPOUND'], 
            'Roberta_Bigram_Result':row['Roberta_Bigram_NPP_Spellcheck.SYMSPELL_COMPOUND'], 'VADER_Bigram_Result':row['Vader_Bigram_NPP_Spellcheck.SYMSPELL_COMPOUND'], 
            'Roberta_Trigram_Result':row['Roberta_Trigram_NPP_Spellcheck.SYMSPELL_COMPOUND'], 'VADER_Trigram_Result':row['Vader_Trigram_NPP_Spellcheck.SYMSPELL_COMPOUND']},
        {'Review':row['Review'], 'Pre_Processed':0, 'Spell_Checked':"TextBlob", 'Roberta_Result':row['Roberta_NPP_Spellcheck.TEXTBLOB'], 'VADER_Result':row['Vader_NPP_Spellcheck.TEXTBLOB'], 
            'Roberta_Bigram_Result':row['Roberta_Bigram_NPP_Spellcheck.TEXTBLOB'], 'VADER_Bigram_Result':row['Vader_Bigram_NPP_Spellcheck.TEXTBLOB'], 
            'Roberta_Trigram_Result':row['Roberta_Trigram_NPP_Spellcheck.TEXTBLOB'], 'VADER_Trigram_Result':row['Vader_Trigram_NPP_Spellcheck.TEXTBLOB']},
        {'Review':row['Review'], 'Pre_Processed':0, 'Spell_Checked':"Symspell_Urban", 'Roberta_Result':row['Roberta_NPP_Spellcheck.SYMSPELL_URBAN'], 'VADER_Result':row['Vader_NPP_Spellcheck.SYMSPELL_URBAN'], 
            'Roberta_Bigram_Result':row['Roberta_Bigram_NPP_Spellcheck.SYMSPELL_URBAN'], 'VADER_Bigram_Result':row['Vader_Bigram_NPP_Spellcheck.SYMSPELL_URBAN'], 
            'Roberta_Trigram_Result':row['Roberta_Trigram_NPP_Spellcheck.SYMSPELL_URBAN'], 'VADER_Trigram_Result':row['Vader_Trigram_NPP_Spellcheck.SYMSPELL_URBAN']},
        {'Review':row['Review'], 'Pre_Processed':0, 'Spell_Checked':"Symspell_Compound_Urban", 'Roberta_Result':row['Roberta_NPP_Spellcheck.SYMSPELL_COMPOUND_URBAN'], 'VADER_Result':row['Vader_NPP_Spellcheck.SYMSPELL_COMPOUND_URBAN'], 
            'Roberta_Bigram_Result':row['Roberta_Bigram_NPP_Spellcheck.SYMSPELL_COMPOUND_URBAN'], 'VADER_Bigram_Result':row['Vader_Bigram_NPP_Spellcheck.SYMSPELL_COMPOUND_URBAN'], 
            'Roberta_Trigram_Result':row['Roberta_Trigram_NPP_Spellcheck.SYMSPELL_COMPOUND_URBAN'], 'VADER_Trigram_Result':row['Vader_Trigram_NPP_Spellcheck.SYMSPELL_COMPOUND_URBAN']},
        {'Review':row['Review'], 'Pre_Processed':0, 'Spell_Checked':"TextBlob_Urban", 'Roberta_Result':row['Roberta_NPP_Spellcheck.TEXTBLOB_URBAN'], 'VADER_Result':row['Vader_NPP_Spellcheck.TEXTBLOB_URBAN'], 
            'Roberta_Bigram_Result':row['Roberta_Bigram_NPP_Spellcheck.TEXTBLOB_URBAN'], 'VADER_Bigram_Result':row['Vader_Bigram_NPP_Spellcheck.TEXTBLOB_URBAN'], 
            'Roberta_Trigram_Result':row['Roberta_Trigram_NPP_Spellcheck.TEXTBLOB_URBAN'], 'VADER_Trigram_Result':row['Vader_Trigram_NPP_Spellcheck.TEXTBLOB_URBAN']}
    ]
    df = pd.DataFrame(data, index=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'])
    tmp.append(df)
print("All rows converted")
print("Concatenating all new rows into single DataFrame")
resultsDataFrame = pd.concat(tmp)
print("Concatenation complete")
print("Writing out to CSV file")
resultsDataFrame.to_csv(directoryString + "Final_All_Results_Combined_for_ANOVA.csv", index=False)
print("File writing complete")
