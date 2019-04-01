README SKIP GRAM PROJECT 

	Project Description

The skipgram project aims at programming an algorithm in Python to determine for any given word its embedding and to assess its degree of similarity with another given word. At the end of the exercise our class may return for a given word of a given corpus the N-closest ones. This file is a detailed descritpion of the creation process for the algorithm.  

	Libraries used 

For basic manipulation:
-numpy  
-pandas  
-argparse   
-pickle  
For tokenisation process:
-nltk  
-future

	Tokenisation of the data 

Definition of text2sentences function: in order to separte the sentence, the words of the corpus, remove the ponctuation elements et stock them into a list.

Definition of stopnumber function: to remove the digit elements of the list.

Definition of dictionnaryoffrequency function: to create a dictionary in which the key is the word of the corpus and the value is its frequency. This function will be used to remove the words that appear too many times.

Definition of stoprecword function: to remove the word for which the frequency is above N% of the total number of words in the corpus. Use of dictionnaryoffrequency function. 

Parameters:   
Percentage to remove word: 0.02

	Generate the pairs

Generate the negative pairs

Definition of init function: to initialize the class.

Definition of wordlistweigthed function: to extract and store the values of our dictionnary which are the frequency of the word. Use for our function probalist. 

Definition of wordlist function: to extract and store the keys of our dictionnary which are the words of our corpus. Use for our function probalist. 

Definition of sumwordfreqweighted function: to compute  Incremental Skip-gram Model with Negative Sampling rate. 

Definition of proba_list function: to create a list with the Incremental Skip-gram Model with Negative Sampling rate for each word.

Generate the positive and negative pairs

Definition of positive_and_negative function: to create our list of pairs; for each positive pair it generates N negative pairs.

parameters:
negative rate: 5  
WinSize: 2

	Optimization Algorithm 

Definition of dico_vectors function: to create an embedding (random vector of size N) for each word. Multiply with 1e-3 to get values small enough for the optimization.

Definition of sigmoid function: to create the function that compute the sigmoid between two embeddings. Use for the computation of the gradients.

Definition of gradientpostar function: to compute the gradient for the target vector update when matched with positive pairs.

Definition of gradientnegtar function: to compute the gradient for the target vector update when matched with negative pairs.

Definition of gradientposcon function: to compute the gradient for the context word vector update.

Definition of gradientnegaid function: to compute the gradient for the negative associated word vector update.

Definition of dotmat function: to actualise the vectors at each iterations. For each pair both vectors are updated. 


Paramaters:
Size of the Vector: 100  
Step for the gradient: 0.01 to get reasonable embeddings after the optimisation. 
negativerates: 5  
Number of epochs: 1, get satisfying results with only one epoch.

	Assess the similarity 

Definition of score_similarity function: to compute the similarity between two word , the larger the output the more similar the 2 word are.

Definition of most_similar function: to find the N most common word of a word.

Parameters:  
Most common words: 5 

	External References

word2vec Explained Deriving Mikolov et al.’s Negative-Sampling Word-Embedding Method - Yoav Goldberg and Omer Levy

Speech and Language, Processing vector semantics part II - Dan Jurafsky and James Martin  

Word to Vectors — Natural Language Processing - Shubham Agarwal

Distributed Representations of Words and Phrases and their Compositionality - Jeffrey Dean, Tomas Mikolov_  

Word2Vec Tutorial Part I: The SkipGram Model - Alex Minnaar 

Text Data: Word Embedding -Yizhou Sun UCLA CS

ATTYASSE Flora  
BENICHOU Vadim  
MICCICHE Carmelo  
VIAL Jennifer 
 




