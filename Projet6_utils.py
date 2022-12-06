import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time



        ## TEXT

import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, wordpunct_tokenize, RegexpTokenizer
from nltk.corpus import words, stopwords

from sklearn import cluster, metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import metrics as kmetrics
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

import tensorflow_hub as hub
import tensorflow_text

import transformers
from transformers import *

import gensim


def get_main_cat(string):
    """Retrieves the main product category from the product category tree"""
    return string.split('>>')[0].strip('[""')


stop_words = set(stopwords.words('english')) #list of stop words used in the tokenizer


        ## CLEANING DOCUMENTS


def tokenize_clean_doc(doc, stem_or_lem='stem', min_size=3, force_alpha=True, to_string=True):
    """Tokenizing and cleaning text, preparing for the bag of words methods (CountVetorizer, tfidf, Word2Vec).
    Returns the tokens in a list or a string.
    
    doc: string to process
    stem_or_lem: defines whether we use a stemmer or a lemmatizer, or nothing if None
    min_size: minimum size of words
    force_alpha: deletes tokens that contain a non-alpha character if True
    to_string: returns the tokens as a joined string if True, or as a list if False"""
    
    #tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(doc)
    
    #stopwords   
    cleaned_tokens = [i for i in tokens if i not in stop_words]
    
    #min size of words
    size_tokens = [i for i in cleaned_tokens if len(i)>=min_size]
    
    #lower and only alpha tokens    
    if force_alpha:
        alpha_tokens = [i.lower() for i in size_tokens if i.isalpha()]
    else:
        alpha_tokens = size_tokens
        
    #stemmer/lemmatizer
    if stem_or_lem == None:
        trans_tokens = alpha_tokens        
    elif stem_or_lem == 'stem':
        transformer = PorterStemmer()
        trans_tokens = [transformer.stem(i) for i in alpha_tokens]
    elif stem_or_lem == 'lem':
        transformer = WordNetLemmatizer()
        trans_tokens = [transformer.lemmatize(i) for i in alpha_tokens]
    else:
        print("Error 'lem_or_stem' argument unknown")
    
    
    #return
    if to_string:
        return " ".join(trans_tokens)
    else:
        return trans_tokens
    
      
    
def tokenize_dl(doc):
    """ Tokenizing text, preparing for the deep learning methods (BERT, USE).
    Returns the tokens as a string"""
    
    tokens = word_tokenize(doc.lower())
    str_tokens = ' '.join(tokens)
    
    return str_tokens
    
    
        ## CLUSTERING & ARI BAG OF WORDS    
    
    
def cluster_bow(feat, n_clust, df, mode, do_pca=True):
    """Clusters vectors with tSNE embedding and Kmeans clustering
    and computes the ARI between the found clusters and the real categories, 
    for the bag of words methods. The Word2Vec model is built and trained in the 'model_w2v()' function below.
    
    feat: vector to cluster
    n_clust: number of clusters to find with the Kmeans clustering
    df: dataframe
    mode: method (CountVectorizer, tfidf, Word2Vec)
    
    Returns the ARI, the tSNE embedding for visual representation and the cluster labels"""
    
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    start_time = time.time()
    
    if mode == 'cvect':
        vect = CountVectorizer(stop_words='english', max_df=0.95, min_df=1)
        vect_trans = vect.fit_transform(df[feat])
    
    elif mode == 'tfidf':
        vect = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=1)
        vect_trans = vect.fit_transform(df[feat])
        
    elif mode == 'w2v':
        vect_trans = model_w2v(feat, df)
        
    #print("vect_trans : ", vect_trans)
    
    
       # PCA
        
    if do_pca:
        if mode !='w2v':

            pca = PCA(n_components=100)
            pca.fit(vect_trans.toarray())    
            vect_trans = pca.transform(vect_trans.toarray())

        else:
            pca = PCA(n_components=100)
            pca.fit(vect_trans)    
            vect_trans = pca.transform(vect_trans)
    
    
    tsne = TSNE(n_components=2, perplexity=60, n_iter=3000, init='random', learning_rate=200, random_state=1)
    X_tsne = tsne.fit_transform(vect_trans)
    
    cls = cluster.KMeans(n_clusters=n_clust, random_state=1)
    cluster_kmeans = cls.fit_predict(X_tsne)
    
    ARI = metrics.adjusted_rand_score(df['main_cat'], cluster_kmeans)
    
    run_time = np.round(time.time() - start_time, 1)
    print("Running time (s) - clustering", mode, ": ", run_time)
    
    if do_pca:        
        df_bow = pd.DataFrame(vect_trans, index=df['image'])
    else:
        df_bow = pd.DataFrame(vect_trans.todense(), index=df['image'])
    
    return ARI, X_tsne, cluster_kmeans, df_bow



def model_w2v(feat, df):
    """Creates and trains the Word2Vec model"""
    
    start_time_tmp = time.time()
    
    #model parameters
    max_length_bow = len(df[feat].max())
    w2v_size=300
    w2v_min_count=1
    w2v_epochs=100
    maxlen = max_length_bow
    sentences = df[feat].to_list()
    sentences = [gensim.utils.simple_preprocess(text) for text in sentences]    
    
    #create and train model    
    print("Creating model...")
    w2v_model = gensim.models.Word2Vec(        
        min_count=w2v_min_count,
        window=w2v_size,
        vector_size=w2v_size,
        seed=1,
        workers=1
    )
    
    print("Training model...")
    w2v_model.build_vocab(sentences)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=w2v_epochs)
    model_vectors = w2v_model.wv
    w2v_words = model_vectors.index_to_key
    
    
    #tokenize
    print("Tokenizing...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    x_sentences = pad_sequences(tokenizer.texts_to_sequences(sentences),
                                                         maxlen=maxlen,
                                                         padding='post') 
    
    
    #Embedding matrix
    print("Embedding...")
    w2v_size = 300
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, w2v_size))
    i=0
    j=0

    for word, idx in word_index.items():
        i +=1
        if word in w2v_words:
            j +=1
            embedding_vector = model_vectors[word]
            if embedding_vector is not None:
                embedding_matrix[idx] = model_vectors[word]
                
    #Embedding model
    input=Input(shape=(len(x_sentences),maxlen),dtype='float64')
    word_input=Input(shape=(maxlen,),dtype='float64')  
    word_embedding=Embedding(input_dim=vocab_size,
                             output_dim=w2v_size,
                             weights = [embedding_matrix],
                             input_length=maxlen)(word_input)
    word_vec=GlobalAveragePooling1D()(word_embedding)  
    embed_model = Model([word_input],word_vec)
    
    print("Executing model...")
    embeddings = embed_model.predict(x_sentences)

    run_time_tmp = np.round(time.time() - start_time_tmp, 1)
    print("Running time (creating and training w2v model) (s): ", run_time_tmp)
    
    return embeddings



        ## CLUSTERING & ARI DL

def cluster_dl(feat, n_clust, df):
    """Clusters vectors for the deep learning methods (BERT, USE)"""
        
    start_time = time.time()
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=2000, init='random', learning_rate=200, random_state=1)
    X_tsne = tsne.fit_transform(feat)
    
    cls = cluster.KMeans(n_clusters=n_clust, random_state=1)
    cluster_kmeans = cls.fit_predict(X_tsne)
    
    ARI = metrics.adjusted_rand_score(df['main_cat'], cluster_kmeans)
    
    run_time = np.round(time.time() - start_time, 1)
    print("Running time (s) - clustering DL: ", run_time)
    
    return ARI, X_tsne, cluster_kmeans


        ## PLOT

def plot_cat(X_tsne, cluster_real, cluster_tsne, true_cat):
    """Visual representation of the tSNE embedding (true categories and clusters found with the previous methods)
    
    X_tsne: tSNE embedding found with previous methods
    cluster_real: real categories
    cluster_tsne: clusters found
    true_cat: name of the true categories"""
    
    fig = plt.figure(figsize=(18,8))
    
    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_tsne[:,0], X_tsne[:,1], c=cluster_real)
    ax.legend(handles=scatter.legend_elements()[0], labels=set(true_cat), loc="best", title="Categories")
    plt.title('Real categories')
    
    ax = fig.add_subplot(122)
    ax.scatter(X_tsne[:,0],X_tsne[:,1], c=cluster_tsne)    
    ax.legend(handles=scatter.legend_elements()[0], labels=set(cluster_tsne), loc="best", title="Clusters")
    plt.title('Clustered categories')
    
    plt.show()
    pass


 

        ## BERT
    
def bert_inp_fct(sentences, bert_tokenizer, max_length):
    """Tokenize and cleans text, preparing for the BERT method"""
    
    input_ids=[]
    token_type_ids = []
    attention_mask=[]
    bert_inp_tot = []

    for sent in sentences:
        bert_inp = bert_tokenizer.encode_plus(sent,
                                              add_special_tokens = True,
                                              max_length = max_length,
                                              padding='max_length',
                                              return_attention_mask = True, 
                                              return_token_type_ids=True,
                                              truncation=True,
                                              return_tensors="tf")
    
        input_ids.append(bert_inp['input_ids'][0])
        token_type_ids.append(bert_inp['token_type_ids'][0])
        attention_mask.append(bert_inp['attention_mask'][0])
        bert_inp_tot.append((bert_inp['input_ids'][0], 
                             bert_inp['token_type_ids'][0], 
                             bert_inp['attention_mask'][0]))

    input_ids = np.asarray(input_ids)
    token_type_ids = np.asarray(token_type_ids)
    attention_mask = np.array(attention_mask)
       
    return input_ids, token_type_ids, attention_mask, bert_inp_tot



    
def feature_BERT_fct(model, model_type, sentences, max_length, b_size):
    """Creates features for the BERT method"""
    
    batch_size = b_size
    batch_size_pred = b_size
    bert_tokenizer = AutoTokenizer.from_pretrained(model_type)
    
    start_time = time.time()

    for step in range(len(sentences)//batch_size) :
        idx = step*batch_size
        input_ids, token_type_ids, attention_mask, bert_inp_tot = bert_inp_fct(sentences[idx:idx+batch_size], 
                                                                      bert_tokenizer, max_length)
        
        text_preprocessed = {"input_word_ids" : input_ids, 
                             "input_mask" : attention_mask, 
                             "input_type_ids" : token_type_ids}
        outputs = model(text_preprocessed)
        last_hidden_states = outputs['sequence_output']
             
        if step ==0 :
            last_hidden_states_tot = last_hidden_states
            last_hidden_states_tot_0 = last_hidden_states
        else :
            last_hidden_states_tot = np.concatenate((last_hidden_states_tot,last_hidden_states))
    
    features_bert = np.array(last_hidden_states_tot).mean(axis=1)
    
    run_time = np.round(time.time() - start_time, 1)
    print("Running time (s) - BERT feature creation: ", run_time)
         
    return features_bert, last_hidden_states_tot


        ## USE

def feature_USE_fct(sentences, b_size):
    """Creates the features for the USE method"""
    
    start_time = time.time()
    
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    
    batch_size = b_size

    for step in range(len(sentences)//batch_size) :
        idx = step*batch_size
        feat = embed(sentences[idx:idx+batch_size])

        if step ==0 :
            features = feat
        else :
            features = np.concatenate((features,feat))
            
    run_time = np.round(time.time() - start_time, 1)
    print("Running time (s) - USE feature creation: ", run_time)
    
    return features









        ## IMAGES
    
    
import PIL
from PIL import Image
from PIL.ImageOps import autocontrast, equalize
from PIL.ImageFilter import MedianFilter

import cv2 as cv

import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array  

# paths to images
PATH_image = "E:/OpenClassrooms/Projet6/Data/flipkart/Images/"
PATH_modified = "E:/OpenClassrooms/Projet6/Data/flipkart/Images/Modified/"



def prepare_image(name):
    """transforms images (contrast, equalize, filter) and reduce size for memory management"""
    
    img_sample = Image.open(PATH_image + name)
    
    img_sample = autocontrast(img_sample)
    img_sample = equalize(img_sample)
    img_sample = img_sample.filter(MedianFilter)
    
    max_size = (600,600)
    img_sample.thumbnail(max_size)
    
    img_sample.save(PATH_modified + name)
    
    
    
    
    
def SIFT(name):
    """computes keypoints and descriptors of images using SIFT algorithm (500 keypoints max)"""
    
    img_sample = cv.imread(PATH_modified + name)
    img_sample = cv.cvtColor(img_sample, cv.COLOR_BGR2GRAY)
    
    sift = cv.SIFT_create(nfeatures=500)
    kp, des = sift.detectAndCompute(img_sample, None)
    
    
    return kp, des



def CNN(name):
    """computes features of an image using the pretrained VGG16 CNN (last 2 layers deleted for feature extraction only and not classification)""" 
    
    image = load_img(PATH_image + name, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    
    features = model.predict(image)
    
    df_temp = pd.DataFrame(features, index=[name])
    
    return df_temp

    
    
    
def cluster_images(n_clust, df, df_vbow, do_pca=True):
    """clusters images based on the features found with previous methods, using kmeans algorithm. Computes ARI between found clusters and real categories. Also used for the clustering of text and images features combined.
    n_clust: number of clusters to find
    df_vbow: dataframe containing the features
    do_pca: whether or not to do a PCA on the dataframe"""
    
    if do_pca:
        
        pca = PCA(n_components=100)
        pca.fit(df_vbow)    
        df_vbow = pca.transform(df_vbow)
        
    tsne = TSNE(n_components=2, perplexity=40, n_iter=2000, init='random', learning_rate=150, random_state=1)
    X_tsne = tsne.fit_transform(df_vbow)
    
    cls = cluster.KMeans(n_clusters=n_clust, random_state=1)
    cluster_kmeans = cls.fit_predict(X_tsne)
    
    ARI = metrics.adjusted_rand_score(df['main_cat'], cluster_kmeans)
    
    return ARI, X_tsne, cluster_kmeans
        
    
    
    
    
    
    
