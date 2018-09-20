from flask import Flask
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import re, nltk, gensim,random
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
import pyLDAvis
import pyLDAvis.sklearn
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from sklearn.cluster import SpectralClustering
from nltk.probability  import FreqDist
import datetime
import sklearn
from kmodes.kprototypes import KPrototypes
import pickle


def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    #print(chunked)
    prev = None
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
            if type(i) == Tree:
                current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue
    return continuous_chunk
	 
	 
	 
def lda(user_last_read_article):
    
    #word_tokenizing
    global sent_to_words

    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        
    data_words = list(sent_to_words(data))


    p_stemmer = PorterStemmer()
    en_stop = get_stop_words('en')

    data_lemmatized = []

    for i in data_words:
        tokens = i
        stopped_tokens = [i for i in tokens if not i in en_stop]
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
        data_lemmatized.append(' '.join(stemmed_tokens))
    
    global vectorizer,data_vectorized,lda_model,lda_output,best_lda_model
    
    if training == 1:
        vectorizer = CountVectorizer(analyzer='word',       
                                 #min_df=10,                        # minimum reqd occurences of a word 
                                 stop_words='english',             # remove stop words
                                 lowercase=True,                   # convert all words to lowercase
                                 token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                 # max_features=30000,             # max number of uniq words
                                )

        data_vectorized = vectorizer.fit_transform(data_lemmatized)

        #Building LDA model
        lda_model = LatentDirichletAllocation(n_components=8,               # Number of topics
                                          max_iter=20,               # Max learning iterations
                                          learning_method='online',   
                                          random_state=100,          # Random state
                                          batch_size=2,            # n docs in each learning iter
                                          evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                          n_jobs = -1,               # Use all available CPUs
                                         )
        lda_output = lda_model.fit_transform(data_vectorized)


        search_params = {'n_components': [3,5,7,9], 'learning_decay': [.5, .7, .9]}

        # Init the Model
        lda = LatentDirichletAllocation()

        # Init Grid Search Class
        model = GridSearchCV(lda, param_grid=search_params)

        # Do the Grid Search
        model.fit(data_vectorized)

        # Printing params for best model among all the generated ones
        # Best Model
        best_lda_model = model.best_estimator_
        
        outfile = open('vectorizer.pickled','wb')
        pickle.dump(vectorizer,outfile)
        outfile.close()
        outfile = open('data_vectorized.pickled','wb')
        pickle.dump(data_vectorized,outfile)
        outfile.close()
        outfile = open('lda_output.pickled','wb')
        pickle.dump(lda_output,outfile)
        outfile.close()
        outfile = open('lda_model.pickled','wb')
        pickle.dump(lda_model,outfile)
        outfile.close()
        outfile = open('best_lda_model.pickled','wb')
        pickle.dump(best_lda_model,outfile)
        outfile.close()
        
    else :
        
        infile = open('vectorizer.pickled','rb')
        vectorizer = pickle.load(infile)
        infile.close()
        infile = open('data_vectorized.pickled','rb')
        data_vectorized = pickle.load(infile)
        infile.close()
        infile = open('lda_output.pickled','rb')
        lda_output = pickle.load(infile)
        infile.close()
        infile = open('lda_model.pickled','rb')
        lda_model = pickle.load(infile)
        infile.close()
        infile = open('best_lda_model.pickled','rb')
        best_lda_model = pickle.load(infile)
        infile.close()


    #dominant topic in each doc

    # Create Document - Topic Matrix
    lda_output = best_lda_model.transform(data_vectorized)

    # column names
    topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]

    # index names
    docnames = ["Doc" + str(i) for i in range(len(data))]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['dominant_topic'] = dominant_topic

    df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")

    # defining topic keywords 
    # Topic-Keyword Matrix
    df_topic_keywords = pd.DataFrame(best_lda_model.components_)

    # Assign Column and Index
    df_topic_keywords.columns = vectorizer.get_feature_names()
    df_topic_keywords.index = topicnames

    # View
    df_topic_keywords.head()

    #get top 15 keywords for each doc


    # Show top n keywords for each topic
    def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
        keywords = np.array(vectorizer.get_feature_names())
        topic_keywords = []
        for topic_weights in lda_model.components_:
            top_keyword_locs = (-topic_weights).argsort()[:n_words]
            topic_keywords.append(keywords.take(top_keyword_locs))
        return topic_keywords

    topic_keywords = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)  

    #Given a piece of text, predicting the topic in document

    def predict_topic(text):
        global sent_to_words

        mytext_2 = list(sent_to_words(text))
        #print(mytext_2)

        mytext_3 =[]

        for i in mytext_2 :

            tokens=i
            stopped_tokens = [i for i in tokens if not i in en_stop]
            #print(stopped_tokens)
            stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
            #print(stemmed_tokens)
            mytext_3.append(' '.join(stemmed_tokens))
            #print(mytext_3)

            mytext_4 = vectorizer.transform(mytext_3)

        topic_probability_scores = best_lda_model.transform(mytext_4)
        topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), :].values.tolist()
        return topic, topic_probability_scores



    #Given a piece of Text, predicting the documents that are related to it most closely

    from sklearn.metrics.pairwise import euclidean_distances

    def similar_documents(text, doc_topic_probs, documents = data, top_n=2, verbose=False):
        topic, x  = predict_topic(text)
        dists = euclidean_distances(x.reshape(1, -1), doc_topic_probs)[0]
        doc_ids = np.argsort(dists)[:top_n]
        return doc_ids, np.take(documents, doc_ids)

    arr=[]
    arr.append(user_last_read_article)
    doc_ids, docs = similar_documents(text=arr, doc_topic_probs=lda_output, documents = data, top_n=2, verbose=True)
    result_api.append(doc_ids[0])
    result_api.append(doc_ids[1])
    print(result_api)



def tfidf():
    stopwords_en = get_stop_words('en')
    #stop words of english are removed by the below function
    def preprocessing(raw):
        wordlist=nltk.word_tokenize(raw)
        text=[w.lower() for w in wordlist if w not in stopwords_en]
        return text


    similarity_scores = []
    doc_number = []

    #We need to find documents that are similar to sample_doc from the corpus built above - data .

    #Sample Usecase
    #Magnum,Qwality walls,ice cream,tasty and sweet
    #fair and lovely makes skin smooth and makes you look young
    #cleaning,dish washing, Vim liquid,lemon,Lifebuoy cleans hands,germs

    sample_doc = " ".join(get_continuous_chunks(user_last_read_article))
    word_set= {'hindustan unilever'}

    for doc in data:
        word_set=word_set.union(set(preprocessing(doc)))
    word_set=word_set.union(set(preprocessing(sample_doc)))


    i=0
    for doc in data:
        text1=preprocessing(doc)
        text2=preprocessing(sample_doc)

        #TF Calculations

        freqd_text1=FreqDist(text1)
        text1_length=len(text1)

        text1_tf_dict = dict.fromkeys(word_set,0)
        for word in text1:
            text1_tf_dict[word] = freqd_text1[word]/text1_length


        freqd_text2=FreqDist(text2)
        text2_length=len(text2)

        text2_tf_dict = dict.fromkeys(word_set,0)
        for word in text2:
            text2_tf_dict[word] = freqd_text2[word]/text2_length


        #IDF Calculations

        text12_idf_dict=dict.fromkeys(word_set,0)
        text12_length = len(data)
        for word in text12_idf_dict.keys():
            if word in text1:
                text12_idf_dict[word]+=1
            if word in text2:
                text12_idf_dict[word]+=1

        import math
        for word,val  in text12_idf_dict.items():
            if val == 0 :
                val=0.01
                text12_idf_dict[word]=1+math.log(text12_length/(float(val)))


        #TF-IDF Calculations

        text1_tfidf_dict=dict.fromkeys(word_set,0)
        for word in text1:
            text1_tfidf_dict[word] = (text1_tf_dict[word])*(text12_idf_dict[word])

        text2_tfidf_dict=dict.fromkeys(word_set,0)
        for word in text2:
            text2_tfidf_dict[word] = (text2_tf_dict[word])*(text12_idf_dict[word])


        #Finding cosine distance which ranges between 0 and 1. 1 implies documents are similar since cos-inverse(0)=1 that is 
        #vectors are collinear.cos-inverse(90)=1 that is vectors are othogonal to each other implying compltely dissimilar.

        v1=list(text1_tfidf_dict.values())
        v2=list(text2_tfidf_dict.values())

        similarity= 1- nltk.cluster.cosine_distance(v1,v2)
        doc_number.append(int(i))
        similarity_scores.append(float(format(similarity*100,'4.2f')))
        i=i+1

        #print("Similarity Index = {:4.2f} % ".format(similarity*100))

    #print('Document IDs : ' + ', '.join(str(e) for e in doc_number))    
    #print('Similarity % : ' + ', '.join(str(e) for e in similarity_scores))
    
    #Based on similarity scores computed previously sort the document indices in ascending leading to most similar document indices
    #present at the end of array
    sorted_doc_list = [doc_number for _,doc_number in sorted(zip(similarity_scores,doc_number))]


    #printing top 3 documents which are most similar to sample_doc
    j = 0
    n=3
    for doc in reversed(sorted_doc_list):
        #print('\n\n',data[doc][:1000])
        result_api.append(doc+1)
        j=j+1
        if j==n :
            break
            
    print(result_api)	
	 

	 
def spectral():
    #Profiling the users based on the articles they read.

    similarity_matrix=[]
    similarity_matrix=df_article_read.values.tolist()
    similarity_matrix
    mat = np.matrix(similarity_matrix)
    output=[]
    
    if training == 1:
        output=(SpectralClustering(4).fit_predict(mat)).tolist()
        outfile = open('spectral.pickled','wb')
        pickle.dump(output,outfile)
        outfile.close()
    else :
        infile = open('spectral.pickled','rb')
        output = pickle.load(infile)
        infile.close()
    
    def trending_article(user_index):
    
        #identifying the cluster to which this user belongs to
        cluster_id=output[user_index]
        #print("cluster id={}".format(cluster_id))

        #creating a list of size number of articles with all zeros to represent the count of times an article is read in cluster
        result=[]
        article_ids=[]
        j=1
        for i in df.iloc[0].tolist():
            result.append(0)
            article_ids.append(j)
            j=j+1

        #print(result)
        #print(article_ids)

        #identifying the article which is trending/read-most number of time in this cluster
        j=0
        for cluster_value in output:
            if cluster_value == cluster_id :
                arr=df.iloc[j].tolist()

                k=0
                for ele in arr:
                    if ele == 1:
                        result[k]=result[k]+1
                    k=k+1

            j=j+1

        #print("Read Counts= {}".format(result))

        sorted_list=[article_ids for _,article_ids in sorted(zip(result,article_ids))]

        response=[]
        j=0
        for i in reversed(sorted_list):
            result_api.append(i)
            j=j+1
            if j==3 :
                break

     
    trending_article(user_index)
    print(result_api)	 
	 
	 

def kprototypes():
    
    df_personal.to_csv('users_sliced', sep=',', encoding='utf-8',index=False)
    X = np.genfromtxt('users_sliced', dtype=object, delimiter=',')[1:, 1:]
    X[:, :1] = X[:,:1 ].astype(float)
    
    costs=[]
    global kproto

    if training ==1 :
        for k in range(2,15):

            kproto = KPrototypes(k, init='Cao', verbose=2)
            clusters = kproto.fit_predict(X,categorical=[1,2])
            costs.append(kproto.cost_)

        k=costs.index(min(costs))+2
        kproto = KPrototypes(k, init='Cao', verbose=2)
        clusters = kproto.fit_predict(X,categorical=[1,2])
        
        outfile = open('kprototypes.pickled','wb')
        pickle.dump(kproto,outfile)
        outfile.close()
    else :
        infile = open('kprototypes.pickled','rb')
        kproto = pickle.load(infile)
        infile.close()
        
    
    def keywords_generation(user_index):
    
        cluster_id=kproto.labels_.tolist()[user_index]
        age=kproto.cluster_centroids_[0][cluster_id][0]
        gender=kproto.cluster_centroids_[1][cluster_id][0].decode("utf-8")
        country=kproto.cluster_centroids_[1][cluster_id][1].decode("utf-8")

        num=random.randint(1,3)
        keywords=''

        if num == 1 :
            if age >=21 and age <=30 :
                keywords=keywords + df_rules['value'][df_rules['key'].tolist().index('21-30')]
            if age >= 31 and age <=40 :
                keywords=keywords + df_rules['value'][df_rules['key'].tolist().index('31-40')]
            if age>=41 and age <=50:
                keywords=keywords + df_rules['value'][df_rules['key'].tolist().index('41-50')]
            if age>=51 and age<=60:
                keywords=keywords + df_rules['value'][df_rules['key'].tolist().index('51-60')]
        if num == 2 :
            if gender == 'M' :
                keywords=keywords + df_rules['value'][df_rules['key'].tolist().index('M')]
            if gender == 'F' :
                keywords=keywords + df_rules['value'][df_rules['key'].tolist().index('F')]

        if num == 3 :
            details=df_rules['value'][df_rules['key'].tolist().index(country)]
            mydate = datetime.datetime.now()
            curr_month=mydate.strftime("%B")

            if details.index(curr_month) <= details.index('Winter'):
                keywords=keywords + df_rules['value'][df_rules['key'].tolist().index('SUMMER')]
            else :
                keywords=keywords + df_rules['value'][df_rules['key'].tolist().index('WINTER')]

        return keywords

    lda(keywords_generation(user_index))


	
	
	 
app = Flask(__name__)

@app.route("/recommendations/<int:userid>",methods=['GET'])
def hello(userid):
    
	global data,df,df_article_read,df_personal,df_rules,user_index,user_last_read_article,result_api,training
	
	df=pd.read_excel("articles.xlsx")
	data=[]
	
	for v in df['content']:
		data.append(v)

	df_article_read=pd.read_excel("users.xlsx")
	df_article_read.drop(['uid', 'age','gender','country','last_read_article'], axis=1, inplace=True) 

	df=pd.read_excel("users.xlsx")
	df_personal=df[['uid','age','gender','country']]

	df_rules=pd.read_excel("rules.xlsx")


	#Getting user statistics
	uid=userid
	training=0
	user_index=df_personal['uid'].tolist().index(uid)
	user_last_read_article_id=df['last_read_article'].tolist()[user_index]
	user_last_read_article=data[user_last_read_article_id-1]

	#result_api will contain article ids of articles that are to be recommended
	result_api=[]

	tfidf()
	lda(user_last_read_article)
	spectral()
	kprototypes()

	#removing articles which he has already read 
	user_read_articles=df_article_read.iloc[user_index].tolist()

	j=0
	for i in user_read_articles:
		
		if i==1 and (j+1) in result_api:
			result_api.remove(j+1)

	print("Articles Recommended: {}".format(result_api))
	result_api=list(set(result_api))
	temp="{\"Recommended_Article_IDs\":\"" + ','.join(str(x) for x in result_api) + "\"}"
	return temp


if __name__ == '__main__':
    app.run(debug=True)