from django.shortcuts import render
from django.urls import reverse,reverse_lazy
from django.http import HttpResponseRedirect, HttpResponse
import re
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
import re
import numpy as np
import random
import pandas as pd
from Reco.models import RecoUser,Restaurant,menuItem
from Reco.forms import userRegisterFormA,userRegisterFormB
from django.contrib.auth.decorators import login_required,user_passes_test
from django.http import HttpResponseRedirect, HttpResponse
from django.contrib.auth import authenticate,login,logout
# from django.conf import settings
# from settings import STATIC_DIR
# Create your views here.
def initial():
    df_item= pd.read_csv('../RESTROREC/static/datasets/all_items.csv')
    df_rest = pd.read_csv('../RESTROREC/static/datasets/all_rest.csv')
    df_rest.columns = ['Name', 'Rating','Cuisine', 'Address', 'No. of Ratings']
    rest_list=df_rest.values.tolist()
    item_list=df_item.values.tolist()

    for row in rest_list:
        Restaurant.objects.get_or_create(name=row[0],rating=row[1],cuisine=row[2],address=row[3],totalRatings=row[4])

    for row in item_list:
        t=row[6]
        if t<24:
            menuItem.objects.get_or_create(category=row[0],name=row[1],price=row[2],description=row[3],diet=row[4],rating=row[5],restaurantId_id=t)
        else:
            menuItem.objects.get_or_create(category=row[0],name=row[1],price=row[2],description=row[3],diet=row[4],rating=row[5],restaurantId_id=t-1)

def foodfun(fname):
    df= pd.read_csv('../RESTROREC/static/datasets/indian_food2.csv')
    food=[fname,-1,-1,-1,-1,-1]
    df.loc[len(df)] = food
    df['name'] = df['name'].str.lower()              #lower chars
    df['name'] = df['name'].apply( lambda text: text.translate(str.maketrans('', '', string.punctuation)))
    STOPWORDS = set(stopwords.words('english'))  # loading stopwords of english
    df['name'] = df['name'].apply(lambda text: " ".join([word for word in str(text).split() if word not in STOPWORDS]))

    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['name'])                #vectorise the description and calculate tfidf values

    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)      
    idx=len(df)-1

    score_series = pd.Series(cosine_similarities[idx]).sort_values(
        ascending=False)  # retriving values with maximum cosine similarity on the basis of index

    top_index = list(score_series.iloc[1:2].index)[0]
    top_score=list(score_series.iloc[1:2])[0]

    if top_score>=0.5:
        return top_index
    else:
        return -1

def getUser(request):
    u = request.user
    uid=u.id
    ru=RecoUser.objects.get(RUser_id=uid)
    desc=[ru.ingredient,ru.diet,ru.flavour,ru.state,ru.region]
    return desc

def getUser3(request):
    u = request.user
    uid=u.id
    print("USER:",uid)
    ru=RecoUser.objects.get(RUser_id=uid)
    return {'positive':ru.positiveFeature,'negative':ru.negativeFeature}
def getUser4(request):
    u = request.user
    uid = u.id
    ru = RecoUser.objects.get(RUser_id=uid)
    return ru.recentfeature
def getUserObj(request):
    u = request.user
    uid=u.id
    ru=RecoUser.objects.get(RUser_id=uid)
    return ru

def model1(selected_dish):
    nltk.download('stopwords')
    all_rest=Restaurant.objects.all()
    all_item=menuItem.objects.all()
    all_items=[]
    all_rests=[]
    for r in all_rest:
        t=[r.name,r.rating,r.cuisine,r.address,r.totalRatings]
        all_rests.append(t)
    for r in all_item:
        t=[r.category,r.name,r.price,r.description,r.diet,r.rating,r.restaurantId_id]
        all_items.append(t)

    df=pd.DataFrame(all_items,columns=['Category', 'Item Name','Price','Description','Veg/Non-veg','Rating','Restaurant Index'])
    df_rest=pd.DataFrame(all_rests,columns=['Name', 'Rating', 'Cuisine', 'Address', 'No. of Ratings'])
    print(df.dtypes)
    print(df_rest.dtypes)
    df['Description'] = df['Description'].str.lower()
    df['Description'] = df['Description'].apply(  # removing punctuation with empty string
        lambda text: text.translate(str.maketrans('', '', string.punctuation)))

    STOPWORDS = set(stopwords.words('english'))  # loading stopwords of english
    df['Description'] = df['Description'].apply(lambda text: " ".join(  # remving stopwords from description
        [word for word in str(text).split() if word not in STOPWORDS]))

    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(
    1, 2), min_df=0, stop_words='english')
    # vectorise the description and calculate tfidf values
    tfidf_matrix = tfidf.fit_transform(df['Description'])

    # calculte correlation matrix of cosine similarity on the basis of tf idf
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    rdishes = list()               # recommended dishes list

    idx = selected_dish  # getting index number of row

    score_series = pd.Series(cosine_similarities[idx]).sort_values(
        ascending=False)  # retriving values with maximum cosine similarity on the basis of index

    # indices of top 30  dishes
    # first position will be for dishes itself
    top10 = list(score_series.iloc[1:31].index)

    # print(top10)
    ntop5 = []

    for each in top10:
        if(each != idx):
            # appending tuple of (item name,restaurant index) to rdishes
            if (df.iloc[each, [1]][0], df.iloc[each, [6]][0]) not in rdishes:
                rdishes.append((df.iloc[each, [1]][0], df.iloc[each, [6]][0]))
                ntop5.append(each)

    # st.write(ntop5)
    # retrieving veg/nonveg of recommended list
    rveg = df.iloc[idx, [4]][0]

    # retrieving category of recommended list
    rcat = df.iloc[idx, [0]][0]
    rprice = df.iloc[idx, [2]][0]
    score = list()
    for nindex in ntop5:
        # retriving veg/nonveg of dish
        veg = df.iloc[nindex, [4]][0]
        # retriving category of dish
        cat = df.iloc[nindex, [0]][0]
        tempscore = 0
        if(veg == rveg):                                           # adding 3 if veg/nonveg matches
            tempscore = tempscore + 3
        if(rcat == cat):
            # adding 1 if category matches
            tempscore = tempscore + 1
        temprating = df.iloc[nindex, [5]][0]
        tempprice = df.iloc[nindex, [2]][0]

        # assigning score on the basis of rating
        tempscore = tempscore + 1.2*(temprating/5)
        normprice = (tempprice/830)
        # penalise on the basis of price
        tempscore = tempscore - 1.05*abs(normprice-rprice)/rprice

        score.append(tempscore)

    # sorting on the basis of score
    rdishes = [x for _, x in sorted(zip(score, rdishes), reverse=True)]
    # sorting dish indices on the basis of score
    ntop5 = [x for _, x in sorted(zip(score, ntop5), reverse=True)]

    dishname = []

    newname = []
    newridshes = []
    newntop5 = []


    # loop to retrieve dishname
    for dish in rdishes:
        dishname.append(dish[0])

    i = 0

    # loop to append dishes if frequency is 3
    for name in dishname:
        if(newname.count(name) <= 2):
            newname.append(name)
            newridshes.append(rdishes[i])
            newntop5.append(ntop5[i])
        i = i+1


    rdishes = newridshes[0:10]  # taking top 10 dishes
    ntop5 = newntop5[0:10]
    # st.write(rdishes)
    rindex = []  # list for restaurant index

    for dish in rdishes:  # appending restaurant index of dish
        rindex.append(dish[1])

    print(rindex)

    dishes_details = []
    i = 0
    for index in rindex:
        templist = []
        templist.append(rdishes[i][0])  # dishname
        templist.append(df.iloc[ntop5[i], [0]][0])
        templist.append(df.iloc[ntop5[i], [2]][0])
        # # Restaurant name
        templist.append(df_rest.iat[index-1,0])
        templist.append(df_rest.iat[index-1,1])
        templist.append(df_rest.iat[index-1,2])
        templist.append(df_rest.iat[index-1,3])

        i = i+1
        print(templist)
        dishes_details.append(templist)
    return dishes_details

def model2(selected_dish,request):
    df_food= pd.read_csv('../RESTROREC/static/datasets/indian_food2.csv')
    nltk.download('stopwords')
    all_rest=Restaurant.objects.all()
    all_item=menuItem.objects.all()
    all_items=[]
    all_rests=[]
    for r in all_rest:
        t=[r.name,r.rating,r.cuisine,r.address,r.totalRatings]
        all_rests.append(t)
    for r in all_item:
        t=[r.category,r.name,r.price,r.description,r.diet,r.rating,r.restaurantId_id]
        all_items.append(t)

    df=pd.DataFrame(all_items,columns=['Category', 'Item Name','Price','Description','Veg/Non-veg','Rating','Restaurant Index'])
    df_rest=pd.DataFrame(all_rests,columns=['Name', 'Rating', 'Cuisine', 'Address', 'No. of Ratings'])
    # print(df.dtypes)
    # print(df_rest.dtypes)
    df['Description'] = df['Description'].str.lower()
    df['Description'] = df['Description'].apply(  # removing punctuation with empty string
        lambda text: text.translate(str.maketrans('', '', string.punctuation)))

    STOPWORDS = set(stopwords.words('english'))  # loading stopwords of english
    df['Description'] = df['Description'].apply(lambda text: " ".join(  # remving stopwords from description
        [word for word in str(text).split() if word not in STOPWORDS]))

    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(
    1, 2), min_df=0, stop_words='english')
    # vectorise the description and calculate tfidf values
    tfidf_matrix = tfidf.fit_transform(df['Description'])

    # calculte correlation matrix of cosine similarity on the basis of tf idf
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

    rdishes = list()               # recommended dishes list

    idx = selected_dish  # getting index number of row

    score_series = pd.Series(cosine_similarities[idx]).sort_values(
        ascending=False)  # retriving values with maximum cosine similarity on the basis of index

    # indices of top 30  dishes
    # first position will be for dishes itself
    top10 = list(score_series.iloc[1:100].index)

    # print(top10)
    ntop5 = []

    for each in top10:
        if(each != idx):
            # appending tuple of (item name,restaurant index) to rdishes
            if (df.iloc[each, [1]][0], df.iloc[each, [6]][0]) not in rdishes:
                rdishes.append((df.iloc[each, [1]][0], df.iloc[each, [6]][0],each))
                ntop5.append(each)

    # st.write(ntop5)
    # retrieving veg/nonveg of recommended list
    rveg = df.iloc[idx, [4]][0]

    # retrieving category of recommended list
    rcat = df.iloc[idx, [0]][0]
    rprice = df.iloc[idx, [2]][0]
    score = list()
    menuItem_objects=menuItem.objects.all()
    for nindex in ntop5:
        tempscore = 0
        top_index=menuItem_objects[nindex].link 
        if top_index!=-1:
            user=getUser(request)
            user=[f.lower() for f in user]
            food=list(df_food.iloc[top_index])
            # print(food)
            # print(type(food[0]))
            # print(type(food[1]))
            # print(type(food[2]))
            # print(type(food[3]))
            # print(type(food[4]))
            food=[f.lower() for f in food]
            if food[5]==user[4]:
                tempscore= tempscore + 1
            if food[4]==user[3]:
                tempscore= tempscore + 1
            if food[3]==user[2]:
                tempscore= tempscore + 1
            if food[2][0:3]==user[1][0:3]:
                tempscore= tempscore + 1
            fing= food[1].split(', ')
            uing= user[0].split(',')
            for u in uing:
                if u in fing:
                    tempscore=tempscore+5
        # retriving veg/nonveg of dish
        veg = df.iloc[nindex, [4]][0]
        # retriving category of dish
        cat = df.iloc[nindex, [0]][0]
        if(veg == rveg):                                           # adding 3 if veg/nonveg matches
            tempscore = tempscore + 3
        if(rcat == cat):
            # adding 1 if category matches
            tempscore = tempscore + 1
        temprating = df.iloc[nindex, [5]][0]
        tempprice = df.iloc[nindex, [2]][0]

        # assigning score on the basis of rating
        tempscore = tempscore + 1.2*(temprating/5)
        normprice = (tempprice/830)
        # penalise on the basis of price
        tempscore = tempscore - 1.05*abs(normprice-rprice)/rprice

        score.append(tempscore)

    # sorting on the basis of score
    rdishes = [x for _, x in sorted(zip(score, rdishes), reverse=True)]
    # sorting dish indices on the basis of score
    ntop5 = [x for _, x in sorted(zip(score, ntop5), reverse=True)]

    dishname = []

    newname = []
    newridshes = []
    newntop5 = []


    # loop to retrieve dishname
    for dish in rdishes:
        dishname.append(dish[0])

    i = 0

    # loop to append dishes if frequency is 3
    for name in dishname:
        if(newname.count(name) < 2):
            newname.append(name)
            newridshes.append(rdishes[i])
            newntop5.append(ntop5[i])
        i = i+1


    rdishes = newridshes[0:10]  # taking top 10 dishes
    ntop5 = newntop5[0:10]
    # st.write(rdishes)
    rindex = []  # list for restaurant index
    dindex = []
    for dish in rdishes:  # appending restaurant index of dish
        rindex.append(dish[1])
        dindex.append(dish[2])

    print(rindex)

    dishes_details = []
    i = 0
    for j in range(len(rindex)):
        index=rindex[j]
        templist = []
        templist.append(rdishes[i][0])  # dishname
        templist.append(df.iloc[ntop5[i], [0]][0])
        templist.append(df.iloc[ntop5[i], [2]][0])
        # # Restaurant name
        templist.append(df_rest.iat[index-1,0])
        templist.append(df_rest.iat[index-1,1])
        templist.append(df_rest.iat[index-1,2])
        templist.append(df_rest.iat[index-1,3])
        templist.append(dindex[j]+1)
        templist.append(df.iloc[ntop5[i], [5]][0])
        print(dindex[j]+1)
        i = i+1
        #print(templist)
        dishes_details.append(templist)
    return dishes_details

def model3(request):
    featDict=getUser3(request)
    posVector=featDict['positive']
    temp=""
    for i in posVector:
        temp+=(i+", ")
    posVector=temp
    negVector=featDict['negative']
    temp=""
    for i in negVector:
        temp+=(i+", ")
    negVector=temp
    nltk.download('stopwords')
    all_rest=Restaurant.objects.all()
    all_item=menuItem.objects.all()
    all_items=[]
    all_rests=[]
    for r in all_rest:
        t=[r.name,r.rating,r.cuisine,r.address,r.totalRatings]
        all_rests.append(t)
    for r in all_item:
        temp=""
        for i in r.features:
            temp+=(i+", ")
        t=[r.category,r.name,r.price,r.description,r.diet,r.rating,r.restaurantId_id,temp,r.numRatings]
        all_items.append(t)

    df=pd.DataFrame(all_items,columns=['Category', 'Item Name','Price','Description','Veg/Non-veg','Rating','Restaurant Index','Feature_Vector','Num_Ratings'])
    df_rest=pd.DataFrame(all_rests,columns=['Name', 'Rating', 'Cuisine', 'Address', 'No. of Ratings'])
    # print(df.dtypes)
    # print(df_rest.dtypes)
    vectorList=list(df['Feature_Vector'])
    vectorList.append(posVector)
    vectorList.append(negVector)
    vectorList= [v.lower() for v in vectorList]
    # vectorList= [v.apply(
    #     lambda text: text.translate(str.maketrans('', '', string.punctuation))) for v in vectorList]

    STOPWORDS = set(stopwords.words('english'))  # loading stopwords of english
    # vectorList= vectorList.apply(lambda text: " ".join(  # remving stopwords from description
    #     [word for word in str(text).split() if word not in STOPWORDS]))

    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(
    1, 2), min_df=0, stop_words='english')
    # vectorise the description and calculate tfidf values
    tfidf_matrix = tfidf.fit_transform(vectorList)

    # calculte correlation matrix of cosine similarity on the basis of tf idf
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    rdishes = list()               # recommended dishes list
    score_series1 = cosine_similarities[len(vectorList)-2]
    score_series2 = cosine_similarities[len(vectorList)-1]
    score_series=np.subtract(score_series1,score_series2)
    score_series=score_series[0:len(score_series)-2]
    indexList=[i for i in range(len(score_series))]
    vector=list(zip(indexList,score_series))
    score_series = sorted(vector,key=lambda x: (x[1]),reverse=True)
    maxsim=score_series[0][1]
    minsim=score_series[len(score_series)-1][1]
    print(maxsim)
    print(minsim)
    topdict={}
    for i in score_series[0:100]:
        topdict.update({i[0]:i[1]})

    ntop100 = []
    for i in topdict:
        ntop100.append(i)
        
    score = list()
    menuItem_objects=menuItem.objects.all()
    df_food= pd.read_csv('../RESTROREC/static/datasets/indian_food2.csv')
    for nindex in ntop100:
        tempscore = 0
        top_index=menuItem_objects[nindex].link 
        if top_index!=-1:
            user=getUser(request)
            user=[f.lower() for f in user]
            food=list(df_food.iloc[top_index])
            food=[f.lower() for f in food]
            if food[4]==user[3]:
                tempscore= tempscore + 0.3*(maxsim-minsim)
            if food[2]==user[1] and user[1]=="Veg":
                tempscore= tempscore + 10(maxsim-minsim)
            elif food[2]==user[1] and user[1][0:3]=="Non":
                tempscore= tempscore + 0.4*(maxsim-minsim)
        temprating = df.iloc[nindex, [5]][0]
        # tempprice = df.iloc[nindex, [2]][0]

        # assigning score on the basis of rating
        tempscore = tempscore + (maxsim-minsim)*(temprating/5)*0.8
        # normprice = (tempprice/830)
        # # penalise on the basis of price
        # tempscore = tempscore - 1.05*abs(normprice-normprice)/tempprice
        score.append(tempscore)

    ntop100scores=list(zip(ntop100,score))
    sorted_scores = sorted(ntop100scores,key=lambda x: (x[1]),reverse=True)
    ntop20=[x[0] for x in sorted_scores[0:20]]

    dishname = []
    newname = []
    newntop5 = []
    # loop to retrieve dishname
    for i in range(len(ntop20)):
        i=ntop20[i]
        dishname.append(df.iloc[i, [1]][0])

    i = 0
    # loop to append dishes if frequency is 3

    for name in dishname:
        if(newname.count(name) < 2):
            newname.append(name)
            newntop5.append(ntop20[i])
        i = i+1

    final5 = newntop5[0:5]
    final10=final5

    vectorList.pop()
    vectorList.pop()

    # Randomisation
    randindices=[]
    for i in range(100,500):
        randindices.append(score_series[i][0])
    randlist=random.sample(randindices, 50)
    #randlist=np.random.randint(100,500,size=50,dtype=int)
    ranlist50=[]
    for i in randlist:
        ranlist50.append([i,vector[i][1],df.iloc[i, [5]][0],df.iloc[i, [8]][0],df.iloc[i, [4]][0]])

    maxsim=score_series[100][1]
    minsim=score_series[499][1]
    maxcount=np.max([i[3] for i in ranlist50])
    mincount=np.min([i[3] for i in ranlist50])
    score2 = []
    for each in ranlist50:
        index=each[0]   #food index
        count=each[3]   #rating count
        diet=each[4]    #diet
        simscore=each[1] #similarity score
        temprating=each[2]  #rating
        tempscore = 0   
        user=getUser(request)
        user=[f.lower() for f in user]
        if diet==user[1] and user[1]=="Veg":
            tempscore= tempscore + 10(maxsim-minsim)
        elif diet==user[1] and user[1][0:3]=="Non":
            tempscore= tempscore + 0.4*(maxsim-minsim)

        # assigning score on the basis of rating
        tempscore = tempscore + (maxsim-minsim)*(temprating/5)*0.8
        tempscore = tempscore + (maxsim-minsim)*(simscore)*0.4
        tempscore = tempscore - (maxsim-minsim)*((count-mincount)/(maxcount-mincount))*3
        score2.append(tempscore)
    
    ran50Ind=[i[0] for i in ranlist50]
    newranlist50=list(zip(ran50Ind,score2))
    sorted50 = sorted(newranlist50,key=lambda x: (x[1]),reverse=True)
    randtop3=sorted50[0:3]
    randtop3Ind=[i[0] for i in randtop3]
    print(randtop3)
    for i in randtop3Ind:
        final10.append(i)
    
    # Based on recent order
    recentFeatureVector = getUser4(request)
    temp = ""
    for i in recentFeatureVector:
        temp += (i+", ")
    recentFeatureVector = temp

    vectorList.append(recentFeatureVector)
    vectorList = [v.lower() for v in vectorList]
    # vectorList= [v.apply(
    #     lambda text: text.translate(str.maketrans('', '', string.punctuation))) for v in vectorList]

    STOPWORDS = set(stopwords.words('english'))  # loading stopwords of english
    # vectorList= vectorList.apply(lambda text: " ".join(  # remving stopwords from description
    #     [word for word in str(text).split() if word not in STOPWORDS]))

    tfidf = TfidfVectorizer(analyzer='word', ngram_range=(
        1, 2), min_df=0, stop_words='english')
    # vectorise the description and calculate tfidf values
    tfidf_matrix = tfidf.fit_transform(vectorList)

    # calculte correlation matrix of cosine similarity on the basis of tf idf
    cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
    rdishes = list()               # recommended dishes list
    score_series = cosine_similarities[len(vectorList)-1]
    score_series = score_series[0:len(score_series)-1]
    indexList = [i for i in range(len(score_series))]
    vector = list(zip(indexList, score_series))
    score_series = sorted(vector, key=lambda x: (x[1]), reverse=True)
    maxsim = score_series[0][1]
    minsim = score_series[len(score_series)-1][1]
    print(maxsim)
    print(minsim)
    topdict = {}
    for i in score_series[0:30]:
        topdict.update({i[0]: i[1]})

    ntop30 = []
    for i in topdict:
        ntop30.append(i)

    score = list()
    menuItem_objects = menuItem.objects.all()
    for nindex in ntop30:
        tempscore = 0
        top_index = menuItem_objects[nindex].link
        if top_index != -1:
            user = getUser(request)
            user = [f.lower() for f in user]
            food = list(df_food.iloc[top_index])
            food = [f.lower() for f in food]
            if food[4] == user[3]:
                tempscore = tempscore + 0.3*(maxsim-minsim)
            if food[2] == user[1] and user[1] == "Veg":
                tempscore = tempscore + 10*(maxsim-minsim)
            elif food[2] == user[1] and user[1][0:3] == "Non":
                tempscore = tempscore + 0.4*(maxsim-minsim)
        temprating = df.iloc[nindex, [5]][0]
        # tempprice = df.iloc[nindex, [2]][0]

        # assigning score on the basis of rating
        tempscore = tempscore + (maxsim-minsim)*(temprating/5)*0.8
        # normprice = (tempprice/830)
        # # penalise on the basis of price
        # tempscore = tempscore - 1.05*abs(normprice-normprice)/tempprice
        score.append(tempscore)

    ntop30scores = list(zip(ntop30, score))
    sorted_scores = sorted(ntop30scores, key=lambda x: (x[1]), reverse=True)
    ntop20 = [x[0] for x in sorted_scores[0:20]]

    dishname = []
    newname = []
    newntop = []
    # loop to retrieve dishname
    for i in range(len(ntop20)):
        i = ntop20[i]
        dishname.append(df.iloc[i, [1]][0])

    i = 0
    # loop to append dishes if frequency is 3

    for name in dishname:
        if(newname.count(name) < 2):
            newname.append(name)
            newntop.append(ntop20[i])
        i = i+1

    count = 0
    for dish_ind in newntop:
        if final10.count(dish_ind) == 0:
            final10.append(dish_ind)
            count += 1
            if count == 2:
                break

    dishes_details = []
    i = 0
    print(final10)
    for j in final10:
        print("J:",j)
        index=(df.iloc[j, [6]][0])
        print("Index:",index)
        templist = []
        templist.append(df.iloc[j, [1]][0])# name
        templist.append(df.iloc[j, [0]][0])# category
        templist.append(df.iloc[j, [2]][0])# price
        templist.append(df_rest.iat[index-1,0])# restaurant
        templist.append(df_rest.iat[index-1,1])# rest rating
        templist.append(df_rest.iat[index-1,2])# cuisine
        templist.append(df_rest.iat[index-1,3])# address
        templist.append(j+1)  #food index
        templist.append(df.iloc[j, [5]][0]) #food rating
        i = i+1
        #print(templist)
        dishes_details.append(templist)
    return dishes_details

@login_required
def showRest(request):
    if request.method == 'POST':
        # df_food= pd.read_csv('../RESTROREC/static/datasets/indian_food2.csv')
        # allitems = menuItem.objects.all()
        # for items in allitems:
        #     link=items.link
        #     food=list(df_food.iloc[link])
        #     ing2=food[1]
        #     ing = ing2.split(', ')
        #     features=[]
        #     for i in ing:
        #         features.append(i)
        #     if food[3]!='-1':
        #         features.append(food[3])
        #     if food[5]!='-1':
        #         features.append(food[5])
        #     items.features=features
        #     items.save()
        d=request.POST.get("restaurant")
        menu=menuItem.objects.filter(restaurantId=d)
        return render(request, 'main2.html',{'menu':menu})
    else:
        rest=Restaurant.objects.all()
        return render(request, 'main.html',{'rest':rest})

@login_required
def showMenu(request):
    if request.method == 'POST':
        d=int(request.POST.get("restaurant"))
        #dishes=model1(d-1)
        #dishes=model2(d-1,request)
        # user=getUserObj(request)
        # posList=user.positiveFeature
        # featDict=user.features
        # for p in posList:
        #     featDict.update({p:5})
        # user.features=featDict
        # user.save()
        dishes=model3(request)
        return render(request, 'display.html',{'dishes':dishes,'User':request.user})
    else:
        # all_rest=list(Restaurant.objects.all())
        return render(request, 'main.html',)

def registerView(request):
    registered=False
    if request.method=='POST':
        formA=userRegisterFormA(data=request.POST)
        formB=userRegisterFormB(data=request.POST)
        if formA.is_valid() and formB.is_valid():
            docA=formA.save(commit=False)
            #print(formB.cleaned_data['username'])
            docA.set_password(docA.password)
            docA.save()
            docB=formB.save(commit=False)
            docB.RUser=docA
            # userObj=RecoUser.objects.get(RUser.username=docA.username)
            ing=formB.cleaned_data.get('multipleIngredients')
            print("1234567890")
            print(ing)
            ing2=""
            for i in ing:
                ing2+=i+","
            docB.ingredient=ing2
            positiveFeature = []
            positiveFeature.append(docB.region )
            positiveFeature.append(docB.flavour)
            for i in ing:
                positiveFeature.append(i)
            featDict=docB.features
            for p in positiveFeature:
                if p!="N/P":
                    featDict.update({p:5})
            docB.features=featDict
            docB.positiveFeature = positiveFeature
            docB.save()
            registered=True
            m="Registration Successful"
            return HttpResponseRedirect(reverse('loginView'))
        else:
            print(userRegisterFormA.errors,userRegisterFormB.errors)
    else:
        formA=userRegisterFormA()
        formB=userRegisterFormB()
        return render(request, 'register.html',{'formA':formA,'formB':formB})

def loginView(request):
    if request.method=='POST':
        username=request.POST.get('username')
        password=request.POST.get('password')

        docuser=authenticate(username=username,password=password)
        if docuser:
            if docuser.is_active and docuser.is_superuser:
                return(HttpResponse("Invalid login details!"))
                # login(request,docuser)
                # return HttpResponseRedirect(reverse('Blood:adminpanel'))
            elif docuser.is_active:
                login(request,docuser)
                return HttpResponseRedirect(reverse('Reco:showRest'))
            else:
                return HttpResponse("Account not active")
        else:
            print("A login failed")
            return(HttpResponse("Invalid login details!"))
    else:
        return render(request, 'login.html',)


@login_required
def profileView(request):
    u = request.user
    uid=u.id
    ru=RecoUser.objects.get(RUser_id=uid)
    #ing=ru.ingredient
    return render(request, 'myprofile.html',{'User':ru})

@login_required
def orderView(request):
    if request.method == 'POST':
        itemIds=request.POST.getlist('order')
        orders=[]
        for id in itemIds:
            name=menuItem.objects.get(itemId=id).name
            price=menuItem.objects.get(itemId=id).price
            resto=menuItem.objects.get(itemId=id).restaurantId
            orders.append([name,price,resto,id])
        return render(request, 'rateFood.html',{'orders':orders})
    else:
        # form=ratingForm()
        return render(request, 'rateFood.html')

@login_required
def rateView(request):
    if request.method == 'POST':
        orderIds=request.POST.getlist('ids')
        ratings=request.POST.getlist('ratings')
        # print(orderIds)
        # print(ratings)
        user=getUserObj(request)
        recent=[]
        for i in range(len(orderIds)):
            id=orderIds[i]
            item=menuItem.objects.get(itemId=id)
            item.rating=round(item.rating+(int(ratings[i])-item.rating)/(item.numRatings+1),1)
            item.numRatings=item.numRatings+1
            itemfeat=item.features
            for i in itemfeat:
                if i not in recent:
                    recent.append(i)
            item.save()
            if int(ratings[i])>4:
                featDict=user.features
                feat=item.features
                for f in feat:
                    prewt=0
                    if f in featDict:
                        prewt=featDict[f]
                    newwt=0.6*prewt+0.4*int(ratings[i])
                    featDict[f]=newwt
                user.features=featDict
            if int(ratings[i])<2:
                featDict=user.features
                feat=item.features
                for f in feat:
                    prewt=0
                    if f in featDict:
                        prewt=featDict[f]
                    newwt=0.6*prewt-0.4*int(ratings[i])
                    featDict[f]=newwt
                user.features=featDict
            user.save()
            user=getUserObj(request)
            featDict=user.features
            posList=[]
            negList=[]
            for f in featDict:
                if featDict[f]>4:
                    posList.append(f)
                if featDict[f]<2:
                    negList.append(f)
            user.positiveFeature=posList
            user.negativeFeature=negList
            
        user.recentfeatures=recent
        user.save()
        return HttpResponseRedirect(reverse('Reco:showRest'))
    else:
        return render(request, 'rateFood.html',)

@login_required
def logoutView(request):
    logout(request)
    return HttpResponseRedirect(reverse('loginView'))


def errorview(request,pid):
    e="You are not logged in!"
    return render(request,'error.html',{'e':e})

def error_404(request, exception):
   context = {}
   return render(request,'404.html', context)