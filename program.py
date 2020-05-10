import tkinter.tix
import tkinter.ttk
import tkinter.messagebox
import tkinter as tk
import math
import sys
import nltk
import re
import os
import numpy as np
import random
import string
import tkinter
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pprint, inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
open
from sklearn import tree
from tkinter import tix
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
from collections import Counter
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB# Mengaktifkan/memanggil/membuat fungsi klasifikasi Naive bayes
from tkinter.filedialog import askopenfilename
global Import_Corpus_TextCorpus
global jumlahcorpus,carii1,v
global hasilstemmingquery
global vdok
import math
global data
global ne
global  optionregex
optionregex=[]
matplotlib.use("TkAgg")
frame = None
canvas = None
ax = None
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

stop_factory = StopWordRemoverFactory()
more_stopword = ['dengan', 'ia','bahwa','oleh']
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()
data = stop_factory.get_stop_words()+more_stopword
stopword = stop_factory.create_stop_word_remover()





def CurSelet(evt):
    global value
    value=str(( listboxDatacorpus.get(ANCHOR)))
    print (value)
    """
    path=value
    teg=[' ']
    teksquran.delete('1.0',END)
    f = open(os.path.basenamepath)), "r")
    for huruf in f:   
        teksquran.insert(END,huruf) 
    f.close()
    """

    
def ecludianWordNet():
    global finalecludianWordnet
    finalecludianWordnet=[]
    tempEcludian=[]
    tempEcludian2=[]
    Select=[]
    #print(hasilpembobotan1)
    panjangDataTraining=len(dataTrainingKnnWordnet)
    #print(hasilpembobotan_Testing1_Testing)
    panjangDataTesting=len(dataTestingKnnWordnet)
    i=0
    ii=0
    print(dataTestingKnnWordnet)
    for i in range(panjangDataTesting):
        print(dataTestingKnnWordnet[i][0])
        print(dataTestingKnnWordnet[i][1])
        print("--------------- Ecludian")
        for ii in range(panjangDataTraining):
            print(ii)
            print(dataTrainingKnnWordnet[ii][0])
            print(dataTrainingKnnWordnet[ii][1])
            temp1=(dataTrainingKnnWordnet[ii][0]-dataTestingKnnWordnet[i][0])**2
            temp2=(dataTrainingKnnWordnet[ii][1]-dataTestingKnnWordnet[i][1])**2
            print(temp1)
            print(temp2)
            allTemp=temp1+temp2
            hasil=math.sqrt(allTemp)
            print("+++++")
            print(hasil)
            print(dataTrainingKnnWordnet[ii][2])
            tree42copy.insert('', 'end', text=ii+1, values=(temp1,temp2,hasil,dataTrainingKnnWordnet[ii][2]))
            tempEcludian.append(hasil)
            tempEcludian.append(dataTrainingKnnWordnet[ii][2])
            tempEcludian.append(ii)
            tempEcludian2.append(tempEcludian)
            tempEcludian=[]
            ii=ii+1
        print(tempEcludian2)
        tempEcludian2sort=tempEcludian2
        tempEcludian2=[]
        tempEcludian2sort=sorted(tempEcludian2sort, key=lambda x: str(x[0]))
        print(tempEcludian2sort)
        iii=0
        for iii in range(len(tempEcludian2sort)):
            if iii<3:
               Select.append(tempEcludian2sort[iii][1]) 
            else:
                break
                
        print(i)
        print(ii)
        print(Select)
        countNetral = Select.count('netral')
        countPositive = Select.count('positif')
        countNegative = Select.count('negatif')
        print(countNetral)
        print(countPositive)
        print(countNegative)
        if countNegative==countPositive and countNegative==countNetral :
            print("netral")
            finalecludianWordnet.append("netral")
        else:    
            print(most_frequent(Select))
            finalecludianWordnet.append(most_frequent(Select))
        Select=[]
        print("----------Next-----------")
        #items = [(1, 'A number'), ('a', 'A letter'), (2, 'Another number')]
        #print(sorted(tempEcludian2, key=lambda x: str(x[0])))
        i=i+1
    print("-------------SORTIR---------------")
    print("--end---")
    #print(finalecludian11)
    #matrix()
    

def ecludian1():
    global finalecludian1
    finalecludian1=[]
    tempEcludian=[]
    tempEcludian2=[]
    Select=[]
    #print(hasilpembobotan1)
    panjangDataTraining=len(dataTrainingKnn)
    #print(hasilpembobotan_Testing1_Testing)
    panjangDataTesting=len(dataTestingKnn)
    i=0
    ii=0
    print(dataTestingKnn)
    for i in range(panjangDataTesting):
        print(dataTestingKnn[i][0])
        print(dataTestingKnn[i][1])
        print("--------------- Ecludian")
        for ii in range(panjangDataTraining):
            print(ii)
            print(dataTrainingKnn[ii][0])
            print(dataTrainingKnn[ii][1])
            temp1=(dataTrainingKnn[ii][0]-dataTestingKnn[i][0])**2
            temp2=(dataTrainingKnn[ii][1]-dataTestingKnn[i][1])**2
            print(temp1)
            print(temp2)
            allTemp=temp1+temp2
            hasil=math.sqrt(allTemp)
            print("+++++")
            print(hasil)
            print(dataTrainingKnn[ii][2])
            tree22copy.insert('', 'end', text=ii+1, values=(temp1,temp2,hasil,dataTrainingKnn[ii][2]))
            tempEcludian.append(hasil)
            tempEcludian.append(dataTrainingKnn[ii][2])
            tempEcludian.append(ii)
            tempEcludian2.append(tempEcludian)
            tempEcludian=[]
            ii=ii+1
        print(tempEcludian2)
        tempEcludian2sort=tempEcludian2
        tempEcludian2=[]
        tempEcludian2sort=sorted(tempEcludian2sort, key=lambda x: str(x[0]))
        print(tempEcludian2sort)
        iii=0
        for iii in range(len(tempEcludian2sort)):
            if iii<3:
               Select.append(tempEcludian2sort[iii][1]) 
            else:
                break
                
        print(i)
        print(ii)
        print(Select)
        countNetral = Select.count('netral')
        countPositive = Select.count('positif')
        countNegative = Select.count('negatif')
        print(countNetral)
        print(countPositive)
        print(countNegative)
        if countNegative==countPositive and countNegative==countNetral :
            print("netral")
            finalecludian1.append("netral")
        else:    
            print(most_frequent(Select))
            finalecludian1.append(most_frequent(Select))
        Select=[]
        print("----------Next-----------")
        #items = [(1, 'A number'), ('a', 'A letter'), (2, 'Another number')]
        #print(sorted(tempEcludian2, key=lambda x: str(x[0])))
        i=i+1
    print("-------------SORTIR---------------")
    print("--end---")
    #print(finalecludian11)
    #matrix()
    
def most_frequent(List): 
    return max(set(List), key = List.count) 
 
  
        
def most_frequent(List): 
    return max(set(List), key = List.count) 

def getDataTestingWordNet(listTweet_Training_Wordnet,listTweet_Testing_Wordnet):
    
    RunningDataTestingWordNet(listTweet_Training_Wordnet,listTweet_Testing_Wordnet)

def RunningDataTestingWordNet(listTweet_Training_Wordnet,listTweet_Testing_Wordnet):
    global dataTestingKnnWordnet
    global dataTestingDecitionWordnet
    global dataTestingBayesWordnet
    global labelingTesting_Wordnert
    global temp_Testing_listTweet
    global treeview
    global treeview2
    labelingTesting_Wordnert=[]
    kalkulasi_Testing=[]
    Kalkulasi1_Testing=[]
    Kalkulasi2_Testing=[]
    Katadasar_Testing=[]
    temp_Testing_listTweet=[]
    Label_Katadasar_Testing=[]
    New_Kalkulasi_Testing=[]
    New_Kalkulasi_Testing_1=[]
    temp_Testing=[]
    
    treeview =tree1
    f = open(os.path.basename(str('label2.csv')), "r")
    i=0
    for huruf in f:
        Katadasar_Testing.append(huruf.split()[0])
        Label_Katadasar_Testing.append(huruf.split()[1])
        i=i+1
    Katadasar_Testing_1= '  '.join(Katadasar_Testing)
    f.close()
    print(Katadasar_Testing)
    print(Label_Katadasar_Testing)
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    Katadasar_Testing_1 = stemmer.stem(Katadasar_Testing_1)
    """
    print('List Kata Adalah :')
    print(Katadasar_Testing_1.split())
    print('List Tweet Adalah :')
    print(tweet)
    """
    i=0
    for i in range(len(listTweet_Testing_Wordnet)):
        temp_Testing_listTweet.append(listTweet_Testing_Wordnet[i].split())
        i=i+1
      
    print(len(listTweet_Testing_Wordnet))
    print(temp_Testing_listTweet)
    print(temp_Testing_listTweet[0])

    i=0
    ii=0
    iii=0
    calculate=0
    for i in range(len(Katadasar_Testing_1.split())):
        print(Katadasar_Testing_1.split()[i])
        kalkulasi_Testing.append(Katadasar_Testing_1.split()[i])
        for ii in range(len(temp_Testing_listTweet)):
              print("      "+'  '.join(temp_Testing_listTweet[ii]))
              print(" ----------------------------------")
              for iii in range(len(temp_Testing_listTweet[ii])):
                   #print("      "+temp_Testing_listTweet[ii][iii])
                   if Katadasar_Testing_1.split()[i] in temp_Testing_listTweet[ii][iii]:
                       xx='  '.join(temp_Testing_listTweet[ii][iii])
                       #print( "Ada di "+temp_Testing_listTweet[ii][iii])
                       calculate=calculate+1
                   else:
                       xx='  '.join(temp_Testing_listTweet[ii][iii])
                       #print( "Tidak Ada di "+temp_Testing_listTweet[ii][iii])
                       
                   iii=iii+1
              #print("Jumlah : ")
              #print(calculate)
              kalkulasi_Testing.append(calculate)
              calculate=0
              ii=ii+1
        #print(kalkulasi_Testing)
        Kalkulasi1_Testing.append(kalkulasi_Testing)
        kalkulasi_Testing=[]
        i=i+1
    #print(Kalkulasi1_Testing)
    #print(len(Kalkulasi1_Testing))

    i=0
    ii=0
    count=0
    for i in range(len(Kalkulasi1_Testing)):
        print("---------------")
        print()
        print(Kalkulasi1_Testing[i])
        for ii in range(len(Kalkulasi1_Testing[i])):
            #print(Kalkulasi1_Testing[i][ii])
            if ii>0:
                if str(Kalkulasi1_Testing[i][ii]) in str(0):
                   print(str(Kalkulasi1_Testing[i][ii]) + " Sama "+str(0))
                else:
                   print(str(Kalkulasi1_Testing[i][ii])+ " Tidak Sama "+str(0))
                   count=count+1
            ii=ii+1
            #print(Kalkulasi1_Testing[i][ii])
        #print(count-1)
        
        
        Kalkulasi1_Testing[i].append(count)
        if str(count) in str(0):
            hasilLog= math.log(1,10)
            Kalkulasi1_Testing[i].append(hasilLog)
            #print(len(tweet))
            #print("Nol")
        else:
            Log=len(listTweet_Testing_Wordnet)/(count)
            #print(len(tweet))
            hasilLog= math.log(Log,10)
            Kalkulasi1_Testing[i].append(hasilLog)
            #print("Tidak Nol")
        
        #print("PanjangTweet")
        nilai=len(listTweet_Testing_Wordnet)
        #print(tweet)
        print(Kalkulasi1_Testing[i])
        Kalkulasi2_Testing.append(Kalkulasi1_Testing[i])
        count=0
        i=i+1

    i=0
    ii=0
    
    for i in range(len(Kalkulasi2_Testing)):
        print()
        print("FINAL")
        print(Kalkulasi2_Testing[i])
        for ii in range(len(Kalkulasi2_Testing[i])):
            panj=len(Kalkulasi2_Testing[i])
            if ii>0:
                if ii<len(Kalkulasi2_Testing[i])-2:
                    #print(Kalkulasi2_Testing[i][ii])
                    Kalkulasi2_Testing[i][ii]=Kalkulasi2_Testing[i][ii]*Kalkulasi2_Testing[i][panj-1]
                    
                    #print(Kalkulasi2_Testing[i][ii])
                    #print("//")
            ii=ii+1
        print("Afetr")
        Kalkulasi2_Testing[i].append(Label_Katadasar_Testing[i])
        print(Kalkulasi2_Testing[i])
        temp_Testing.append(Kalkulasi2_Testing[i])
        i=i+1
    print()
    print()
    print(temp_Testing)
    print(listTweet_Testing_Wordnet)
    print(len(listTweet_Testing_Wordnet))

    i=0
    ii=0
    Label_Katadasar_Testing_final=[]
    Label_Katadasar_Testing_final2=[]
    negatif_Testing=[]
    count_positif_Testing_Testing=[]
    count_negatif_Testing_Testing=[]
    hitung1=0
    hitung2=0

    for i in range(len(temp_Testing)):
        #print(temp_Testing[i])
        print("-----")
        Label_Katadasar_Testing_final.append(temp_Testing[i][0])
        Label_Katadasar_Testing_final.append(temp_Testing[i][len(temp_Testing[i])-1])
        print(temp_Testing[i][0])
        print(temp_Testing[i][len(temp_Testing[i])-1])
        for ii in range(len(temp_Testing[i])):
            if ii>0 :
                if ii<len(temp_Testing[i])-3:
                    Label_Katadasar_Testing_final.append(temp_Testing[i][ii])  
                    print(temp_Testing[i][ii])
            ii=ii+1
        Label_Katadasar_Testing_final2.append(Label_Katadasar_Testing_final)
        Label_Katadasar_Testing_final=[]
        """
        if str(temp_Testing[len(temp_Testing)-1]) in "positif_Testing":
            print(str(temp_Testing[ii]) + "positif_Testing")
        else:
            print(str(temp_Testing[len(temp_Testing)-1]) + "negatif_Testing")
        """
        
        i=i+1

    
    print("Final")
    print(Label_Katadasar_Testing_final2)
    panjangResult=len(Label_Katadasar_Testing_final2)-2
    print(len(Label_Katadasar_Testing_final2)-2)
    
    print("List Tweet")
    print(listTweet_Testing_Wordnet)
   
    
    
    i=0
    ii=0
    iii=0
    xi=0
    xii=0
    count=0
    count2=0
    positif_Testing=[]
    negatif_Testing=[]
    positif_Testing1=[]
    negatif_Testing1=[]
    hasilpembobotan_Testing=[]
    global hasilpembobotan_Testing1_Testing
    hasilpembobotan_Testing1_Testing=[]
    for i in range(len(listTweet_Testing_Wordnet)):
        print("Tweet ke"+'  '.join(str(i)))
        #hasilpembobotan_Testing.append('  '.join(str(i)))
        for ii in range(len(Label_Katadasar_Testing_final2)):
            if Label_Katadasar_Testing_final2[ii][1] in "negatif":
                print(str(Label_Katadasar_Testing_final2[ii][i+2])+" negatif")
                negatif_Testing.append(Label_Katadasar_Testing_final2[ii][i+2])
            if Label_Katadasar_Testing_final2[ii][1] in "positif":
                print(str(Label_Katadasar_Testing_final2[ii][i+2])+" ")
                positif_Testing.append(Label_Katadasar_Testing_final2[ii][i+2])
            ii=ii+1
        print("-----")
        print(positif_Testing)
        print(sum(positif_Testing))
        hasilpembobotan_Testing.append(sum(positif_Testing))
        positif_Testing1.append(positif_Testing)
        print(negatif_Testing)
        print(sum(negatif_Testing))
        hasilpembobotan_Testing.append(sum(negatif_Testing))
        negatif_Testing1.append(negatif_Testing)
        if sum(positif_Testing) > sum(negatif_Testing):
            hasilpembobotan_Testing.append("positif")
            print("positif")
        else:
            if sum(positif_Testing) < sum(negatif_Testing):
                hasilpembobotan_Testing.append("negatif")
                print("negatif")
            else:
                hasilpembobotan_Testing.append("netral")
                print("netral")
                
        print("-----")
        hasilpembobotan_Testing1_Testing.append(hasilpembobotan_Testing)
        hasilpembobotan_Testing=[]
        positif_Testing=[]
        negatif_Testing=[]
        i=i+1
  
    
    print("Finnaly Testing") 
    print(hasilpembobotan_Testing1_Testing)
    dataTestingKnnWordnet=hasilpembobotan_Testing1_Testing
    dataTestingDecitionWordnet=hasilpembobotan_Testing1_Testing
    dataTestingBayesWordnet=hasilpembobotan_Testing1_Testing
    i=0
    for i in range(len(hasilpembobotan_Testing1_Testing)):
        labelingTesting_Wordnert.append(hasilpembobotan_Testing1_Testing[i][2])
        tree41copy.insert('', 'end', text=i+1, values=(hasilpembobotan_Testing1_Testing[i][0],hasilpembobotan_Testing1_Testing[i][1],hasilpembobotan_Testing1_Testing[i][2]))
        i=i+1
    ecludianWordNet()

def getDataTesting(listTweet_Training,listTweet_Testing):
    print("tanpa wordnet")
    """
    #print(tweet_2)
    global testing_2
    testing_2=[]
    rang=len(listTweet) 
    i=0
    for i in range(rang):
        testing_2.append(listTweet[i])
        i=i+1
    """
    print(listTweet_Testing)
    RunningDataTesting(listTweet_Training,listTweet_Testing)
   

def RunningDataTesting(listTweet_Training,listTweet_Testing):
    global dataTestingKnn
    global dataTestingDecition
    global dataTestingBayes
    global labelingTesting
    global dataTestingDecition
    global dataTestingBayes
    global temp_Testing_listTweet
    global treeview
    global treeview2
    labelingTesting=[]
    kalkulasi_Testing=[]
    Kalkulasi1_Testing=[]
    Kalkulasi2_Testing=[]
    Katadasar_Testing=[]
    temp_Testing_listTweet=[]
    Label_Katadasar_Testing=[]
    New_Kalkulasi_Testing=[]
    New_Kalkulasi_Testing_1=[]
    temp_Testing=[]
    
    treeview =tree1
    f = open(os.path.basename(str('label.csv')), "r")
    i=0
    for huruf in f:
        Katadasar_Testing.append(huruf.split()[0])
        Label_Katadasar_Testing.append(huruf.split()[1])
        i=i+1
    Katadasar_Testing_1= '  '.join(Katadasar_Testing)
    f.close()
    print(Katadasar_Testing)
    print(Label_Katadasar_Testing)
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    Katadasar_Testing_1 = stemmer.stem(Katadasar_Testing_1)
    """
    print('List Kata Adalah :')
    print(Katadasar_Testing_1.split())
    print('List Tweet Adalah :')
    print(tweet)
    """
    i=0
    for i in range(len(listTweet_Testing)):
        temp_Testing_listTweet.append(listTweet_Testing[i].split())
        i=i+1
      
    print(len(listTweet_Testing))
    print(temp_Testing_listTweet)
    print(temp_Testing_listTweet[0])

    i=0
    ii=0
    iii=0
    calculate=0
    for i in range(len(Katadasar_Testing_1.split())):
        print(Katadasar_Testing_1.split()[i])
        kalkulasi_Testing.append(Katadasar_Testing_1.split()[i])
        for ii in range(len(temp_Testing_listTweet)):
              print("      "+'  '.join(temp_Testing_listTweet[ii]))
              print(" ----------------------------------")
              for iii in range(len(temp_Testing_listTweet[ii])):
                   #print("      "+temp_Testing_listTweet[ii][iii])
                   if Katadasar_Testing_1.split()[i] in temp_Testing_listTweet[ii][iii]:
                       xx='  '.join(temp_Testing_listTweet[ii][iii])
                       #print( "Ada di "+temp_Testing_listTweet[ii][iii])
                       calculate=calculate+1
                   else:
                       xx='  '.join(temp_Testing_listTweet[ii][iii])
                       #print( "Tidak Ada di "+temp_Testing_listTweet[ii][iii])
                       
                   iii=iii+1
              #print("Jumlah : ")
              #print(calculate)
              kalkulasi_Testing.append(calculate)
              calculate=0
              ii=ii+1
        #print(kalkulasi_Testing)
        Kalkulasi1_Testing.append(kalkulasi_Testing)
        kalkulasi_Testing=[]
        i=i+1
    #print(Kalkulasi1_Testing)
    #print(len(Kalkulasi1_Testing))

    i=0
    ii=0
    count=0
    for i in range(len(Kalkulasi1_Testing)):
        print("---------------")
        print()
        print(Kalkulasi1_Testing[i])
        for ii in range(len(Kalkulasi1_Testing[i])):
            #print(Kalkulasi1_Testing[i][ii])
            if ii>0:
                if str(Kalkulasi1_Testing[i][ii]) in str(0):
                   print(str(Kalkulasi1_Testing[i][ii]) + " Sama "+str(0))
                else:
                   print(str(Kalkulasi1_Testing[i][ii])+ " Tidak Sama "+str(0))
                   count=count+1
            ii=ii+1
            #print(Kalkulasi1_Testing[i][ii])
        #print(count-1)
        
        
        Kalkulasi1_Testing[i].append(count)
        if str(count) in str(0):
            hasilLog= math.log(1,10)
            Kalkulasi1_Testing[i].append(hasilLog)
            #print(len(tweet))
            #print("Nol")
        else:
            Log=len(listTweet_Testing)/(count)
            #print(len(tweet))
            hasilLog= math.log(Log,10)
            Kalkulasi1_Testing[i].append(hasilLog)
            #print("Tidak Nol")
        
        #print("PanjangTweet")
        nilai=len(listTweet_Testing)
        #print(tweet)
        print(Kalkulasi1_Testing[i])
        Kalkulasi2_Testing.append(Kalkulasi1_Testing[i])
        count=0
        i=i+1

    i=0
    ii=0
    
    for i in range(len(Kalkulasi2_Testing)):
        print()
        print("FINAL")
        print(Kalkulasi2_Testing[i])
        for ii in range(len(Kalkulasi2_Testing[i])):
            panj=len(Kalkulasi2_Testing[i])
            if ii>0:
                if ii<len(Kalkulasi2_Testing[i])-2:
                    #print(Kalkulasi2_Testing[i][ii])
                    Kalkulasi2_Testing[i][ii]=Kalkulasi2_Testing[i][ii]*Kalkulasi2_Testing[i][panj-1]
                    
                    #print(Kalkulasi2_Testing[i][ii])
                    #print("//")
            ii=ii+1
        print("Afetr")
        Kalkulasi2_Testing[i].append(Label_Katadasar_Testing[i])
        print(Kalkulasi2_Testing[i])
        temp_Testing.append(Kalkulasi2_Testing[i])
        i=i+1
    print()
    print()
    print(temp_Testing)
    print(listTweet_Testing)
    print(len(listTweet_Testing))

    i=0
    ii=0
    Label_Katadasar_Testing_final=[]
    Label_Katadasar_Testing_final2=[]
    negatif_Testing=[]
    count_positif_Testing_Testing=[]
    count_negatif_Testing_Testing=[]
    hitung1=0
    hitung2=0

    for i in range(len(temp_Testing)):
        #print(temp_Testing[i])
        print("-----")
        Label_Katadasar_Testing_final.append(temp_Testing[i][0])
        Label_Katadasar_Testing_final.append(temp_Testing[i][len(temp_Testing[i])-1])
        print(temp_Testing[i][0])
        print(temp_Testing[i][len(temp_Testing[i])-1])
        for ii in range(len(temp_Testing[i])):
            if ii>0 :
                if ii<len(temp_Testing[i])-3:
                    Label_Katadasar_Testing_final.append(temp_Testing[i][ii])  
                    print(temp_Testing[i][ii])
            ii=ii+1
        Label_Katadasar_Testing_final2.append(Label_Katadasar_Testing_final)
        Label_Katadasar_Testing_final=[]
        """
        if str(temp_Testing[len(temp_Testing)-1]) in "positif_Testing":
            print(str(temp_Testing[ii]) + "positif_Testing")
        else:
            print(str(temp_Testing[len(temp_Testing)-1]) + "negatif_Testing")
        """
        
        i=i+1

    
    print("Final")
    print(Label_Katadasar_Testing_final2)
    panjangResult=len(Label_Katadasar_Testing_final2)-2
    print(len(Label_Katadasar_Testing_final2)-2)
    
    print("List Tweet")
    print(listTweet_Testing)
   
    
    i=0
    ii=0
    iii=0
    xi=0
    xii=0
    count=0
    count2=0
    positif_Testing=[]
    negatif_Testing=[]
    positif_Testing1=[]
    negatif_Testing1=[]
    hasilpembobotan_Testing=[]
    global hasilpembobotan_Testing1_Testing
    hasilpembobotan_Testing1_Testing=[]
    for i in range(len(listTweet_Testing)):
        print("Tweet ke"+'  '.join(str(i)))
        #hasilpembobotan_Testing.append('  '.join(str(i)))
        for ii in range(len(Label_Katadasar_Testing_final2)):
            if Label_Katadasar_Testing_final2[ii][1] in "negatif":
                print(str(Label_Katadasar_Testing_final2[ii][i+2])+" negatif")
                negatif_Testing.append(Label_Katadasar_Testing_final2[ii][i+2])
            if Label_Katadasar_Testing_final2[ii][1] in "positif":
                print(str(Label_Katadasar_Testing_final2[ii][i+2])+" positif")
                positif_Testing.append(Label_Katadasar_Testing_final2[ii][i+2])
            ii=ii+1
        print("-----")
        print(positif_Testing)
        print(sum(positif_Testing))
        hasilpembobotan_Testing.append(sum(positif_Testing))
        positif_Testing1.append(positif_Testing)
        print(negatif_Testing)
        print(sum(negatif_Testing))
        hasilpembobotan_Testing.append(sum(negatif_Testing))
        negatif_Testing1.append(negatif_Testing)
        if sum(positif_Testing) > sum(negatif_Testing):
            hasilpembobotan_Testing.append("positif")
            print("positif")
        else:
            if sum(positif_Testing) < sum(negatif_Testing):
                hasilpembobotan_Testing.append("negatif")
                print("negatif")
            else:
                hasilpembobotan_Testing.append("netral")
                print("netral")
                
        print("-----")
        hasilpembobotan_Testing1_Testing.append(hasilpembobotan_Testing)
        hasilpembobotan_Testing=[]
        positif_Testing=[]
        negatif_Testing=[]
        i=i+1
  
    
    print("Finnaly Testing") 
    print(hasilpembobotan_Testing1_Testing)
    dataTestingKnn=hasilpembobotan_Testing1_Testing
    dataTestingDecition=hasilpembobotan_Testing1_Testing
    dataTestingBayes=hasilpembobotan_Testing1_Testing
    i=0
    for i in range(len(hasilpembobotan_Testing1_Testing)):
        labelingTesting.append(hasilpembobotan_Testing1_Testing[i][2])
        tree21copy.insert('', 'end', text=i+1, values=(hasilpembobotan_Testing1_Testing[i][0],hasilpembobotan_Testing1_Testing[i][1],hasilpembobotan_Testing1_Testing[i][2]))
        i=i+1
    ecludian1()
    
  
def getDataTrainingWordNet(listTweet_Training_Wordnet,listTweet_Testing_Wordnet):
    
    #print("List Tweet Training With WordNet")
    #print(listTweet)
    
    i=0
    for i in range(len(listTweet)):
        fr_kanan_tab_tweet_viewtweet2.insert(END,listTweet[i])
        i=i+1
    """
    print("List Tweet Training With WordNet")
    print(listTweet_Training_Wordnet)
    print("List Tweet Testing With WordNet")
    print(listTweet_Testing_Wordnet)
    """
    RunningDataTrainingWordNet(listTweet_Training_Wordnet,listTweet_Testing_Wordnet)


def RunningDataTrainingWordNet(listTweet_Training_Wordnet,listTweet_Testing_Wordnet):
    print("Tahap 3. ----------------------------------------------------------------")
    global dataTrainingKnnWordnet
    global dataTrainingDecitionWordnet
    global dataTrainingBayesWordnet
    global final
    global labelingTesting1
    global tweet_2
    global treeview
    global treeview2
    labelingTesting1=[]
    kalkulat=[]
    kalkulat_1=[]
    kalkulat_2=[]
    treeview =tree1
    katadasar=[]
    tweet_2=[]
    label=[]
    i=0
    f = open(os.path.basename(str('label2.csv')), "r")
    for huruf in f:
        katadasar.append(huruf.split()[0])
        label.append(huruf.split()[1])
        i=i+1
    katadasar_1= '  '.join(katadasar)
    f.close()
    print("Kata dasar")
    print(katadasar_1)
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    katadasar_1 = stemmer.stem(katadasar_1)
    
    i=0
    for i in range(len(listTweet_Training_Wordnet)):
        tweet_2.append(listTweet_Training_Wordnet[i].split())
        i=i+1
    print(len(listTweet_Training_Wordnet))
    print(tweet_2)
    print(tweet_2[0])
    i=0
    ii=0
    iii=0
    calculate=0
    for i in range(len(katadasar_1.split())):
        print(katadasar_1.split()[i])
        kalkulat.append(katadasar_1.split()[i])
        for ii in range(len(tweet_2)):
              print("      "+'  '.join(tweet_2[ii]))
              print(" ----------------------------------")
              for iii in range(len(tweet_2[ii])):
                   print("      "+tweet_2[ii][iii])
                   if katadasar_1.split()[i] in tweet_2[ii][iii]:
                       xx='  '.join(tweet_2[ii][iii])
                       print( "Ada di "+tweet_2[ii][iii])
                       calculate=calculate+1
                   else:
                       xx='  '.join(tweet_2[ii][iii])
                       print( "Tidak Ada di "+tweet_2[ii][iii])
                       
                   iii=iii+1
              print("Jumlah : ")
              print(calculate)
              kalkulat.append(calculate)
              calculate=0
              ii=ii+1
        print(kalkulat)
        kalkulat_1.append(kalkulat)
        kalkulat=[]
        i=i+1
    
    print(kalkulat_1)
    i=0
    no=0
    for i in range(len(kalkulat_1)):  
        print(kalkulat_1[i][0])
        ii=1
        for ii in range(len(kalkulat_1[i])):
            print(kalkulat_1[i][ii])
            tree3.insert('', 'end', text=no+1, values=(kalkulat_1[i][0],kalkulat_1[i][ii],ii))
            ii=ii+1
            no=no+1
        ii=i+1
    
    i=0
    ii=0
    count=0
    for i in range(len(kalkulat_1)):
        print("---------------")
        print()
        print(kalkulat_1[i])
        for ii in range(len(kalkulat_1[i])):
            #print(kalkulat_1[i][ii])
            if ii>0:
                if str(kalkulat_1[i][ii]) in str(0):
                   print(str(kalkulat_1[i][ii]) + " Sama "+str(0))
                else:
                   print(str(kalkulat_1[i][ii])+ " Tidak Sama "+str(0))
                   count=count+1
            ii=ii+1
            #print(kalkulat_1[i][ii])
        #print(count-1)
        
        
        kalkulat_1[i].append(count)
        if str(count) in str(0):
            hasilLog= math.log(1,10)
            kalkulat_1[i].append(hasilLog)
            #print(len(listTweet))
            #print("Nol")
        else:
            Log=len(listTweet)/(count)
            #print(len(listTweet))
            hasilLog= math.log(Log,10)
            kalkulat_1[i].append(hasilLog)
            #print("Tidak Nol")
        
        #print("PanjangTweet")
        nilai=len(listTweet_Training_Wordnet)
        #print(listTweet)
        print(kalkulat_1[i])
        kalkulat_2.append(kalkulat_1[i])
        count=0
        i=i+1
        i=0
    
    i=0
    no=0
    
    for i in range(len(kalkulat_2)):
       
        tree3copy.insert('', 'end', text=i+1, values=(kalkulat_2[i][0],kalkulat_2[i][len(kalkulat_2[i])-2],kalkulat_2[i][len(kalkulat_2[i])-1]))
   
        i=i+1
    print("TF-IDF Berakhir")
    print(kalkulat_2)
    ii=0
    new_kakulat=[]
    new_kakulat_1=[]
    temp=[]
    for i in range(len(kalkulat_2)):
        print()
        print("FINAL")
        print(kalkulat_2[i])
        for ii in range(len(kalkulat_2[i])):
            panj=len(kalkulat_2[i])
            if ii>0:
                if ii<len(kalkulat_2[i])-2:
                    #print(kalkulat_2[i][ii])
                    kalkulat_2[i][ii]=kalkulat_2[i][ii]*kalkulat_2[i][panj-1]
                    
                    #print(kalkulat_2[i][ii])
                    #print("//")
            ii=ii+1
        print("Afetr")
        kalkulat_2[i].append(label[i])
        print(kalkulat_2[i])
        temp.append(kalkulat_2[i])
        i=i+1
    print()
    print()
    print(temp)
    print(listTweet_Training_Wordnet)
    print(len(listTweet_Training_Wordnet))
    print("-------------------------------------")
    i=0
    for i in range(len(temp)):
        print(temp[i])
        print(temp[i][0])
        ii=0
        for ii in range(len(temp[i])-4):
            ii=ii+1
            print(temp[i][ii])
            tree4.insert('', 'end', text=ii, values=(temp[i][ii],temp[i][0],temp[i][len(temp[i])-1]))
    
        i=i+1

    
    i=0
    ii=0
    label_final=[] 
    label_final2=[]
    negatif=[]
    count_positif=[]
    count_negatif=[]
    hitung1=0
    hitung2=0

    for i in range(len(temp)):
        #print(temp[i])
        print("-----")
        label_final.append(temp[i][0])
        label_final.append(temp[i][len(temp[i])-1])
        print(temp[i][0])
        print(temp[i][len(temp[i])-1])
        for ii in range(len(temp[i])):
            if ii>0 :
                if ii<len(temp[i])-3:
                    label_final.append(temp[i][ii])  
                    print(temp[i][ii])
            ii=ii+1
        label_final2.append(label_final)
        label_final=[]
     
        i=i+1

    
    print("Final")
    print(label_final2)
    panjangResult=len(label_final2)-2
    print(len(label_final2)-2)
    
    print("List listTweet")
    print(listTweet_Training_Wordnet)
    
    
    i=0
    ii=0
    iii=0
    xi=0
    xii=0
    count=0
    count2=0
    positif=[]
    negatif=[]
    positif1=[]
    negatif1=[]
    hasilpembobotan=[]
    hasilpembobotan1=[]
    for i in range(len(listTweet_Training_Wordnet)):
        print("Tweet ke"+'  '.join(str(i)))
        #hasilpembobotan.append('  '.join(str(i)))
        for ii in range(len(label_final2)):
            if label_final2[ii][1] in "negatif":
                print(str(label_final2[ii][i+2])+" negatif")
                negatif.append(label_final2[ii][i+2])
            if label_final2[ii][1] in "positif":
                print(str(label_final2[ii][i+2])+" positif")
                positif.append(label_final2[ii][i+2])
            ii=ii+1
        print("-----")
        print(positif)
        print(sum(positif))
        hasilpembobotan.append(sum(positif))
        positif1.append(positif)
        print(negatif)
        print(sum(negatif))
        hasilpembobotan.append(sum(negatif))
        negatif1.append(negatif)
        if sum(positif) > sum(negatif):
            hasilpembobotan.append("positif")
            print("positif")
        else:
            if sum(positif) < sum(negatif):
                hasilpembobotan.append("negatif")
                print("negatif")
            else:
                hasilpembobotan.append("netral")
                print("netral")
                
        print("-----")
        hasilpembobotan1.append(hasilpembobotan)
        hasilpembobotan=[]
        positif=[]
        negatif=[]
        i=i+1
  
    
    print("Finnalyyy") 
    print(hasilpembobotan1)
    dataTrainingKnnWordnet=hasilpembobotan1
    dataTrainingDecitionWordnet=hasilpembobotan1
    dataTrainingBayesWordnet=hasilpembobotan1
    i=0
    for i in range(len(hasilpembobotan1)):
        print(hasilpembobotan1[i])
        labelingTesting1.append(hasilpembobotan1[i][2])
        tree4copy.insert('', 'end', text=i+1, values=(hasilpembobotan1[i][0],hasilpembobotan1[i][1],hasilpembobotan1[i][2]))    
        i=i+1
    getDataTestingWordNet(listTweet_Training_Wordnet,listTweet_Testing_Wordnet)

    
    
def getDataTraining(listTweet_Training,listTweet_Testing):
    print("Tahap 2. ----------------------------------------------------------------")
    print(listTweet_Training)
    RunningDataTraining(listTweet_Training,listTweet_Testing)
    
   



def RunningDataTraining(listTweet_Training,listTweet_Testing):
    print("Tahap 3. ----------------------------------------------------------------")
    global dataTrainingKnn
    global dataTrainingDecition
    global dataTrainingBayes
    global dataTrainingDecition
    global dataTrainingBayes
    global tweet_2
    global treeview
    global treeview2
    kalkulat=[]
    kalkulat_1=[]
    kalkulat_2=[]
    treeview =tree1
    katadasar=[]
    tweet_2=[]
    label=[]
    i=0
    f = open(os.path.basename(str('label.csv')), "r")
    for huruf in f:
        katadasar.append(huruf.split()[0])
        label.append(huruf.split()[1])
        i=i+1
    katadasar_1= '  '.join(katadasar)
    f.close()
    print("Kata dasar")
    print(katadasar_1)
    print("Label")
    print(label)

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    katadasar_1 = stemmer.stem(katadasar_1)
    
    i=0
    for i in range(len(listTweet_Training)):
        tweet_2.append(listTweet_Training[i].split())
        i=i+1
    print(len(listTweet_Training))
    print(tweet_2)
    print(tweet_2[0])
    i=0
    ii=0
    iii=0
    calculate=0
    for i in range(len(katadasar_1.split())):
        print(katadasar_1.split()[i])
        kalkulat.append(katadasar_1.split()[i])
        for ii in range(len(tweet_2)):
              print("      "+'  '.join(tweet_2[ii]))
              print(" ----------------------------------")
              for iii in range(len(tweet_2[ii])):
                   print("      "+tweet_2[ii][iii])
                   if katadasar_1.split()[i] in tweet_2[ii][iii]:
                       xx='  '.join(tweet_2[ii][iii])
                       print( "Ada di "+tweet_2[ii][iii])
                       calculate=calculate+1
                   else:
                       xx='  '.join(tweet_2[ii][iii])
                       print( "Tidak Ada di "+tweet_2[ii][iii])
                       
                   iii=iii+1
              print("Jumlah : ")
              print(calculate)
              kalkulat.append(calculate)
              calculate=0
              ii=ii+1
        print(kalkulat)
        kalkulat_1.append(kalkulat)
        kalkulat=[]
        i=i+1
    
    print(kalkulat_1)
    i=0
    no=0
    for i in range(len(kalkulat_1)):  
        print(kalkulat_1[i][0])
        ii=1
        for ii in range(len(kalkulat_1[i])):
            print(kalkulat_1[i][ii])
            tree1.insert('', 'end', text=no+1, values=(kalkulat_1[i][0],kalkulat_1[i][ii],ii))
            ii=ii+1
            no=no+1
        ii=i+1
    #print(len(kalkulat_1))
    
    
    i=0
    ii=0
    count=0
    for i in range(len(kalkulat_1)):
        print("---------------")
        print()
        print(kalkulat_1[i])
        for ii in range(len(kalkulat_1[i])):
            #print(kalkulat_1[i][ii])
            if ii>0:
                if str(kalkulat_1[i][ii]) in str(0):
                   print(str(kalkulat_1[i][ii]) + " Sama "+str(0))
                else:
                   print(str(kalkulat_1[i][ii])+ " Tidak Sama "+str(0))
                   count=count+1
            ii=ii+1
            #print(kalkulat_1[i][ii])
        #print(count-1)
        
        
        kalkulat_1[i].append(count)
        if str(count) in str(0):
            hasilLog= math.log(1,10)
            kalkulat_1[i].append(hasilLog)
            #print(len(listTweet))
            #print("Nol")
        else:
            Log=len(listTweet)/(count)
            #print(len(listTweet))
            hasilLog= math.log(Log,10)
            kalkulat_1[i].append(hasilLog)
            #print("Tidak Nol")
        
        #print("PanjangTweet")
        nilai=len(listTweet)
        #print(listTweet)
        print(kalkulat_1[i])
        kalkulat_2.append(kalkulat_1[i])
        count=0
        i=i+1

        
        i=0
    
    i=0
    no=0
    
    
    for i in range(len(kalkulat_2)):
        tree1copy.insert('', 'end', text=i+1, values=(kalkulat_2[i][0],kalkulat_2[i][len(kalkulat_2[i])-2],kalkulat_2[i][len(kalkulat_2[i])-1])) 
        i=i+1
    
    
    print("TF-IDF Berakhir")
    print(kalkulat_2)
    
    ii=0
    new_kakulat=[]
    new_kakulat_1=[]
    temp=[]
    for i in range(len(kalkulat_2)):
        print()
        print("FINAL")
        print(kalkulat_2[i])
        for ii in range(len(kalkulat_2[i])):
            panj=len(kalkulat_2[i])
            if ii>0:
                if ii<len(kalkulat_2[i])-2:
                    #print(kalkulat_2[i][ii])
                    kalkulat_2[i][ii]=kalkulat_2[i][ii]*kalkulat_2[i][panj-1]
                    
                    #print(kalkulat_2[i][ii])
                    #print("//")
            ii=ii+1
        print("Afetr")
        kalkulat_2[i].append(label[i])
        print(kalkulat_2[i])
        temp.append(kalkulat_2[i])
        i=i+1
    print()
    print()
    print("ini temp")
    print(temp)
    
    print(listTweet_Training)
    print(len(listTweet_Training))
    print("-------------------FELT---------------")
    i=0
    for i in range(len(temp)):
        print(temp[i])
        #print(temp[i][0])
        ii=0
        print(i)
        for ii in range(len(temp[i])-4):
            ii=ii+1
            print(ii)
            print(temp[i][ii])
            print(temp[i][0])
            print(temp[i][len(temp[i])-1])
            tree2.insert('', 'end', text=ii, values=(temp[i][ii],temp[i][0],temp[i][len(temp[i])-1]))
        i=i+1
    
    i=0
    ii=0
    label_final=[] 
    label_final2=[]
    negatif=[]
    count_positif=[]
    count_negatif=[]
    hitung1=0
    hitung2=0

    for i in range(len(temp)):
        #print(temp[i])
        print("-----")
        label_final.append(temp[i][0])
        label_final.append(temp[i][len(temp[i])-1])
        print(temp[i][0])
        print(temp[i][len(temp[i])-1])
        for ii in range(len(temp[i])):
            if ii>0 :
                if ii<len(temp[i])-3:
                    label_final.append(temp[i][ii])  
                    print(temp[i][ii])
            ii=ii+1
        label_final2.append(label_final)
        label_final=[]
     
        i=i+1

    
    print("Final")
    print(label_final2)
    panjangResult=len(label_final2)-2
    print(len(label_final2)-2)
    
    print("List listTweet")
    print(listTweet_Training)
   
    i=0
    ii=0
    iii=0
    xi=0
    xii=0
    count=0
    count2=0
    positif=[]
    negatif=[]
    positif1=[]
    negatif1=[]
    hasilpembobotan=[]
    hasilpembobotan1=[]
    for i in range(len(listTweet_Training)):
        print("Tweet ke"+'  '.join(str(i)))
        #hasilpembobotan.append('  '.join(str(i)))
        for ii in range(len(label_final2)):
            if label_final2[ii][1] in "negatif":
                print(str(label_final2[ii][i+2])+" negatif")
                negatif.append(label_final2[ii][i+2])
            if label_final2[ii][1] in "positif":
                print(str(label_final2[ii][i+2])+" positif")
                positif.append(label_final2[ii][i+2])
            ii=ii+1
        print("-----")
        print(positif)
        print(sum(positif))
        hasilpembobotan.append(sum(positif))
        positif1.append(positif)
        print(negatif)
        print(sum(negatif))
        hasilpembobotan.append(sum(negatif))
        negatif1.append(negatif)
        if sum(positif) > sum(negatif):
            hasilpembobotan.append("positif")
            print("positif")
        else:
            if sum(positif) < sum(negatif):
                hasilpembobotan.append("negatif")
                print("negatif")
            else:
                hasilpembobotan.append("netral")
                print("netral")
                
        print("-----")
        hasilpembobotan1.append(hasilpembobotan)
        hasilpembobotan=[]
        positif=[]
        negatif=[]
        i=i+1
  
    
    print("Finnalyyy") 
    print()
  
    i=0
    dataTrainingKnn=hasilpembobotan1
    dataTrainingDecition=hasilpembobotan1
    dataTrainingBayes=hasilpembobotan1
    for i in range(len(hasilpembobotan1)):
        print(hasilpembobotan1[i])
        tree2copy.insert('', 'end', text=i+1, values=(hasilpembobotan1[i][0],hasilpembobotan1[i][1],hasilpembobotan1[i][2]))    
        i=i+1

    getDataTesting(listTweet_Training,listTweet_Testing)
    

def decition():
    print("ini decition 1")
    global finaldecition1
    finaldecition1=[]
    t1=[]
    t2=[]
    XD=[]
    X2D=[]
    YD=[]
    
    panjangtrainingdecition=len(dataTrainingDecition)
    i=0
    for i in range(panjangtrainingdecition):
        XD.append(dataTrainingDecition[i][0])
        XD.append(dataTrainingDecition[i][1])
        X2D.append(XD)
        XD=[]
        YD.append(dataTrainingDecition[i][2])
        i=i+1
    x=X2D
    y=YD
    print(x)
    print(y)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x, y)

    print(dataTestingDecition)
    print("--------------------------------")
    panjangtestingdecition=len(dataTestingDecition)
    i=0
    for i in range(panjangtestingdecition):
        t1.append(dataTestingDecition[i][0])
        t1.append(dataTestingDecition[i][1])
        if clf.predict([[dataTestingDecition[i][0], dataTestingDecition[i][1]]]):
            #print(clf.predict([[dataTestingDecition[i][0], dataTestingDecition[i][1]]]))
            
            if clf.predict([[dataTestingDecition[i][0], dataTestingDecition[i][1]]])=="positif":
                print("positif")
                temppp1=clf.predict([[dataTestingDecition[i][0], dataTestingDecition[i][1]]])
                finaldecition1.append("positif")
            if clf.predict([[dataTestingDecition[i][0], dataTestingDecition[i][1]]])=="negatif":
                print("negatif")
                finaldecition1.append("negatif")
            else:
                print("netral")
                finaldecition1.append("netral")
            
        else:
            #print(clf.predict([[dataTestingDecition[i][0], dataTestingDecition[i][1]]]))
            
            if clf.predict([[dataTestingDecition[i][0], dataTestingDecition[i][1]]])=="positif":
                print("positif")
                temppp1=clf.predict([[dataTestingDecition[i][0], dataTestingDecition[i][1]]])
                finaldecition1.append("positif")
            if clf.predict([[dataTestingDecition[i][0], dataTestingDecition[i][1]]])=="negatif":
                print("negatif")
                finaldecition1.append("negatif")
            else:
                print("netral")
                finaldecition1.append("netral")
            
        t2.append(t1)
        t1=[]
        i=i+1
    print(finaldecition1)

def decitionWordnet():
    global finaldecition1Wordnet
    finaldecition1Wordnet=[]
    t1=[]
    t2=[]
    XD=[]
    X2D=[]
    YD=[]
    
    panjangtrainingdecition=len(dataTrainingDecitionWordnet)
    i=0
    for i in range(panjangtrainingdecition):
        XD.append(dataTrainingDecitionWordnet[i][0])
        XD.append(dataTrainingDecitionWordnet[i][1])
        X2D.append(XD)
        XD=[]
        YD.append(dataTrainingDecitionWordnet[i][2])
        i=i+1
    x=X2D
    y=YD
    print(x)
    print(y)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x, y)

    print(dataTestingDecitionWordnet)
    print("--------------------------------")
    panjangtestingdecition=len(dataTestingDecitionWordnet)
    i=0
    for i in range(panjangtestingdecition):
        t1.append(dataTestingDecitionWordnet[i][0])
        t1.append(dataTestingDecitionWordnet[i][1])
        if clf.predict([[dataTestingDecitionWordnet[i][0], dataTestingDecitionWordnet[i][1]]]):
            
            if clf.predict([[dataTestingDecition[i][0], dataTestingDecition[i][1]]])=="positif":
                print("positif")
                finaldecition1Wordnet.append("positif")
            if clf.predict([[dataTestingDecition[i][0], dataTestingDecition[i][1]]])=="negatif":
                print("negatif")
                finaldecition1Wordnet.append("negatif")
            else:
                print("netral")
                finaldecition1Wordnet.append("netral")
        else:
            #print(clf.predict([[dataTestingDecition[i][0], dataTestingDecition[i][1]]]))
            temppp2=clf.predict([[dataTestingDecitionWordnet[i][0], dataTestingDecitionWordnet[i][1]]])
            finaldecition1Wordnet.append(str(temppp2))
        t2.append(t1)
        t1=[]
        i=i+1
    print(finaldecition1Wordnet)



def bayesWordnet():
    #print(dataTrainingBayes)
    global finalbayes1Wordnet
    finalbayes1Wordnet=[]
    t1=[]
    t2=[]
    XD=[]
    X2D=[]
    YD=[]
    panjangtrainingbayes=len(dataTrainingBayesWordnet)
    i=0
    for i in range(panjangtrainingbayes):
        XD.append(dataTrainingBayesWordnet[i][0])
        XD.append(dataTrainingBayesWordnet[i][1])
        X2D.append(XD)
        XD=[]
        YD.append(dataTrainingBayesWordnet[i][2])
        i=i+1
    x=X2D
    y=YD
    print(x)
    print(y) 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)
    modelnb = GaussianNB()# Memasukkan data training pada fungsi klasifikasi naive bayes
    nbtrain = modelnb.fit(x, y)
    
    print(dataTestingBayesWordnet)
    i=0
    dataTestingbayes2=[]
    dataTestingbayes22=[]
    for i in range(len(dataTestingBayesWordnet)):
        dataTestingbayes2.append(dataTestingBayesWordnet[i][0])
        dataTestingbayes2.append(dataTestingBayesWordnet[i][1])
        print(dataTestingbayes2)
        dataTestingbayes22.append(dataTestingbayes2)
        dataTestingbayes2=[]
        i=i+1
    
   
    #x_test = [[82,87],[90,86]]

    y_pred = nbtrain.predict(dataTestingbayes22)
    print(y_pred)
    finalbayes1Wordnet= y_pred




def bayes():
    #print(dataTrainingBayes)
    global finalbayes1
    finalbayes1=[]
    t1=[]
    t2=[]
    XD=[]
    X2D=[]
    YD=[]
    panjangtrainingbayes=len(dataTrainingBayes)
    i=0
    for i in range(panjangtrainingbayes):
        XD.append(dataTrainingBayes[i][0])
        XD.append(dataTrainingBayes[i][1])
        X2D.append(XD)
        XD=[]
        YD.append(dataTrainingBayes[i][2])
        i=i+1
    x=X2D
    y=YD
    print(x)
    print(y) 
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)
    modelnb = GaussianNB()# Memasukkan data training pada fungsi klasifikasi naive bayes
    nbtrain = modelnb.fit(x, y)
    
    print(dataTestingBayes)
    i=0
    dataTestingbayes2=[]
    dataTestingbayes22=[]
    for i in range(len(dataTestingBayes)):
        dataTestingbayes2.append(dataTestingBayes[i][0])
        dataTestingbayes2.append(dataTestingBayes[i][1])
        print(dataTestingbayes2)
        dataTestingbayes22.append(dataTestingbayes2)
        dataTestingbayes2=[]
        i=i+1
    
   
    #x_test = [[82,87],[90,86]]

    y_pred = nbtrain.predict(dataTestingbayes22)
    print(y_pred)
    finalbayes1= y_pred



def chart():
    labels = ['K-NN', 'Naive Bayes', 'Decition Tree']
    men_means = [akurasi1, akurasi2, akurasi3]
    women_means = [akurasi4, akurasi5, akurasi6]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    rects1 = ax.bar(x - width/2, men_means, width, label='Without Wordnet & Spelling Checker')
    rects2 = ax.bar(x + width/2, women_means, width, label='With Wordnet & Spelling Checker')
    autolabel(rects1)
    autolabel(rects2)
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores in %')
    ax.set_title('Accuracy of Classifier Methods')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

     
def opendir(corpus):
    # ---------------------------------- Tahap 1 ------------------------------------------------."
    global hasil1
    global hasil2
    global hasil3
    global hasil4
    global hasil5
    global hasil6
    global tweet_1
    global listTweet
    global listTweet_Training
    global listTweet_Testing
    global listTweet_Training_Wordnet
    global listTweet_Testing_Wordnet
    global path_Label
    global teg
    global value;
    global path
    listTweet_Training_Wordnet=[]
    listTweet_Testing_Wordnet=[]
    teg=[' ']
    path_Label="label.csv"
    path_Label1="slang.csv"
    path_Label2="label2.csv"
    fr_kanan_tab_tweet_viewtweet1.delete('1.0',END)
    in_path = askopenfilename()
    path=in_path;
    value=path
    Import_Corpus_TextCorpus.insert(END, in_path)
    jumlahcorpus=listboxDatacorpus.size()
    jumlahcorpus=jumlahcorpus+1
    Import_Corpus_TextCorpus.insert(END, ' ')
    #menmabil label kata positif dannegatif dan memasukan ke list box
    f1 = open(os.path.basename(str(path_Label)), "r")
    for huruf1 in f1:
          listboxDatacorpus.insert(END,huruf1)    
    f1.close()
    
    f1 = open(os.path.basename(str(path_Label1)), "r")
    for huruf1 in f1:
          listboxDatacorpus1.insert(END,huruf1)    
    f1.close()
    

    f1 = open(os.path.basename(str(path_Label2)), "r")
    for huruf1 in f1:
          listboxDatacorpus2.insert(END,huruf1)    
    f1.close()


    listTweet=[]
    listTweet_Training=[]
    listTweet_Testing=[]
    tweet_1=[]
    f_1 = open(os.path.basename(str(path)), "r")
    for list_1 in f_1:
        tweet_1.append(list_1)
    f_1.close()
    print("Print List Original")
    print(tweet_1)
    panjang_testing=len(tweet_1)
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    #Bagian WithOut WordNet & Spelling Checker
    i=0
    #Bagian 1. Lower, Stopword Removal, stemming
    for i in range(panjang_testing):
        testingT=tweet_1[i].lower()
        testingT=stopword.remove(testingT)
        testingT = stemmer.stem(testingT)
        fr_kanan_tab_tweet_viewtweet1.insert(END,testingT)
        listTweet.append(testingT)
        i=i+1
    print("List Tweet Training WithOut WordNet")
   
    
    i=0
    for i in range(len(listTweet)):
        if i<(len(listTweet)-600):
            listTweet_Training.append(listTweet[i])
        else:
            listTweet_Testing.append(listTweet[i])
            
        i=i+1
    
    getDataTraining(listTweet_Training,listTweet_Testing)
    
    bayes()
    decition()
    
    hasil1=labelingTesting
    hasil2=finalecludian1
    hasil3=finaldecition1
    hasil4=finalbayes1
    

 
    
    #Bagian With WordNet & Spelling Checker
    f_2 = open(os.path.basename(str('slang.csv')), "r")
    #Bagian 1. Spelling Checker Prosess
    for list_2 in f_2:
        i=0
        for i in range(len(listTweet)):
            res = [sub.replace(str(list_2.split()[0]), str(list_2.split()[1])) for sub in listTweet[i].split()]
            #print(str(res))
            s1 = ' '
            #print('s1.join(s2):', s1.join(res))
            listTweet[i]=s1.join(res)
            i=i+1
    
    f_2.close()
    i=0
    for i in range(len(listTweet)):
        if i<(len(listTweet)-600):
            listTweet_Training_Wordnet.append(listTweet[i])
        else:
            listTweet_Testing_Wordnet.append(listTweet[i])
            
        i=i+1
    
    getDataTrainingWordNet(listTweet_Training_Wordnet,listTweet_Testing_Wordnet)
    
    bayesWordnet()
    decitionWordnet()
    print("----------DATA TANPA WORDNET & SPELLING CHECKER---------------")
    print("TF-IDF")
    hasil1=labelingTesting
    print(labelingTesting)
    print("GENERAL K-NN")
    hasil2=finalecludian1
    print(finalecludian1)
    print("DECITION")
    hasil3=finaldecition1
    print(finaldecition1)
    print("BAYES")
    hasil4=finalbayes1
    print(finalbayes1)


    print("----------DATA DENGAN WORDNET & SPELLING CHECKER---------------")
    print("TF-IDF")
    hasil5=labelingTesting_Wordnert
    print(hasil5)
    print("K-NN WITH WORDNET")
    hasil6=finalecludianWordnet
    print(hasil6)
    print("DECITION WITH WORDNET")
    hasil7=finaldecition1Wordnet
    print(hasil7)
    print("BAYES WITH WORDNET")
    hasil8=finalbayes1Wordnet
    print(hasil8)
   
   
    i=0
    for i in range(len(hasil5)):
        tree5.insert('', 'end', text=i+1, values=(hasil1[i],hasil5[i],hasil2[i],hasil3[i],hasil4[i],hasil6[i],hasil7[i],hasil8[i]))
        i=i+1
    global result1
    global result2
    global result3
    global result4
    global result5
    global result6
    result1=[]
    result2=[]
    result3=[]
    result4=[]
    result5=[]
    result6=[]
    print("------------------KNN---------------")
    TP1=0
    TN1=0
    FP1=0
    FN1=0
    i=0
    print(hasil5)
    print(hasil2)
    for i in range(len(hasil5)):
       
        if hasil5[i] =="positif" and hasil5[i]==hasil2[i]:
            
            TP1=TP1+1
           
        if hasil5[i] =="negatif" and hasil5[i]==hasil2[i]:
         
            TN1=TN1+1
           
        if hasil5[i] =="netral" and hasil5[i]==hasil2[i]:
            
            TP1=TP1+1
            
        if hasil5[i] =="negatif" and hasil2[i]=="netral":
            
            FP1=FP1+1
            
        if hasil5[i] =="positif" and hasil2[i]=="netral":
            
            FN1=FN1+1
           
        if hasil5[i] =="netral" and hasil2[i]=="positif":
           
            FN1=FN1+1
          
        if hasil5[i] =="netral" and hasil2[i]=="negatif":
          
            FN1=FN1+1
          
        i=i+1
    
    print("Hasil TP")
    print(TP1)
    print("Hasil TN")
    print(TN1)
    print("Hasil FP")
    print(FP1)
    print("Hasil FN")
    print(FN1)
    print("Precision :")
    if (TP1+FP1) ==0:
        precision1=0
    else:  
        precision1=TP1/(TP1+FP1)
    print("Recall :")
    if TP1+FN1==0:
        recall1=0
    else:
        recall1=TP1/(TP1+FN1)
    print("Accuracy :")
    if (TP1+FN1+TN1+FP1)==0:
        accuracy1=0
    else:
        accuracy1=(TP1+TN1)/(TP1+FN1+TN1+FP1)
    print("F1 Score :")
    if (recall1+precision1)==0:
        f1score1=0
    else:
        f1score1=(2 * recall1*precision1)/(recall1+precision1)
    result1.append("K-NN")
    result1.append(TP1)
    result1.append(TN1)
    result1.append(FP1)
    result1.append(FN1)
    result1.append(precision1)
    result1.append(recall1)
    result1.append(accuracy1*100)
    result1.append(f1score1*100)

    print(result1)
    
    print("------------------Naive Bayes---------------")
    TP2=0
    TN2=0
    FP2=0
    FN2=0
    i=0
    print(hasil5)
    print(hasil3)
    for i in range(len(hasil5)):
        if hasil5[i] =="positif" and hasil5[i]==hasil2[i]:
            TP2=TP2+1
        if hasil5[i] =="negatif" and hasil5[i]==hasil2[i]:
            TN2=TN2+1
        if hasil5[i] =="netral" and hasil5[i]==hasil2[i]:
            TP2=TP2+1
        if hasil5[i] =="negatif" and hasil2[i]=="netral":
            FP2=FP2+1
        if hasil5[i] =="positif" and hasil2[i]=="netral":
            FN2=FN2+1
        if hasil5[i] =="netral" and hasil2[i]=="positif":
            FN2=FN2+1
        if hasil5[i] =="netral" and hasil2[i]=="negatif":
            FN2=FN2+1
        i=0
    
    print("Hasil TP")
    print(TP2)
    print("Hasil TN")
    print(TN2)
    print("Hasil FP")
    print(FP2)
    print("Hasil FN")
    print(FN2)
    print("Precision :")
    if (TP2+FP2)==0:
        precision2=0
        print("0")
    else:
        precision2=TP2/(TP2+FP2)
        print(TP2/(TP2+FP2))
    print("Recall :")
    if (TP2+FN2)==0:
        recall2=0
        print("0")
    else:
        recall2=TP2/(TP2+FN2)
        print(TP2/(TP2+FN2))
    print("Accuracy :")
    if (TP2+FN2+TN2+FP2)==0:
        accuracy2=0
        print(accuracy2)
    else:
        accuracy2=(TP2+TN2)/(TP2+FN2+TN2+FP2)
        print(accuracy2)
    print("F1 Score :")
    if (recall2+precision2)==0:
        f1score2=0
        print(f1score2)
    else:
        f1score2=(2 * recall2*precision2)/(recall2+precision2)
        print(f1score2)
    result2.append("Naive Bayes")
    result2.append(TP2)
    result2.append(TN2)
    result2.append(FP2)
    result2.append(FN2)
    result2.append(precision2)
    result2.append(recall2)
    result2.append(accuracy2*100)
    result2.append(f1score2*100)
    print(result2)
    

    print("------------------Decition 3---------------")
    TP3=0
    TN3=0
    FP3=0
    FN3=0
    i=0
    
    print(hasil5)
    print(hasil4)
    for i in range(len(hasil5)):
        if hasil5[i] =="positif" and hasil5[i]==hasil3[i]:
            TP3=TP3+1
        if hasil5[i] =="negatif" and hasil5[i]==hasil3[i]:
            TN3=TN3+1
        if hasil5[i] =="netral" and hasil5[i]==hasil3[i]:
            TP3=TP3+1
        if hasil5[i] =="negatif" and hasil3[i]=="netral":
            FP3=FP3+1
        if hasil5[i] =="positif" and hasil3[i]=="netral":
            FN3=FN3+1
        if hasil5[i] =="netral" and hasil3[i]=="positif":
            FN3=FN3+1
        if hasil5[i] =="netral" and hasil3[i]=="negatif":
            FN3=FN3+1
        i=0
    
    print("Hasil TP")
    print(TP3)
    print("Hasil TN")
    print(TN3)
    print("Hasil FP")
    print(FP3)
    print("Hasil FN")
    print(FN3)
    print("Precision :")
    if (TP3+FP3)==0:
        precision3=0
        print("0")
    else:
        precision3=TP3/(TP3+FP3)
        print(TP3/(TP3+FP3))
    print("Recall :")
    if (TP3+FN3)==0:
        recall3=0
        print("0")
    else:
        recall3=TP3/(TP3+FN3)
        print(TP3/(TP3+FN3))
    print("Accuracy :")
    if (TP3+FN3+TN3+FP3)==0:
        accuracy3=0
        print("0")
    else:
        accuracy3=(TP3+TN3)/(TP3+FN3+TN3+FP3)
        print(print(accuracy3))
    print("F1 Score :")        
    if (recall3+precision3)==0:
        f1score3=0
        print(f1score3)
    else:
        f1score3=(2 * recall3*precision3)/(recall3+precision3)
        print(f1score3)
    result3.append("Decition Tree")
    result3.append(TP3)
    result3.append(TN3)
    result3.append(FP3)
    result3.append(FN3)
    result3.append(precision3)
    result3.append(recall3)
    result3.append(accuracy3*100)
    result3.append(f1score3*100)
    print(result3)



    print("------------------K-NN Wordnet---------------")
    TP4=0
    TN4=0
    FP4=0
    FN4=0
    i=0
    print(hasil5)
    print(hasil6)
    for i in range(len(hasil5)):
        if hasil5[i] =="positif" and hasil5[i]==hasil6[i]:
            TP4=TP4+1
        if hasil5[i] =="negatif" and hasil5[i]==hasil6[i]:
            TN4=TN4+1
            
        if hasil5[i] =="netral" and hasil5[i]==hasil6[i]:
           
            TP4=TP4+1
            
        if hasil5[i] =="negatif" and hasil6[i]=="netral":
            
            FP4=FP4+1
            
        if hasil5[i] =="positif" and hasil6[i]=="netral":
            
            FN4=FN4+1
            
        if hasil5[i] =="netral" and hasil6[i]=="positif":
           
            FN4=FN4+1
            
        if hasil5[i] =="netral" and hasil6[i]=="negatif":
          
            FN4=FN4+1
           
        i=i+1
    
    print("Hasil TP")
    print(TP4)
    print("Hasil TN")
    print(TN4)
    print("Hasil FP")
    print(FP4)
    print("Hasil FN")
    print(FN4)
    print("Precision :")
    if (TP4+FP4)==0:
        precision4=0
        print("0")
    else:
        precision4=TP4/(TP4+FP4)
        print(TP4/(TP4+FP4))
    print("Recall :")
    if (TP4+FN4)==0:
        recall4=0
        print("0")
    else:
        recall4=TP4/(TP4+FN4)
        print(TP4/(TP4+FN4))
    print("Accuracy :")
    if (TP4+FN4+TN4+FP4)==0:
        accuracy4=0
        print("0")
    else:
        accuracy4=(TP4+TN4)/(TP4+FN4+TN4+FP4)
        print((TP4+TN4)/(TP4+FN4+TN4+FP4))
    print("F1 Score :")
    if (recall4+precision4)==0:
        f1score4=0
        print(f1score4)
    else:
        f1score4=(2 * recall4*precision4)/(recall4+precision4)
        print(f1score4)
    result4.append("K-NN With Spelling Chceker & Wordnet")
    result4.append(TP4)
    result4.append(TN4)
    result4.append(FP4)
    result4.append(FN4)
    result4.append(precision4)
    result4.append(recall4)
    result4.append(accuracy4*100)
    result4.append(f1score4*100)
    print(result4)


    print("------------------Naive Bayes Wordnet---------------")
    TP5=0
    TN5=0
    FP5=0
    FN5=0
    i=0
    
    print(hasil5)
    print(hasil7)
    for i in range(len(hasil5)):
        if hasil5[i] =="positif" and hasil5[i]==hasil7[i]:
            TP5=TP5+1
        if hasil5[i] =="negatif" and hasil5[i]==hasil7[i]:
            TN5=TN5+1
        if hasil5[i] =="netral" and hasil5[i]==hasil7[i]:
            TP5=TP5+1
        if hasil5[i] =="negatif" and hasil7[i]=="netral":
            FP5=FP5+1
        if hasil5[i] =="positif" and hasil7[i]=="netral":
            FN5=FN5+1
        if hasil5[i] =="netral" and hasil7[i]=="positif":
            FN5=FN5+1
        if hasil5[i] =="netral" and hasil7[i]=="negatif":
            FN5=FN5+1    
        i=0
    
    print("Hasil TP")
    print(TP5)
    print("Hasil TN")
    print(TN5)
    print("Hasil FP")
    print(FP5)
    print("Hasil FN")
    print(FN5)
    print("Precision :")
    if (TP5+FP5)==0:
        precision5=0
        print("0")
    else:
        precision5=TP5/(TP5+FP5)
        print(TP5/(TP5+FP5))
    print("Recall :")
    if (TP5+FN5)==0:
        recall5=0
        print("0")
    else:
        recall5=TP5/(TP5+FN5)
        print(TP5/(TP5+FN5))
    print("Accuracy :")
    if (TP5+FN5+TN5+FP5)==0:
        accuracy5=0
        print("0")
    else:
        accuracy5=(TP5+TN5)/(TP5+FN5+TN5+FP5)
        print((TP5+TN5)/(TP5+FN5+TN5+FP5))
    print("F1 Score :")
    if (recall5+precision5)==0:
        f1score5=0
        print(f1score5)
    else:
        f1score5=(2 * recall5*precision5)/(recall5+precision5)
        print(f1score5)
    result5.append("Naive Bayes With Spelling Chceker & Wordnet")
    result5.append(TP5)
    result5.append(TN5)
    result5.append(FP5)
    result5.append(FN5)
    result5.append(precision5)
    result5.append(recall5)
    result5.append(accuracy5*100)
    result5.append(f1score5*100)
    print(result5)


    print("------------------Decition Tree Wordnet---------------")
    TP6=0
    TN6=0
    FP6=0
    FN6=0
    i=0
    print(hasil5)
    print(hasil8)
    for i in range(len(hasil5)):
        if hasil5[i] =="positif" and hasil5[i]==hasil8[i]:
            TP6=TP6+1
        if hasil5[i] =="negatif" and hasil5[i]==hasil8[i]:
            TN6=TN6+1
        if hasil5[i] =="netral" and hasil5[i]==hasil8[i]:
            TP6=TP6+1
        if hasil5[i] =="negatif" and hasil8[i]=="netral":
            FP6=FP6+1
        if hasil5[i] =="positif" and hasil8[i]=="netral":
            FN6=FN6+1
        if hasil5[i] =="netral" and hasil8[i]=="positif":
            FN6=FN6+1
        if hasil5[i] =="netral" and hasil8[i]=="negatif":
            FN6=FN6+1
        i=0
    
    print("Hasil TP")
    print(TP6)
    print("Hasil TN")
    print(TN6)
    print("Hasil FP")
    print(FP6)
    print("Hasil FN")
    print(FN6)
    print("Precision :")
    if (TP6+FP6)==0:
        precision6=0
        print("0")
    else:
        precision6=TP6/(TP6+FP6)
        print(TP6/(TP6+FP6))
    print("Recall :")
    if (TP6+FN6)==0:
        recall6=0
        print("0")
    else:
        recall6=TP6/(TP6+FN6)
        print(TP6/(TP6+FN6))
    print("Accuracy :")
    if (TP6+FN6+TN6+FP6)==0:
        accuracy6=0
        print("0")
    else:
        accuracy6=(TP6+TN6)/(TP6+FN6+TN6+FP6)
        print((TP6+TN6)/(TP6+FN6+TN6+FP6))
    print("F1 Score :")
    if (recall6+precision6)==0:
        f1score6=0
        print(f1score6)
    else:
        f1score6=(2 * recall6*precision6)/(recall6+precision6)
        print(f1score6)
    result6.append("Decition Tree With Spelling Chceker & Wordnet")
    result6.append(TP6)
    result6.append(TN6)
    result6.append(FP6)
    result6.append(FN6)
    result6.append(precision6)
    result6.append(recall6)
    result6.append(accuracy6*100)
    result6.append(f1score6*100)
    print(result6)
 
    print("-----")

    print(result1)
    print(result2)
    print(result3)
    print(result4)
    print(result5)
    print(result6)

    tree5copy.insert('', 'end', text=1, values=(result1[0],result1[1],result1[2],result1[3],result1[4],result1[5],result1[6],str(result1[7])+" %",str(result1[8])+" %"))
    tree5copy.insert('', 'end', text=2, values=(result2[0],result2[1],result2[2],result2[3],result2[4],result2[5],result2[6],str(result2[7])+" %",str(result2[8])+" %"))
    tree5copy.insert('', 'end', text=3, values=(result3[0],result3[1],result3[2],result3[3],result3[4],result3[5],result3[6],str(result3[7])+" %",str(result3[8])+" %"))
    tree5copy.insert('', 'end', text=4, values=(result4[0],result4[1],result4[2],result4[3],result4[4],result4[5],result4[6],str(result4[7])+" %",str(result4[8])+" %"))
    tree5copy.insert('', 'end', text=5, values=(result5[0],result5[1],result5[2],result5[3],result5[4],result5[5],result5[6],str(result5[7])+" %",str(result5[8])+" %"))
    tree5copy.insert('', 'end', text=6, values=(result6[0],result6[1],result6[2],result6[3],result6[4],result6[5],result6[6],str(result6[7])+" %",str(result6[8])+" %"))
    

    global presisi1
    global presisi2
    global presisi3
    global presisi4
    global presisi5
    global presisi6
    global recall_1
    global recall_2
    global recall_3
    global recall_4
    global recall_5
    global recall_6
    global akurasi1
    global akurasi2
    global akurasi3
    global akurasi4
    global akurasi5
    global akurasi6
    global fig
    global ax
    akurasi1=round(result1[7])
    akurasi2=round(result2[7])
    akurasi3=round(result3[7])
    akurasi4=round(result4[7])
    akurasi5=round(result5[7])
    akurasi6=round(result6[7])

    fig, ax = plt.subplots()
    chart()
    fig.tight_layout()
    plt.show()
   


   

def main():
    global Import_Corpus_TextCorpus
    global listboxDatacorpus
    global fr_kanan_tab_tweet_viewtweet1
    global fr_kanan_tab_tweet_viewtweet2
    global fr_kanan_tab_proses1_viewtweet1
    global fr_kanan_tab_proses1_viewtweet2
    global sample,sample1
    global fr_kanan_tab_proses1
    global fr_kanan_tab_proses1_TabelKiri
    global fr_kanan_tab_proses1_TabelKanan
    global tree1
    Sample=["baik","keren","bahagia","bantu","rusak","bug","error","cepat","kreatif"]
    Sample1=["positif","positif","positif","positif","negatif","negatif","negatif","positif","positif"]
    fr_footer = tkinter.tix.Frame(mainFrame, bd=1,bg='#E3E6F0',relief=SUNKEN)
    fr_footer.pack(side=BOTTOM,expand=YES,fill=X)
    footer = tkinter.tix.Label(fr_footer, text="Copyright : 1301158593", bd=1)
    footer.pack(fill=X)
    
    #---------------------------------------------FRAME KIRI----------------------------------------
    fr_kiri = tkinter.tix.Frame(mainFrame, bd=2,relief=SUNKEN,bg='#E3E6F0')
    fr_kiri.pack(side=LEFT,expand=YES,fill=Y)
    fr_kiri_Top1=tkinter.tix.Frame(fr_kiri, bd=2,relief=SUNKEN,heigh=10,bg='white')
    fr_kiri_Top1.pack(side=TOP,anchor=NW,fill=X)
    fr_kiri_Top2=tkinter.tix.Frame(fr_kiri, bd=2,relief=SUNKEN,heigh=10,bg='silver')
    fr_kiri_Top2.pack(side=TOP)
    fr_kiri_Top2_Import_Corpus=LabelFrame(fr_kiri_Top2,text='Import Corpus',bd=1,bg='silver')
    fr_kiri_Top2_Import_Corpus.pack()
    
    fr_kiri_Top2_Import_Corpus_TextCorpus=Frame(fr_kiri_Top2_Import_Corpus,bd=1,relief=SUNKEN,bg='silver')
    fr_kiri_Top2_Import_Corpus_TextCorpus.pack(fill=X,expand=YES)

    """
    Import_Corpus_TextCorpus=Entry(fr_kiri_Top2_Import_Corpus_TextCorpus,bg='white',width=10)
    Import_Corpus_TextCorpus.pack(side=LEFT,ipadx=5)
    
    Button(fr_kiri_Top2_Import_Corpus_TextCorpus,text='Browser',command=lambda:opendir(fr_kiri_Top2_Import_Corpus_TextCorpus)).pack(side=RIGHT)
    """
    """global teksinput
    teksinput=Entry(fr_kiri_Top2_Import_Corpus_TextCorpus,bg='white')
    teksinput.pack(side=LEFT,ipadx=5)
    Button(fr_kiri_Top2_Import_Corpus_TextCorpus,text='Browser',command=lambda:opendir(fr_kiri_Top2_Import_Corpus_TextCorpus)).pack(side=RIGHT)
    """
    #Button(fr_kiri_Top2_Import_Corpus_TextCorpus,text='Search',command=lambda:Cari(teksinput)).pack(side=RIGHT)

    fr_kiri_Top2_Import_Corpus_corpus2=Frame(fr_kiri_Top2_Import_Corpus,bd=1,bg='silver')
    fr_kiri_Top2_Import_Corpus_corpus2.pack(fill=Y)
    
    scrollcorpusa = Scrollbar( fr_kiri_Top2_Import_Corpus_corpus2, orient=VERTICAL)
    scrollcorpusa.pack(side=RIGHT, fill=Y)
    
    scrollcorpusb = Scrollbar( fr_kiri_Top2_Import_Corpus_corpus2, orient=HORIZONTAL)
    scrollcorpusb.pack(side=BOTTOM, fill=X)
    
    listboxDatacorpus = Listbox( fr_kiri_Top2_Import_Corpus_corpus2,yscrollcommand=scrollcorpusa.set,xscrollcommand=scrollcorpusb.set,width=17,height=40)
    listboxDatacorpus.pack(fill=X)
    listboxDatacorpus.bind('<<ListboxSelect>>',CurSelet)
    #---------------------------------------------FRAME KANAN----------------------------------------

    global fr_kanan_tab_proses1_TabelKiri_Atas
    global fr_kanan_tab_proses1_TabelKiri_bawah
    fr_kanan = tkinter.tix.Frame(mainFrame, bd=2,relief=SUNKEN)
    fr_kanan.pack(side=LEFT,expand=YES,fill=Y)

    fr_kanan_Atas = tkinter.tix.Frame(fr_kanan ,bd=2,relief=SUNKEN)
    fr_kanan_Atas.pack(side=TOP,fill=X)
    Button(fr_kanan_Atas,text='Import Data Twitter',command=lambda:opendir(fr_kiri_Top2_Import_Corpus_TextCorpus)).pack(side=LEFT,fill=X)
    Import_Corpus_TextCorpus=Entry(fr_kanan_Atas,bg='white',width=40)
    Import_Corpus_TextCorpus.pack(side=LEFT, fill=X)
    Label(fr_kanan_Atas,text="Recall ( K-NN ) :").pack(fill=X,side=LEFT)
    Entry1=Entry(fr_kanan_Atas,bg='white',width=5)
    Entry1.pack(side=LEFT, fill=X)
    Label(fr_kanan_Atas,text="Precision ( K-NN ) :").pack(fill=X,side=LEFT)
    Entry2=Entry(fr_kanan_Atas,bg='white',width=5)
    Entry2.pack(side=LEFT, fill=X)
    Label(fr_kanan_Atas,text="Recall ( K-NN & WordNet) :").pack(fill=X,side=LEFT)
    Entry3=Entry(fr_kanan_Atas,bg='white',width=5)
    Entry3.pack(side=LEFT, fill=X)
    Label(fr_kanan_Atas,text="Precision ( K-NN & WordNet) :").pack(fill=X,side=LEFT)
    Entry4=Entry(fr_kanan_Atas,bg='white',width=5)
    Entry4.pack(side=LEFT, fill=X)
    
    

    fr_kanan_Bawah = tkinter.tix.Frame(fr_kanan, bd=2,relief=SUNKEN)
    fr_kanan_Bawah.pack(side=BOTTOM,expand=YES,fill=Y)
    fr_kanan_tab=tkinter.tix.NoteBook(fr_kanan_Bawah, name='nb1')
    fr_kanan_tab['bg']='blue'
    fr_kanan_tab.nbframe['backpagecolor']='silver'
    fr_kanan_tab.add('tweet', label="List of Tweets", underline=0)
    fr_kanan_tab.add('proses1', label="Calculating K-NN Methode & WordNet", underline=0)
    fr_kanan_tab.add('result', label="Resut of Method", underline=0)
    fr_kanan_tab.add('result2', label="Performance Method", underline=0)
    fr_kanan_tab.pack(expand=2, fill=tkinter.tix.BOTH, padx=0,pady=0, side=tkinter.tix.TOP)
    
    
    fr_kanan_tab_tweet=fr_kanan_tab.tweet
    fr_kanan_tab_tweet_Kiri = tkinter.tix.Frame(fr_kanan_tab_tweet,width=170)
    fr_kanan_tab_tweet_Kiri .pack(side=tkinter.LEFT,fill=tkinter.Y)
    fr_kanan_tab_tweet_Kanan = tkinter.tix.Frame(fr_kanan_tab_tweet,width=170)
    fr_kanan_tab_tweet_Kanan .pack(side=tkinter.LEFT,fill=tkinter.X)
    fr_kanan_tab_tweet_Kiri_Label=LabelFrame(fr_kanan_tab_tweet_Kiri,text='With Out WordNet',bd=1,bg='silver')
    fr_kanan_tab_tweet_Kiri_Label.pack(side=LEFT, fill=X)
    fr_kanan_tab_tweet_Kanan_Label=LabelFrame(fr_kanan_tab_tweet_Kanan,text='With WordNet',bd=1,bg='silver')
    fr_kanan_tab_tweet_Kanan_Label.pack(side=RIGHT, fill=X)
    
    fr_kanan_tab_tweet_viewtweet1 = tkinter.Text(fr_kanan_tab_tweet_Kiri_Label,height=60,width=62)
    fr_kanan_tab_tweet_viewtweet1.pack(side=tkinter.LEFT,fill=tkinter.Y)
    fr_kanan_tab_tweet_viewtweet2 = tkinter.Text(fr_kanan_tab_tweet_Kanan_Label,height=60,width=62)
    fr_kanan_tab_tweet_viewtweet2.pack(side=tkinter.RIGHT,fill=tkinter.Y)
    

    fr_kanan_tab_proses1=fr_kanan_tab.proses1
    fr_kanan_tab_proses1_TabelKiri = LabelFrame(fr_kanan_tab_proses1,text='TF-IDF & Labeling WithOut WordNet',width=170)
    fr_kanan_tab_proses1_TabelKiri .pack(side=tkinter.LEFT,fill=tkinter.Y)
    fr_kanan_tab_proses1_TabelKanan = LabelFrame(fr_kanan_tab_proses1,text='TF-IDF & Labeling With WordNet',width=170)
    fr_kanan_tab_proses1_TabelKanan .pack(side=tkinter.LEFT,fill=tkinter.Y)
    
    fr_kanan_tab_proses1_TabelKiri_TOP = tkinter.tix.Frame(fr_kanan_tab_proses1_TabelKiri,width=70, bd=2,bg='#E3E6F0')
    fr_kanan_tab_proses1_TabelKiri_TOP.pack(side=TOP,fill=tkinter.Y)
    fr_kanan_tab_proses1_TabelKiri_TOP_LEFT = tkinter.tix.Frame(fr_kanan_tab_proses1_TabelKiri_TOP,width=70, bd=2,bg='#E3E6F0')
    fr_kanan_tab_proses1_TabelKiri_TOP_LEFT.pack(side=LEFT)
    fr_kanan_tab_proses1_TabelKiri_TOP_RIGHT = tkinter.tix.Frame(fr_kanan_tab_proses1_TabelKiri_TOP,width=70, bd=2,bg='#E3E6F0')
    fr_kanan_tab_proses1_TabelKiri_TOP_RIGHT.pack(side=LEFT)
    fr_kanan_tab_proses1_TabelKiri_BOTTOM = tkinter.tix.Frame(fr_kanan_tab_proses1_TabelKiri,width=70, bd=2,bg='#E3E6F0')
    fr_kanan_tab_proses1_TabelKiri_BOTTOM.pack(side=TOP,fill=tkinter.Y)
    fr_kanan_tab_proses1_TabelKiri_BOTTOM_LEFT = tkinter.tix.Frame(fr_kanan_tab_proses1_TabelKiri_BOTTOM,width=70, bd=2,bg='#E3E6F0')
    fr_kanan_tab_proses1_TabelKiri_BOTTOM_LEFT.pack(side=LEFT)
    fr_kanan_tab_proses1_TabelKiri_BOTTOM_RIGHT = tkinter.tix.Frame(fr_kanan_tab_proses1_TabelKiri_BOTTOM,width=70, bd=2,bg='#E3E6F0')
    fr_kanan_tab_proses1_TabelKiri_BOTTOM_RIGHT.pack(side=LEFT)

    
    fr_kanan_tab_proses1_TabelKiri_BOTTOM2 = LabelFrame(fr_kanan_tab_proses1_TabelKiri,text='Testing Data Training & Testing',width=70, bd=2,bg='#E3E6F0')
    fr_kanan_tab_proses1_TabelKiri_BOTTOM2.pack(side=TOP,fill=tkinter.Y)
    fr_kanan_tab_proses1_TabelKiri_BOTTOM2_LEFT = tkinter.tix.Frame(fr_kanan_tab_proses1_TabelKiri_BOTTOM2,width=70, bd=2,bg='#E3E6F0')
    fr_kanan_tab_proses1_TabelKiri_BOTTOM2_LEFT.pack(side=LEFT)
    fr_kanan_tab_proses1_TabelKiri_BOTTOM2_RIGHT = tkinter.tix.Frame(fr_kanan_tab_proses1_TabelKiri_BOTTOM2,width=70, bd=2,bg='#E3E6F0')
    fr_kanan_tab_proses1_TabelKiri_BOTTOM2_RIGHT.pack(side=LEFT)
    global tree1
    global tree1copy
    global tree2
    global tree2copy
    global tree21copy
    global tree22copy
    global tree3
    global tree3copy
    global tree4
    global tree4copy
    
    tree1 = tkinter.ttk.Treeview(fr_kanan_tab_proses1_TabelKiri_TOP_LEFT, columns=('Document','TF','Words'),height=9)
    tree1.heading('#0', text='No')
    tree1.heading('#1', text='Words')
    tree1.heading('#2', text='TF')
    tree1.heading('#3', text='Document')
    tree1.column('#0', stretch=tkinter.YES,width=50)
    tree1.column('#1', stretch=tkinter.YES,width=60)
    tree1.column('#2', stretch=tkinter.YES,width=75)
    tree1.column('#3', stretch=tkinter.YES,width=75)
    tree1.grid(row=2, columnspan=2, sticky='nsew')

    tree1copy = tkinter.ttk.Treeview(fr_kanan_tab_proses1_TabelKiri_TOP_RIGHT, columns=('words', 'DF','IDF'),height=9)
    tree1copy.heading('#0', text='No')
    tree1copy.heading('#1', text='words')
    tree1copy.heading('#2', text='DF')
    tree1copy.heading('#3', text='IDF')
    tree1copy.column('#0', stretch=tkinter.YES,width=50)
    tree1copy.column('#1', stretch=tkinter.YES,width=50)
    tree1copy.column('#2', stretch=tkinter.YES,width=50)
    tree1copy.column('#3', stretch=tkinter.YES,width=75)
    tree1copy.grid(row=3, columnspan=3, sticky='nsew')
    

    
    
    tree2 = tkinter.ttk.Treeview(fr_kanan_tab_proses1_TabelKiri_BOTTOM_LEFT, columns=( 'TF x IDF','Words','Labeling'),height=9)
    tree2.heading('#0', text='Document')
    tree2.heading('#1', text='TF x IDF')
    tree2.heading('#2', text='Words')
    tree2.heading('#3', text='Labeling')
    tree2.column('#0', stretch=tkinter.YES,width=60)
    tree2.column('#1', stretch=tkinter.YES,width=50)
    tree2.column('#2', stretch=tkinter.YES,width=75)
    tree2.column('#3', stretch=tkinter.YES,width=75)
    tree2.grid(row=3, columnspan=3, sticky='nsew')
    treeview2 =tree2


    tree2copy = tkinter.ttk.Treeview(fr_kanan_tab_proses1_TabelKiri_BOTTOM_RIGHT, columns=( 'Positive','Negative','Labeling'),height=9)
    tree2copy.heading('#0', text='Document')
    tree2copy.heading('#1', text='+ ')
    tree2copy.heading('#2', text='- ')
    tree2copy.heading('#3', text='Labeling')
    tree2copy.column('#0', stretch=tkinter.YES,width=60)
    tree2copy.column('#1', stretch=tkinter.YES,width=50)
    tree2copy.column('#2', stretch=tkinter.YES,width=50)
    tree2copy.column('#3', stretch=tkinter.YES,width=70)
    tree2copy.grid(row=3, columnspan=3, sticky='nsew')
    treeview2 =tree2

    tree21copy = tkinter.ttk.Treeview(fr_kanan_tab_proses1_TabelKiri_BOTTOM2_LEFT, columns=( 'Positive','Negative','Labeling'),height=9)
    tree21copy.heading('#0', text='Document')
    tree21copy.heading('#1', text='+ ')
    tree21copy.heading('#2', text='- ')
    tree21copy.heading('#3', text='Labeling')
    tree21copy.column('#0', stretch=tkinter.YES,width=50)
    tree21copy.column('#1', stretch=tkinter.YES,width=50)
    tree21copy.column('#2', stretch=tkinter.YES,width=50)
    tree21copy.column('#3', stretch=tkinter.YES,width=70)
    tree21copy.grid(row=3, columnspan=3, sticky='nsew')
    treeview2 =tree2

    tree22copy = tkinter.ttk.Treeview(fr_kanan_tab_proses1_TabelKiri_BOTTOM2_RIGHT, columns=( '+','-','Ecludian','Labeling'),height=9)
    tree22copy.heading('#0', text='No ')
    tree22copy.heading('#1', text='+')
    tree22copy.heading('#2', text='-')
    tree22copy.heading('#3', text='Ecludian')
    tree22copy.heading('#4', text='Labeling')
    tree22copy.column('#0', stretch=tkinter.YES,width=30)
    tree22copy.column('#1', stretch=tkinter.YES,width=50)
    tree22copy.column('#2', stretch=tkinter.YES,width=50)
    tree22copy.column('#3', stretch=tkinter.YES,width=70)
    tree22copy.column('#4', stretch=tkinter.YES,width=70)
 
    tree22copy.grid(row=4, columnspan=4, sticky='nsew')
    treeview2 =tree2

   
    """
    tree23copy = tkinter.ttk.Treeview(fr_kanan_tab_proses1_TabelKiri_BOTTOM2_RIGHT, columns=('Labeling'),height=9)
    tree23copy.heading('#0', text='Document')
    tree23copy.heading('#1', text='Labeling')
    tree23copy.column('#0', stretch=tkinter.YES,width=10)
    tree23copy.column('#1', stretch=tkinter.YES,width=10)
    tree23copy.grid(row=1, columnspan=1, sticky='nsew')
    treeview2 =tree2
    """
    
    

    fr_kanan_tab_proses1_TabelKanan_TOP = tkinter.tix.Frame(fr_kanan_tab_proses1_TabelKanan,width=70, bd=2,bg='#E3E6F0')
    fr_kanan_tab_proses1_TabelKanan_TOP.pack(side=TOP,fill=tkinter.Y)
    fr_kanan_tab_proses1_TabelKanan_TOP_LEFT = tkinter.tix.Frame(fr_kanan_tab_proses1_TabelKanan_TOP,width=70, bd=2,bg='#E3E6F0')
    fr_kanan_tab_proses1_TabelKanan_TOP_LEFT.pack(side=LEFT)
    fr_kanan_tab_proses1_TabelKanan_TOP_RIGHT = tkinter.tix.Frame(fr_kanan_tab_proses1_TabelKanan_TOP,width=70, bd=2,bg='#E3E6F0')
    fr_kanan_tab_proses1_TabelKanan_TOP_RIGHT.pack(side=RIGHT)
    
    tree3 = tkinter.ttk.Treeview(fr_kanan_tab_proses1_TabelKanan_TOP_LEFT, columns=('Document', 'TF','Words'),height=9)
    tree3.heading('#0', text='No')
    tree3.heading('#1', text='Words')
    tree3.heading('#2', text='TF')
    tree3.heading('#3', text='Document')
    tree3.column('#0', stretch=tkinter.YES,width=50)
    tree3.column('#1', stretch=tkinter.YES,width=60)
    tree3.column('#2', stretch=tkinter.YES,width=75)
    tree3.column('#3', stretch=tkinter.YES,width=75)
    tree3.grid(row=3, columnspan=3, sticky='nsew')


    tree3copy = tkinter.ttk.Treeview(fr_kanan_tab_proses1_TabelKanan_TOP_RIGHT, columns=('Words', 'DF','IDF'),height=9)
    tree3copy.heading('#0', text='No')
    tree3copy.heading('#1', text='Words')
    tree3copy.heading('#2', text='DF')
    tree3copy.heading('#3', text='IDF')
    tree3copy.column('#0', stretch=tkinter.YES,width=50)
    tree3copy.column('#1', stretch=tkinter.YES,width=50)
    tree3copy.column('#2', stretch=tkinter.YES,width=50)
    tree3copy.column('#3', stretch=tkinter.YES,width=75)
    tree3copy.grid(row=3, columnspan=3, sticky='nsew')



    
    
    
    fr_kanan_tab_proses1_TabelKanan_BOTTOM = tkinter.tix.Frame(fr_kanan_tab_proses1_TabelKanan,width=70, bd=2,bg='#E3E6F0')
    fr_kanan_tab_proses1_TabelKanan_BOTTOM.pack(side=TOP)
    fr_kanan_tab_proses1_TabelKanan_BOTTOM_LEFT = tkinter.tix.Frame(fr_kanan_tab_proses1_TabelKanan_BOTTOM,width=70, bd=2,bg='#E3E6F0')
    fr_kanan_tab_proses1_TabelKanan_BOTTOM_LEFT.pack(side=LEFT)
    fr_kanan_tab_proses1_TabelKanan_BOTTOM_RIGHT = tkinter.tix.Frame(fr_kanan_tab_proses1_TabelKanan_BOTTOM,width=70, bd=2,bg='#E3E6F0')
    fr_kanan_tab_proses1_TabelKanan_BOTTOM_RIGHT.pack(side=LEFT)
    fr_kanan_tab_proses1_TabelKanan_BOTTOM2 = LabelFrame(fr_kanan_tab_proses1_TabelKanan,text='Testing Data Training & Testing',width=70, bd=2,bg='#E3E6F0')
    fr_kanan_tab_proses1_TabelKanan_BOTTOM2.pack(side=TOP,fill=tkinter.Y)
    fr_kanan_tab_proses1_TabelKanan_BOTTOM2_LEFT = tkinter.tix.Frame(fr_kanan_tab_proses1_TabelKanan_BOTTOM2,width=70, bd=2,bg='#E3E6F0')
    fr_kanan_tab_proses1_TabelKanan_BOTTOM2_LEFT.pack(side=LEFT)
    fr_kanan_tab_proses1_TabelKanan_BOTTOM2_RIGHT = tkinter.tix.Frame(fr_kanan_tab_proses1_TabelKanan_BOTTOM2,width=70, bd=2,bg='#E3E6F0')
    fr_kanan_tab_proses1_TabelKanan_BOTTOM2_RIGHT.pack(side=LEFT)

    tree4 = tkinter.ttk.Treeview(fr_kanan_tab_proses1_TabelKanan_BOTTOM_LEFT, columns=( 'TF x IDF','Words','Labeling'),height=9)
    tree4.heading('#0', text='Document')
    tree4.heading('#1', text='TF x IDF')
    tree4.heading('#2', text='Word')
    tree4.heading('#3', text='Labeling')
    tree4.column('#0', stretch=tkinter.YES,width=60)
    tree4.column('#1', stretch=tkinter.YES,width=50)
    tree4.column('#2', stretch=tkinter.YES,width=75)
    tree4.column('#3', stretch=tkinter.YES,width=75)
    tree4.grid(row=3, columnspan=3, sticky='nsew')
    treeview4 =tree4

    tree4copy = tkinter.ttk.Treeview(fr_kanan_tab_proses1_TabelKanan_BOTTOM_RIGHT, columns=('Words', 'DF','IDF'),height=9)
    tree4copy.heading('#0', text='Document')
    tree4copy.heading('#1', text='+')
    tree4copy.heading('#2', text='-')
    tree4copy.heading('#3', text='Labeling')
    tree4copy.column('#0', stretch=tkinter.YES,width=50)
    tree4copy.column('#1', stretch=tkinter.YES,width=50)
    tree4copy.column('#2', stretch=tkinter.YES,width=50)
    tree4copy.column('#3', stretch=tkinter.YES,width=75)
    tree4copy.grid(row=3, columnspan=3, sticky='nsew')
    global tree41copy
    global tree42copy
    tree41copy = tkinter.ttk.Treeview(fr_kanan_tab_proses1_TabelKanan_BOTTOM2_LEFT, columns=( 'Positive','Negative','Labeling'),height=9)
    tree41copy.heading('#0', text='Document')
    tree41copy.heading('#1', text='+ ')
    tree41copy.heading('#2', text='- ')
    tree41copy.heading('#3', text='Labeling')
    tree41copy.column('#0', stretch=tkinter.YES,width=50)
    tree41copy.column('#1', stretch=tkinter.YES,width=50)
    tree41copy.column('#2', stretch=tkinter.YES,width=50)
    tree41copy.column('#3', stretch=tkinter.YES,width=70)
    tree41copy.grid(row=3, columnspan=3, sticky='nsew')
    treeview2 =tree2

    tree42copy = tkinter.ttk.Treeview(fr_kanan_tab_proses1_TabelKanan_BOTTOM2_RIGHT, columns=( '+','-','Ecludian','Labeling'),height=9)
    tree42copy.heading('#0', text='No ')
    tree42copy.heading('#1', text='+ ')
    tree42copy.heading('#2', text='-')
    tree42copy.heading('#3', text='Ecludian')
    tree42copy.heading('#4', text='Labeling')
    tree42copy.column('#0', stretch=tkinter.YES,width=50)
    tree42copy.column('#1', stretch=tkinter.YES,width=50)
    tree42copy.column('#2', stretch=tkinter.YES,width=50)
    tree42copy.column('#3', stretch=tkinter.YES,width=70)
    tree42copy.column('#4', stretch=tkinter.YES,width=70)
    tree42copy.grid(row=4, columnspan=4, sticky='nsew')

    fr_kanan_tab_result=fr_kanan_tab.result
    fr_kanan_tab_result1 = tkinter.tix.Frame(fr_kanan_tab_result)
    fr_kanan_tab_result1.pack()
    fr_kanan_tab_result_Kiri_TOP= tkinter.tix.Frame(fr_kanan_tab_result1)
    fr_kanan_tab_result_Kiri_TOP.pack(side=tkinter.TOP)
    
    global tree5
    tree5 = tkinter.ttk.Treeview(fr_kanan_tab_result_Kiri_TOP, columns=( 'Training Labeling TF-IDF Without Wordnet','Labeling TF-IDF With Wordnet','K-NN','Naive Bayes','Decition Tree','K-NN','Naive Bayes','Decition Tree'),height=50)
    tree5.heading('#0', text='No ')
    tree5.heading('#1', text='Training TF-IDF Without Wordnet')
    tree5.heading('#2', text='Training TF-IDF With Wordnet')
    tree5.heading('#3', text='K-NN')
    tree5.heading('#4', text='Naive Bayes')
    tree5.heading('#5', text='Decition Tree')
    tree5.heading('#6', text='K-NN With ')
    tree5.heading('#7', text='Naive Bayes With')
    tree5.heading('#8', text='Decition Tree With')
    tree5.column('#0', stretch=tkinter.YES,width=30)
    tree5.column('#1', stretch=tkinter.YES,width=190)
    tree5.column('#2', stretch=tkinter.YES,width=190)
    tree5.column('#3', stretch=tkinter.YES,width=80)
    tree5.column('#4', stretch=tkinter.YES,width=80)
    tree5.column('#5', stretch=tkinter.YES,width=130)
    tree5.column('#6', stretch=tkinter.YES,width=100)
    tree5.column('#7', stretch=tkinter.YES,width=100)
    tree5.column('#8', stretch=tkinter.YES,width=130)
    tree5.grid(row=8, columnspan=8, sticky='nsew')


    
    fr_kanan_tab_result2=fr_kanan_tab.result2
    fr_kanan_tab_result22 = tkinter.tix.Frame(fr_kanan_tab_result2)
    fr_kanan_tab_result22.pack()
    fr_kanan_tab_result_Kiri_BOTTOM = tkinter.tix.Frame(fr_kanan_tab_result22)
    fr_kanan_tab_result_Kiri_BOTTOM.pack(side=tkinter.TOP,fill=X)
    global tree5copy
    tree5copy = tkinter.ttk.Treeview(fr_kanan_tab_result_Kiri_BOTTOM, columns=( 'Method','TP','TN','FP','FN','FP','Precision','Recall','F1 Score'),height=30)
    tree5copy.heading('#0', text='No ')
    tree5copy.heading('#1', text='Method')
    tree5copy.heading('#2', text='TP')
    tree5copy.heading('#3', text='TN')
    tree5copy.heading('#4', text='FP')
    tree5copy.heading('#5', text='FN')
    tree5copy.heading('#6', text='Precision')
    tree5copy.heading('#7', text='Recall')
    tree5copy.heading('#8', text='Accuracy')
    tree5copy.heading('#9', text='F1 Score')
    tree5copy.column('#0', stretch=tkinter.YES,width=50)
    tree5copy.column('#1', stretch=tkinter.YES,width=270)
    tree5copy.column('#2', stretch=tkinter.YES,width=60)
    tree5copy.column('#3', stretch=tkinter.YES,width=60)
    tree5copy.column('#4', stretch=tkinter.YES,width=60)
    tree5copy.column('#5', stretch=tkinter.YES,width=60)
    tree5copy.column('#6', stretch=tkinter.YES,width=100)
    tree5copy.column('#7', stretch=tkinter.YES,width=100)
    tree5copy.column('#8', stretch=tkinter.YES,width=100)
    tree5copy.column('#9', stretch=tkinter.YES,width=100)
    tree5copy.grid(row=9, columnspan=7, sticky='nsew')

    #---------------------------------------------FRAME KANAN 2----------------------------------------
    fr_kanan2 = tkinter.tix.Frame(mainFrame, bd=2,relief=SUNKEN,bg='#E3E6F0')
    fr_kanan2.pack(side=LEFT,expand=YES,fill=Y)
    fr_kanan2_Top1=tkinter.tix.Frame(fr_kanan2, bd=2,relief=SUNKEN,heigh=10,bg='white')
    fr_kanan2_Top1.pack(side=TOP,anchor=NW,fill=X)
    fr_kanan2_Top2=tkinter.tix.Frame(fr_kanan2, bd=2,relief=SUNKEN,heigh=10,bg='silver')
    fr_kanan2_Top2.pack(side=TOP)
    fr_kanan2_Top2_Import_Corpus=LabelFrame(fr_kanan2_Top2,text='Import Corpus',bd=1,bg='silver')
    fr_kanan2_Top2_Import_Corpus.pack()
    
    fr_kanan2_Top2_Import_Corpus_TextCorpus=Frame(fr_kanan2_Top2_Import_Corpus,bd=1,relief=SUNKEN,bg='silver')
    fr_kanan2_Top2_Import_Corpus_TextCorpus.pack(fill=X,expand=YES)

   
    global listboxDatacorpus1
    global listboxDatacorpus2
    listboxDatacorpus1 = Listbox( fr_kanan2_Top2_Import_Corpus_TextCorpus,width=15,height=40)
    listboxDatacorpus1.pack(fill=X, side=LEFT)
    listboxDatacorpus2 = Listbox( fr_kanan2_Top2_Import_Corpus_TextCorpus,width=15,height=40)
    listboxDatacorpus2.pack(fill=X,side=RIGHT)
    
    
 

 


def color_config(widget, color, event):
    widget.configure(background=color)

root = tkinter.tix.Tk()
mainFrame = tkinter.tix.Frame(root)
mainFrame.pack(fill=NONE, expand=YES)

stop=stopwords.words('english')
vdok=StringVar()
main()
root.mainloop()





