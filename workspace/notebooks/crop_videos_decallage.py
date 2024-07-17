from hashlib import new
from ipaddress import AddressValueError
import sys
import moviepy
import pandas as pd 
from moviepy.editor import *
from datetime import  datetime
import numpy as np 
import os
import  copy


#########################################################
######pretraitement necessaire pour les correspondances## 
#########################################################

def extract_interesting_sections(interesting_sections):
    ##on stocke les videos dans un format date debut date fin 
    availabe_videos=pd.DataFrame(columns=["names","start time","end time"])
    #we shall store all the file names in this list
    filelist = []
    #paths =["/home/ulaval.ca/amngb2/projects/ul-val-prj-def-erpaq33/sante_veau/videos/2022-03-12-2022-03-17","/home/ulaval.ca/amngb2/projects/ul-val-prj-def-erpaq33/sante_veau/videos/2022-03-18-2022-03-23"]
    paths =["/home/ulaval.ca/amngb2/projects/ul-val-prj-def-erpaq33/sante_veau/videos/"+folder for folder in os.listdir("/home/ulaval.ca/amngb2/projects/ul-val-prj-def-erpaq33/sante_veau/videos") if "$" not in folder]
    for path in paths: 
        for root, dirs, files in os.walk(path):
            for file in files:
                #append the file name to the list
                filelist.append(os.path.join(root,file))
    for name in filelist:
        #print(name)
        availabe_video = {}
        availabe_video["names"]=name
        availabe_video["start time"]=(name.split(".mp4")[0].split("_")[-2])[:-6]+" "+(name.split(".mp4")[0].split("_")[-2])[-6:-4]+":"+(name.split(".mp4")[0].split("_")[-2])[-4:-2]+":"+(name.split(".mp4")[0].split("_")[-2])[-2:] #for name in  availabe_videos["names"].values]
        availabe_video["end time"]=(name.split(".mp4")[0].split("_")[-1])[:-6]+" "+(name.split(".mp4")[0].split("_")[-1])[-6:-4]+":"+(name.split(".mp4")[0].split("_")[-1])[-4:-2]+":"+(name.split(".mp4")[0].split("_")[-1])[-2:]  #for name in  availabe_videos["names"].values]
        availabe_videos =availabe_videos.append(availabe_video, ignore_index=True)#pd.concat ([availabe_videos, availabe_video], axis=1,ignore_index=True)
        
    availabe_videos["start time"] = pd.to_datetime(availabe_videos['start time'],infer_datetime_format=True) # format ='%y%m%d %H:%M:%S')
    availabe_videos["end time"] = pd.to_datetime(availabe_videos['end time'],infer_datetime_format=True) 

    #############                                 ###########
    #############        Extracting videos        ###########
    #############                                 ###########

    def crop_video(video_name, new_name, start, end):
        video = VideoFileClip(video_name).subclip(start,end)
        return video
        

    def search_interesting_section (interesting_section):
        
        for video_idx, availabe_video in availabe_videos.iterrows():
            if "_ch"+str(interesting_section["station"])+"_" in availabe_video["names"] :
                if interesting_section["start visit dateTime"]>= availabe_video["start time"] and interesting_section["start visit dateTime"] <= availabe_video["end time"]:
                    print("hi",availabe_video["names"])
                    print(interesting_section["start visit dateTime"]>= availabe_video["start time"], interesting_section["start visit dateTime"], availabe_video["start time"])

                    start=  (interesting_section["start visit dateTime"] - availabe_video["start time"]).total_seconds()
                    end=  (interesting_section["end visit dateTime"] - availabe_video["start time"]).total_seconds()

                    rest = -(availabe_video["end time"] - interesting_section["end visit dateTime"]).total_seconds()
                    end_video= (availabe_video["end time"] - availabe_video["start time"]).total_seconds() #-1
                    print("**",start,end, "the rest is", rest)

                    if rest>0:
                        video = crop_video(availabe_video["names"], interesting_section["start visit dateTime"].strftime("%d%m%Y%H%M%S")+"__to__"+interesting_section["end visit dateTime"].strftime("%d%m%Y%H%M%S"), start, end_video )
                        print("we should take in another video")
                        rest_interesting_section=copy.deepcopy(interesting_section)
                        rest_interesting_section["start visit dateTime"]= availabe_video["end time"]+pd.to_timedelta(1, unit='s')
                        rest_video = search_interesting_section (rest_interesting_section)

                        if rest_video!=None:
                            final = concatenate_videoclips([video, rest_video])
                            return final
                        else:
                            return video

                    else:
                        print("we crop")
                        video = crop_video(availabe_video["names"], interesting_section["start visit dateTime"].strftime("%d%m%Y%H%M%S")+"__to__"+interesting_section["end visit dateTime"].strftime("%d%m%Y%H%M%S"), start, end )
                        return video  
            
        return None

    #this is to  keep trace of videos we have already created         
    path ="/home/ulaval.ca/amngb2/projects/ul-val-prj-def-erpaq33/sante_veau/dataset/coupure_video_veaux/"
    filelist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            #append the file name to the list
            filelist.append(os.path.join(root,file))
        


    for section_idx, interesting_section in interesting_sections.iterrows():
        new_name="vealnum_"+str(interesting_section["calfNumber"])+"_ch"+str(interesting_section["station"])+"_from_"+interesting_section["start visit dateTime"].strftime("%d%m%Y%H%M%S")+"__to__"+interesting_section["end visit dateTime"].strftime("%d%m%Y%H%M%S")
        path= "/home/ulaval.ca/amngb2/projects/ul-val-prj-def-erpaq33/sante_veau/dataset/coupure_video_veaux/"+interesting_section["start visit dateTime"].strftime("%d%m%Y")
        print("let us check if ",new_name, "is of interest and isn't already created")
        if (path+"/"+new_name+".mp4" not in filelist) and (pd.to_datetime(interesting_section["start visit dateTime"].date()) in [datetime(2022, 3, 15),datetime(2022, 3, 18),datetime(2022, 3, 21),datetime(2022, 3, 23),datetime(2022, 3, 25) ]) :#,datetime(2022, 2, 20), datetime(2022, 2, 27) datetime(2022, 3, 6), datetime(2022, 3,13)] :
            #,datetime(2022, 2, 20), datetime(2022, 2, 27) datetime(2022, 3, 6), datetime(2022, 3,13)] :
            print("***** yes, it is. we will create",new_name+".mp4")
            video= search_interesting_section(interesting_section)
            if video!=None :
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
                    print("The new directory is created!")
                
                path2 = "../../sante_veau/dataset/label_veaux/"+interesting_section["start visit dateTime"].strftime("%d%m%Y")
                isLExist = os.path.exists(path2)
                if not isLExist:
                    os.makedirs(path2)
                    print("The dataset directory is created!")
                print("******are createting","sante_veau/dataset/coupure_video_veaux/"+interesting_section["start visit dateTime"].strftime("%d%m%Y")+"/"+new_name, "***********FPS",video.fps)

                video.write_videofile(path+"/"+new_name+".mp4")#,fps=25)
            else:
                print("we haven't find the sample video so we will go next")
    
        print("we finished the treatment")

import copy
interesting_sections=pd.read_csv("/home/ulaval.ca/amngb2/projects/ul-val-prj-def-erpaq33/sophie/louves/visits_point.csv", sep=";")
interesting_sections=interesting_sections.loc[interesting_sections['Duration'] !=0]
interesting_sections=interesting_sections[["date", "station", "Duration", "calfNumber","feederLong"]]
#### We will replace station to be the number corresponding to the x on ch_x on each video#####
interesting_sections.loc[(interesting_sections["station"]==1) & (interesting_sections["feederLong"]=="DAL 2 (2496)"), 'parc']="9" #louve 3
interesting_sections.loc[(interesting_sections["station"]==2) & (interesting_sections["feederLong"]=="DAL 2 (2496)"), 'parc']="10" #louve4
interesting_sections.loc[(interesting_sections["station"]==2) & (interesting_sections["feederLong"]=="DAL 1 (2494)"), 'parc']="1"  #louve 2 
interesting_sections.loc[(interesting_sections["station"]==1) & (interesting_sections["feederLong"]=="DAL 1 (2494)"), 'parc']="2" #louve 1
interesting_sections["station"]=interesting_sections["parc"]
#interesting_sections["station"] =interesting_sections["station"].replace([1,2],[2,1])

interesting_sections["start visit dateTime"]= copy.deepcopy(pd.to_datetime(interesting_sections["date"]))
interesting_sections["end visit dateTime"]= interesting_sections["start visit dateTime"] + pd.to_timedelta(interesting_sections['Duration']+150, unit='s') + pd.to_timedelta(5, unit='h')
interesting_sections["start visit dateTime"]= pd.to_datetime(interesting_sections["date"])- pd.to_timedelta(90, unit='s') + pd.to_timedelta(5, unit='h')#.agg(' '.join, axis=1))#.dt.time
#print("sophie*******",interesting_sections.loc[ (interesting_sections["date"].str.contains ("2022-03-06")) & (interesting_sections["calfNumber"]==6371) ])  #["2022-03-06" in str(interesting_sections["date"]) ])# & (interesting_sections["calfNumber"]=="6371")])
"""#####to remove 
interesting_sections["date2"]=pd.to_datetime(interesting_sections["start visit dateTime"].dt.date)

interesting_sections =interesting_sections.loc[(interesting_sections["date2"].isin( [datetime(2022, 3, 23),(2022,3,25) ]))]
#interesting_sections =interesting_sections.loc[(interesting_sections["calfNumber"]==6422)]
"""
#####
interesting_sections= interesting_sections.sort_values(['calfNumber','start visit dateTime'],ascending=True).groupby(('calfNumber')).head(interesting_sections.shape[0])

interesting_sections=interesting_sections[[ "station", "calfNumber","end visit dateTime","start visit dateTime" ,"feederLong"]]
interesting_sections.reset_index(drop=True, inplace=True)
pd.set_option('display.max_rows', None)

to_drop=[]
for idx in interesting_sections.index.values[:-1] :
    if interesting_sections.iloc[idx]["calfNumber"]== interesting_sections.iloc[idx+1]["calfNumber"] and interesting_sections.iloc[idx]["end visit dateTime"]>= interesting_sections.iloc[idx+1]["start visit dateTime"]- pd.to_timedelta(3, unit='min'):  # en gros 90s + 90s de marge d'après les lignes 175 ete 176
        interesting_sections["start visit dateTime"].iloc[idx+1] =  interesting_sections.iloc[idx]["start visit dateTime"]
        to_drop.append(idx)
    """elif interesting_sections.iloc[idx]["calfNumber"]== interesting_sections.iloc[idx+1]["calfNumber"] and interesting_sections.iloc[idx]["start visit dateTime"]== interesting_sections.iloc[idx+1]["start visit dateTime"]: 
        first_end=interesting_sections.iloc[idx]["end visit dateTime"]
        second_end=interesting_sections.iloc[idx+1]["end visit dateTime"]
        interesting_sections.iloc[idx+1]["end visit dateTime"] = first_end if  first_end > second_end else second_end
        print("***",interesting_sections.iloc[idx+1]["end visit dateTime"]) 
        to_drop.append(idx)"""
        
          
interesting_sections.drop(index=to_drop, inplace=True)
interesting_sections.reset_index(drop=True, inplace=True)


interesting_sections.reset_index(drop=True, inplace=True)
interesting_sections["end visit dateTime"]=pd.to_datetime(interesting_sections["end visit dateTime"])
interesting_sections["start visit dateTime"]=pd.to_datetime(interesting_sections["start visit dateTime"])
interesting_sections["Duration"]= (interesting_sections["end visit dateTime"] -interesting_sections["start visit dateTime"]).dt.total_seconds()
interesting_sections["date2"]=pd.to_datetime(interesting_sections["start visit dateTime"].dt.date)

interesting_sections =interesting_sections.loc[(interesting_sections["date2"].isin( [datetime(2022, 3, 15),datetime(2022, 3, 18),datetime(2022, 3, 21),datetime(2022, 3, 23),datetime(2022, 3, 25) ]))]

interesting_sections = interesting_sections.sort_values(by='start visit dateTime',ascending=True)
extract_interesting_sections(interesting_sections)


#interesting_sections =interesting_sections.where(interesting_sections["start visit dateTime"].isin( [datetime(2022, 3, 14), datetime(2022, 3, 21) ]))
#interesting_sections.to_excel("to_read_distribution.xlsx")
#exit()


"""
***************************
interesting_sections= copy.deepcopy(All)

interesting_sections =interesting_sections[interesting_sections["feederLong"]=="DAL 2 (2496)"]
interesting_sections.loc[ (interesting_sections["feederLong"]=="DAL 2 (2496)"), "end visit dateTime"]= interesting_sections["start visit dateTime"] + pd.to_timedelta(interesting_sections['Duration']+90, unit='s') + pd.to_timedelta(6, unit='h')
interesting_sections.loc[ (interesting_sections["feederLong"]=="DAL 2 (2496)"), "start visit dateTime"]= pd.to_datetime(interesting_sections["date"])- pd.to_timedelta(90, unit='s') + pd.to_timedelta(6, unit='h')#.agg(' '.join, axis=1))#.dt.time
"""

    

"""for interesting_section in interesting_sections :
    calfNumber =interesting_section["CalfNumber"]
    print(type(calfNumber))

    for other_interesting_section in interesting_sections.loc[interesting_sections["CalfNumber"]==calfNumber]:

        if interesting_section["end visit dateTime"]>= other_interesting_section["start visit dateTime"] and interesting_section["end visit dateTime"] other_interesting_section["start visit dateTime"]:  # en gros 90s + 90s de marge d'après les lignes 175 ete 176
         """