#start imports 
import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
#End imports
st.markdown(f'<H1 style="color:green; font-size:75px;">{"Pick 16"}</H1>', unsafe_allow_html=True)
st.markdown(f'<H6 style="color:darkgreen; font-size:25px;">{"A tool designed to assit in building a fantasy football championship roster"}</H6>', unsafe_allow_html=True)
st.markdown(f'<body>{"Instructions:"}</body>', unsafe_allow_html=True)
st.markdown(f'<body>{"Pick one starting quarterback (1)"}</body>', unsafe_allow_html=True)
st.markdown(f'<body>{"Pick two starting runningbacks (2-3)"}</body>', unsafe_allow_html=True)
st.markdown(f'<body>{"Pick two starting wide recievers (2-3)"}</body>', unsafe_allow_html=True)
st.markdown(f'<body>{"Pick one flex player either an additional running back or an additional wide reciever (1)"}</body>', unsafe_allow_html=True)
st.markdown(f'<body>{"Pick one starting tightend (1) "}</body>', unsafe_allow_html=True)
'\n'
'\n'
#Start scarping NFL fantasy stats from pro football refrence
#Start create original dataframe (Com_Stats - contains every fantasy position with all stats)
url='https://www.pro-football-reference.com/years/2022/fantasy.htm'
Complete_stats_df= pd.read_html(url)
Com_Stats = pd.DataFrame(Complete_stats_df [0])
#End create original dataframe 
#Start edit cleaning Com_Stats to allow editing further down the line
    #Drops the orginal level 1 of the data frame and rows that are not need and that will affect editing 
    #Renames all columns do to repeats and poor naming 
    #Fills in "Nones" and changes the data types of the majority of the columns in the dataframe  
    #Ensures player names are uniform
Com_Stats = Com_Stats.droplevel(level=0, axis=1)
Com_Stats.columns=['Rank', 'Player','Team_Name','Position', 'Age','Games_Played','Games_Started', 'Passes_Completed','Passing_Att','Passing_Yds','Passing_TDs','Ints', 'Rushing_Att','Rushing_Yds','Rushing_Yds/Att', 'Rushing_TDs','Targets','Receptions', 'Recieving_Yds','Recieving_Yds/Rec','Recieving_TDs', 'Fumbles','Fumbles_Lost','Total_Tds', '2PM','2PP','Ftpts', 'PPR','DKpt','FDpt','VBD', 'PosRank','OVRank']
Com_Stats=Com_Stats.drop([29,60,91,122,153,184,215,246,277,308,339,370,401,432,463,494,525,556,587,618,649,])
Com_Stats = Com_Stats.fillna(0)
convert_dict ={'Rank':np.int64,
               'Age':np.int64,
               'Games_Played':np.int64,
               'Games_Started':np.int64,
               'Passes_Completed':np.int64,
               'Passing_Att':np.int64,
               'Passing_Yds':np.int64,
               'Passing_TDs':np.int64,
               'Ints':np.int64,
               'Rushing_Att':np.int64,
               'Rushing_Yds':np.int64,
               'Rushing_Yds/Att':float,
               'Rushing_TDs':np.int64,
               'Targets':np.int64,
               'Receptions':np.int64,
               'Recieving_Yds':np.int64,
               'Recieving_Yds/Rec':float,
               'Recieving_TDs':np.int64,
               'Fumbles':np.int64,
               'Fumbles_Lost':np.int64,
               'Total_Tds':np.int64,
               '2PM':np.int64,
               '2PP':np.int64,
               'Ftpts':float,
               'PPR':float,
               'DKpt':float,
               'FDpt':float,
               'VBD':np.int64,
               'PosRank':np.int64,
               'OVRank':np.int64,
}
Com_Stats = Com_Stats.astype(convert_dict )
Com_Stats['Player']=Com_Stats['Player'].str.replace('[*,+]','')
#End dataframe cleaning
#Start QB data frame
    #Creates a data frame for QBs only 
    #Drops columns that are not needed 
    #Sets the index to player Name 
    #Sorts the dataframe by greatest to least points scored 
st.markdown(f'<H6 style="color:darkgreen; font-size:25px;">{"Quarterbacks"}</H6>', unsafe_allow_html=True)    
QB_stats = Com_Stats[Com_Stats['Position'] =='QB']
QB_stats = QB_stats.drop(labels = ['Position','Rank','Team_Name',  'Targets', 'Receptions', 'Recieving_Yds', 'Recieving_Yds/Rec','Recieving_TDs', '2PM', '2PP','Ftpts', 'DKpt', 'FDpt', 'VBD', 'PosRank', 'OVRank'], axis=1)
QB_stats = QB_stats.set_index('Player')
QB_stats=QB_stats.sort_values('PPR', ascending=False)
QB_stats
#End QB data Frame
#Start Select QB
QB_Selection = st.selectbox('Select a Starting QB (1)', QB_stats.index)
QB_Modifer = st.multiselect('Select QB stat modifiers', QB_stats.columns)
'\n'
#End Select QB
#Start RB data frame
    #Creates a data frame for RBs only 
    #Drops columns that are not needed 
    #Sets the index to player Name 
    #Sorts the dataframe by greatest to least points scored 
st.markdown(f'<H6 style="color:darkgreen; font-size:25px;">{"Running Backs"}</H6>', unsafe_allow_html=True)        
RB_stats = Com_Stats[Com_Stats['Position'] =='RB']
RB_stats = RB_stats.drop(labels = ['Position','Rank','Team_Name', 'Passes_Completed', 'Passing_Att', 'Passing_Yds','Passing_TDs', 'Ints', '2PM', '2PP','Ftpts', 'DKpt', 'FDpt', 'VBD', 'PosRank', 'OVRank'], axis=1)
RB_stats = RB_stats.set_index('Player')
RB_stats=RB_stats.sort_values('PPR', ascending=False)
RB_stats
#End RB data Frame
#Start Select RB
RB_Selection = st.multiselect('Select Starting RBs (2-3)', RB_stats.index)
RB_Modifer = st.multiselect('Select RB stat modifiers', RB_stats.columns)
'\n'
#End Select RB
#Start WR data frame
    #Creates a data frame for WRs only 
    #Drops columns that are not needed 
    #Sets the index to player Name 
    #Sorts the dataframe by greatest to least points scored 
st.markdown(f'<H6 style="color:darkgreen; font-size:25px;">{"Wide Recievers"}</H6>', unsafe_allow_html=True)        
WR_stats= Com_Stats[Com_Stats['Position'] =='WR']
WR_stats = WR_stats.drop(labels = ['Position','Rank','Team_Name', 'Passes_Completed', 'Passing_Att', 'Passing_Yds','Passing_TDs', 'Ints','2PM', '2PP','Ftpts', 'DKpt', 'FDpt', 'VBD', 'PosRank', 'OVRank'], axis=1)
WR_stats = WR_stats.set_index('Player')
WR_stats=WR_stats.sort_values('PPR', ascending=False)
WR_stats
#End WR data Frame
#Start Select WR
WR_Selection = st.multiselect('Select Starting WRs (2-3)', WR_stats.index)
WR_Modifer = st.multiselect('Select WR stat modifiers', WR_stats.columns)
'\n'
#End Select WR
#Start TE data frame
    #Creates a data frame for TEs only 
    #Drops columns that are not needed 
    #Sets the index to player Name 
    #Sorts the dataframe by greatest to least points scored 
st.markdown(f'<H6 style="color:darkgreen; font-size:25px;">{"Tight Ends"}</H6>', unsafe_allow_html=True)        
TE_stats= Com_Stats[Com_Stats['Position'] =='TE']
TE_stats = TE_stats.drop(labels = ['Position','Rank','Team_Name', 'Passes_Completed', 'Passing_Att', 'Passing_Yds','Passing_TDs', 'Ints','2PM', '2PP','Ftpts', 'DKpt', 'FDpt', 'VBD', 'PosRank', 'OVRank'], axis=1)
TE_stats = TE_stats.set_index('Player')
TE_stats=TE_stats.sort_values('PPR', ascending=False)
TE_stats
#End TE data Frame
#Start Select TE
TE_Selection = st.selectbox('Select a Starting TE (1)', TE_stats.index)
TE_Modifer = st.multiselect('Select TE stat modifiers', TE_stats.columns)
#End Select TE
#Start QB model
QB_X_test = pd.DataFrame(QB_stats[QB_Modifer].loc[QB_Selection]) 
QB_X_test = QB_X_test.transpose()
QB_Y_test = pd.DataFrame(QB_stats[['PPR']].loc[QB_Selection])
QB_Y_test = QB_Y_test.transpose()
QB_X_train = pd.DataFrame(QB_stats[QB_Modifer].drop(QB_Selection))
QB_Y_train = pd.DataFrame(QB_stats['PPR'].drop([QB_Selection]))
QB_num_attributes=QB_Modifer
col_transform = ColumnTransformer(transformers=[('num',MinMaxScaler(),QB_num_attributes)],remainder='passthrough')
pipeline=Pipeline([('trans',col_transform),('mlr_model', LinearRegression())])
pipeline.fit(QB_X_train,(QB_Y_train))
QB_y_pred = pipeline.predict(QB_X_test)
QB_Results = pd.DataFrame(QB_Y_test)
QB_Results ['Projected_PTs'] = QB_y_pred
#End QB model
#Start RB model
RB_X_test = pd.DataFrame(RB_stats[RB_Modifer].loc[RB_Selection]) 
RB_Y_test = pd.DataFrame(RB_stats[['PPR']].loc[RB_Selection])
RB_X_train = pd.DataFrame(RB_stats[RB_Modifer].drop(RB_Selection))
RB_Y_train = pd.DataFrame(RB_stats['PPR'].drop(RB_Selection))
RB_num_attributes=RB_Modifer
col_transform = ColumnTransformer(transformers=[('num',MinMaxScaler(),RB_num_attributes)],remainder='passthrough')
pipeline=Pipeline([('trans',col_transform),('mlr_model', LinearRegression())])
pipeline.fit(RB_X_train,(RB_Y_train))
RB_y_pred = pipeline.predict(RB_X_test)
RB_Results = pd.DataFrame(RB_Y_test)
RB_Results ['Projected_PTs'] = RB_y_pred
#End RB model
#Start WR model
WR_X_test = pd.DataFrame(WR_stats[WR_Modifer].loc[WR_Selection]) 
WR_Y_test = pd.DataFrame(WR_stats[['PPR']].loc[WR_Selection])
WR_X_train = pd.DataFrame(WR_stats[WR_Modifer].drop(WR_Selection))
WR_Y_train = pd.DataFrame(WR_stats['PPR'].drop(WR_Selection))
WR_num_attributes=WR_Modifer
col_transform = ColumnTransformer(transformers=[('num',MinMaxScaler(),WR_num_attributes)],remainder='passthrough')
pipeline=Pipeline([('trans',col_transform),('mlr_model', LinearRegression())])
pipeline.fit(WR_X_train,(WR_Y_train))
WR_y_pred = pipeline.predict(WR_X_test)
WR_Results = pd.DataFrame(WR_Y_test)
WR_Results ['Projected_PTs'] = WR_y_pred
#End WR model
#Start TE Model
TE_X_test = pd.DataFrame(TE_stats[TE_Modifer].loc[TE_Selection]) 
TE_X_test = TE_X_test.transpose()
TE_Y_test = pd.DataFrame(TE_stats[['PPR']].loc[TE_Selection])
TE_Y_test = TE_Y_test.transpose()
TE_X_train = pd.DataFrame(TE_stats[TE_Modifer].drop(TE_Selection))
TE_Y_train = pd.DataFrame(TE_stats['PPR'].drop([TE_Selection]))
TE_num_attributes=TE_Modifer
col_transform = ColumnTransformer(transformers=[('num',MinMaxScaler(),TE_num_attributes)],remainder='passthrough')
pipeline=Pipeline([('trans',col_transform),('mlr_model', LinearRegression())])
pipeline.fit(TE_X_train,(TE_Y_train))
TE_y_pred = pipeline.predict(TE_X_test)
TE_Results = pd.DataFrame(TE_Y_test)
TE_Results ['Projected_PTs'] = TE_y_pred
#End TE model
#Start display resluts  
QB_RB=pd.concat([QB_Results,RB_Results])
QB_RB_WR=pd.concat([QB_RB,WR_Results])
QB_RB_WR_TE=pd.concat([QB_RB_WR,TE_Results])
QB_RB_WR_TE
#End display results