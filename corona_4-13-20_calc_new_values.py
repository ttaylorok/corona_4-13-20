import pandas as pd
import shapefile
import xlrd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from math import sqrt
import datetime

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

#https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide
df = pd.read_excel("COVID-19-geographic-disbtribution-worldwide-2020-04-13.xlsx")
df.set_index(keys = ['countryterritoryCode'], inplace = True)

def fit_exp(x, a, b, c):
    return a * np.power(2,x/b) + c

def fit_lin(x, a, b):
    return a * x + b

ar_fits = []

#df.dropna(inplace = True)
df = df[df.index.astype(str) != 'nan']
for i in df.index.unique():
    ds1 = df.loc[i,:].dropna().sort_values('dateRep')
    ds1['cases_cs'] = ds1['cases'].cumsum()
    ln = len(ds1.index)
    ds2 = ds1.iloc[ln-10:ln,:]
    days = np.arange(10)
    minval = ds2['cases_cs'].min()
    print(ds2)
    try:
        popt, pcov = curve_fit(fit_exp, days, ds2['cases_cs']-minval)
        rms = sqrt(mean_squared_error(ds2['cases_cs']-minval, fit_exp(days,*popt)))
    except:
        popt = [0,0,0]
        rms = 9999999
    print(popt)
    try:
        popt2, pcov2 = curve_fit(fit_lin, days, ds2['cases_cs']-minval)
        rms2 = sqrt(mean_squared_error(ds2['cases_cs']-minval, fit_lin(days,*popt2)))
    except:
        popt2 = [0,0]
        rms2 = 9999999

    arr = [i,popt[1],rms,popt2[0],rms2]
    print(arr)
    ar_fits.append(arr)
    
dff = pd.DataFrame(ar_fits, columns = ['country', 'exp_growth', 'exp_error', 'lin_growth', 'lin_error'])

dff['gtype'] = np.where(dff['exp_error'] < dff['lin_error'], 'exp', 'lin')

dff.to_csv("growth_rates_4-13-20.csv")



#for col in df3.columns:
#     print(col)
    # if type(df3[col][0]) == str:
    #     w.field(col, 'C')
    # elif "perc" in col:
    #     w.field(col, 'F', decimal=4)
    # elif col == 'PCA1':
    #     w.field(col, 'F', decimal=4)
    # else:
    #     w.field(col,'N')

#df['Date'] = pd.to_datetime(df[["Year","Month","Day"]])

#trans = df.groupby(['Countries and territories'])['DateRep'].transform(max)
#idx = trans == df['DateRep']
#df_today = df[idx]

#df.plot(kind="line")
#df_plot = df[['DateRep','Cases','Countries and territories']]

df_sort = df.sort_values(by='DateRep')
df['cum_sum']=df_sort.groupby('Countries and territories')['Cases'].cumsum()
df_country = df_sort.groupby('Countries and territories')['Cases','Deaths'].sum()
df_country.sort_values(by="Cases", inplace=True, ascending=False)
df_country['Death Rate'] = df_country['Deaths']/df_country['Cases']*100

df_plot = df.pivot(index='DateRep', columns='Countries and territories', values='cum_sum')
df_plot_sort = df_plot.sort_values(by='DateRep', ascending=False)

fig, ax = plt.subplots(2,1,figsize=(9,8))
fig.suptitle("Cumulative Covid-19 Cases by Date and Country")

#plt.subplot(121)
#plt.xlim(datetime.datetime(2020,1,15),datetime.datetime(2020, 3, 15))
ax[0].set_xlim([datetime.datetime(2020,3,1),datetime.datetime(2020, 3, 24)])
ax[0].set_ylim([0,70000])
ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
top10=["United_States_of_America","China","Italy","Spain","Germany","Iran","France","South_Korea"]
for col in df_plot.columns:
    if col in top10:
        ax[0].plot(df_plot.index,df_plot[col],label=col,linewidth=2)
    else:
        ax[0].plot(df_plot.index,df_plot[col],c="gray", alpha=0.4)


ax[1].set_xlim([datetime.datetime(2020,3,1),datetime.datetime(2020, 3, 24)])
ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
#plt.subplot(122)
for col in df_plot.columns:
    if col in top10:
        ax[1].plot(df_plot.index,df_plot[col],label=col, linewidth=2)
    else:
        ax[1].plot(df_plot.index,df_plot[col],c="gray", alpha=0.4)
ax[1].set_ylim([50,70000])
plt.legend(bbox_to_anchor=(1, 1.6))
plt.subplots_adjust(right=0.7)
plt.subplots_adjust(top=0.93)
plt.yscale("log")

fig.savefig("country_charts") 
plt.show()



#df_plot_sort['day'] = np.arange(len(df_plot_sort),0,-1)
df_sample = df_plot_sort.iloc[1:10,:]
df_plot_sort2 = df_sample.sort_values(by="DateRep")
df_plot_sort2['day'] = np.arange(len(df_plot_sort2))

def fit_exp(x, a, b, c):
    return a * np.power(2,x/b) + c

def fit_lin(x, a, b):
    return a * x + b

fig2, ax2 = plt.subplots(4,2, figsize=(7,12))

#df_fits = pd.DataFrame(columns = ["country","exp_growth","rms_exp","linear_growth","rms_linear"])
ar_fits = []
for col in df_plot_sort2:
    df_small = df_plot_sort2[['day',col]].dropna()
    min_val = df_small[col].min()
    try:
        popt, pcov = curve_fit(fit_exp, df_small['day'], df_small[col]-min_val)
        rms = sqrt(mean_squared_error(df_small[col]-min_val, fit_exp(df_small['day'],*popt)))
    except:
        popt = [0,0,0]
        rms = 9999999
    try:
        popt2, pcov2 = curve_fit(fit_lin, df_small['day'], df_small[col]-min_val)
        rms2 = sqrt(mean_squared_error(df_small[col]-min_val, fit_lin(df_small['day'],*popt2)))
    except:
        popt2 = [0,0]
        rms2 = 9999999

    arr = [col,popt[1],rms,popt2[0],rms2]
    print(arr)
    ar_fits.append(arr)

    maxval = (df_small[col]-min_val).max()
    
    if col == "China":
        plt.subplot(421)
        plt.plot(df_small['day'], fit_exp(df_small['day'],*popt),'r-')
        plt.scatter(df_small['day'], df_small[col]-min_val, c='blue')
        plt.text(0, maxval*(.7), 'exponential\nb='+str(round(popt[1],2)), fontsize=8, color='red')
        plt.xlabel("China")
    
    if col == "Italy":
        plt.subplot(422)
        plt.plot(df_small['day'], fit_lin(df_small['day'],*popt2),'r-')
        plt.scatter(df_small['day'], df_small[col]-min_val, c='purple')
        plt.text(0, maxval*(.7), 'linear\na='+str(round(popt2[0],2)), fontsize=8, color='red')
        plt.xlabel("Italy")

    if col == "United_States_of_America":
        plt.subplot(423)
        plt.plot(df_small['day'], fit_exp(df_small['day'],*popt),'r-')
        plt.scatter(df_small['day'], df_small[col]-min_val, c='gray')
        plt.text(0, maxval*(.7), 'exponential\nb='+str(round(popt[1],2)), fontsize=8, color='red')
        plt.xlabel("USA")

    if col == "Spain":
        plt.subplot(424)
        plt.plot(df_small['day'], fit_lin(df_small['day'],*popt2),'r-')
        plt.scatter(df_small['day'], df_small[col]-min_val, c='pink')
        plt.text(0, maxval*(.7), 'linear\na='+str(round(popt2[0],2)), fontsize=8, color='red')
        plt.xlabel("Spain")
        

    if col == "Germany":
        plt.subplot(425)
        plt.plot(df_small['day'], fit_exp(df_small['day'],*popt),'r-')
        plt.scatter(df_small['day'], df_small[col]-min_val,c='green')
        plt.text(0, maxval*(.7), 'exponential\nb='+str(round(popt[1],2)), fontsize=8, color='red')
        plt.xlabel("Germany")

    if col == "Iran":
        plt.subplot(426)
        plt.plot(df_small['day'], fit_lin(df_small['day'],*popt2),'r-')
        plt.scatter(df_small['day'], df_small[col]-min_val, c='red')
        plt.text(0, maxval*(.7), 'linear\na='+str(round(popt2[0],2)), fontsize=8, color='red')
        plt.xlabel("Iran")

    if col == "France":
        plt.subplot(427)
        plt.plot(df_small['day'], fit_exp(df_small['day'],*popt),'r-')
        plt.scatter(df_small['day'], df_small[col]-min_val, c='orange')
        plt.text(0, maxval*(.7), 'exponential\nb='+str(round(popt[1],2)), fontsize=8, color='red')
        plt.xlabel("France")

    if col == "South_Korea":
        plt.subplot(428)
        plt.plot(df_small['day'], fit_exp(df_small['day'],*popt),'r-')
        plt.scatter(df_small['day'], df_small[col]-min_val, c='brown')
        plt.text(0, maxval*(.7), 'exponential\nb='+str(round(popt[1],2)), fontsize=8, color='red')
        plt.xlabel("South Korea")

plt.tight_layout()
fig2.savefig("exp_fits")
plt.show()


##    if col == "Dominican_Republic":
##        plt.plot(df_small['day'], fit_exp(df_small['day'],*popt),'r-')
##        plt.scatter(df_small['day'], df_small[col]-min_val)
##        plt.show()
df_fits = pd.DataFrame(data=ar_fits,columns = ["country","exp_growth","rms_exp","linear_growth","rms_linear"])

df_fits['growth_type'] = df_fits[["rms_exp","rms_linear"]].idxmin(axis=1)

df_merge = pd.merge(df_country,df_fits,left_index=True,right_on="country")

df_merge.to_excel("table_out.xlsx")




    



#df_plot = df.set_index(keys=["DateRep","Countries and territories"])
#df_plot_u = df_plot.unstack()

#plt.plot(df['DateRep'],df['Cases'], label=df.index.tolist())
#plt.show()


df_today = df.groupby(['Countries and territories'])['Cases'].sum()

#df_today.set_index(keys=["Countries and territories"], inplace=True)

df_today_sort = df_today.sort_values(ascending=False)

#df_today_sort["Cases"].plot(kind="bar")

top_20 = df_today_sort.iloc[1:20]

top_20.plot(kind="bar")

#plt.show()

#fout = open("matches.csv",'w')
#matches = []

w.field('Country', 'C')
w.field('Cases', 'N')
w.field('Deaths', 'N')
w.field('Exp_growth', 'F', decimal=2)
w.field('Lin_growth', 'F', decimal=2)
w.field('growth_type', 'C')

df_merge.set_index(keys="country",inplace=True)
names = [["Bahamas", "Bahamas"],
         ["Belarus", "Byelarus"],
         ["Brunei_Darussalam" , "Brunei"],
         ["Cote_dIvoire" , "Ivory Coast"],
         ["Gambia" , "Gambia, The"],
         ["Isle_of_Man" , "Man, Isle of"],
         ["Myanmar" , "Myanmar (Burma)"],
         ["North_Macedonia" , "Macedonia"],
         ["United_Republic_of_Tanzania" , "Tanzania"],
         ["United_States_of_America" , "United States"]]

ind_mod = []
is_found = False
for x in df_merge.index:
    is_found = False
    for y in names:
        if x == y[0]:
            #print(x)
            ind_mod.append(y[1])
            is_found = True
            break
    if is_found == False:
        ind_mod.append(x)

df_merge.index = ind_mod 


#df_merge = df_merge[df_merge['growth_type']=="rms_exp"]


for shaperec in sf.iterShapeRecords():
    print(shaperec.record[1])
    country = shaperec.record[1]
    is_found = 0  
    for x,i in enumerate(df_merge.index):
        if i.lower().replace("_"," ") == country.lower():
            print(i)
            is_found = 1
            w.record(i,df_merge.loc[i,"Cases"],df_merge.loc[i,"Deaths"],df_merge.loc[i,"exp_growth"],df_merge.loc[i,"linear_growth"],df_merge.loc[i,"growth_type"])
            #matches.append([country, x, i])
            #fout.write(country + ", " + str(x) + ", " + i + '\n')
            break
    if is_found == 0:
        #print("NOT FOUND: ",country)
        #fout.write(country + ",,"+'\n')
        w.record(country,0,0,0,0,'')
    w.shape(shaperec.shape)

##for x,i in enumerate(df_today.index):
##    fout.write(str(x) + ", " + i + '\n')
##
##fout2 = open("matches_2.csv",'w')
##for x,i in enumerate(df_today.index):
##    is_found = False
##    for shaperec in sf.iterShapeRecords():
##        country = shaperec.record[1]
##        if i.lower().replace("_"," ") == country.lower():
##            fout2.write(str(x) + ", " + i + ", " + country + '\n')
##            is_found = True
##            break
##    if not is_found:
##        fout2.write(str(x) + ", " + i + "," + '\n')
##fout2.close()
##    





w.close()
#fout.close()
        


