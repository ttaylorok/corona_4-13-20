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

import plotly.graph_objects as go
import pandas as pd

pd.set_option('display.max_columns', 7)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

#https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide
df = pd.read_excel("COVID-19-geographic-disbtribution-worldwide-2020-04-13.xlsx")

dff = pd.read_csv("growth_rates_4-13-20.csv")
dfe = dff[dff['gtype'] == 'exp']
        
df2 = df.groupby(['countryterritoryCode'])['cases','deaths'].sum()
df2['text'] = 'Deaths: ' + df2['deaths'].astype('str')

fig = go.Figure(data=go.Choropleth(
    locations = dfe['country'],
    z = dfe['exp_growth'],
    text = dfe['country'],
    colorscale= 'Reds',
    #colorbar = {'yanchor' : 'bottom'},
    autocolorscale=False,
    reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    #colorbar_tickprefix = '$',
    colorbar_title = 'Duplication<br>Rate (days)',
    colorbar = {'len' : 0.2, 'x' : 0.05, 'y' : 0.4, 'thickness' : 15, 'bgcolor' : 'rgba(255,255,255,255)', 'borderwidth' : 1, 'tickfont' : {'size' : 9}},
    zmin = 1,
    zmax=6
))

fig.update_layout(
    title_text='Countries with Exponential Growth Rates',
    #colorbar_orientation="h",
    margin={"r":0,"t":30,"l":0,"b":0},
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    ),
    annotations = [dict(
        x=0.55,
        y=0.1,
        xref='paper',
        yref='paper',
        text='Source: <a href="https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide">\
            ECDPC</a>',
        showarrow = False
    )]
)

fig.show()
fig.write_html("world_map_exp_v2.html")