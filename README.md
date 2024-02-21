# Forecasting Thailand Electricity Consumption by ARIMA model and STL

This project aims to study electricity consumption in Thailand (monthly) in the past 22 years during 2002 to 2023. First I have downloaded the data from the Energy Policy and Planning office website to a csv file.

```
import pandas as pd

data = pd.read_csv('/Users/lune.j/Desktop/project/Thailand_electricity_consumption.csv');data
```
Dataset:
```
 Month	          Elec_consump
2002-01-01		7327
2002-02-01		7359
2002-03-01		8471
2002-04-01		8461
2002-05-01		8730
...	...	...
2023-08-01		17863
2023-09-01		17125
2023-10-01		17180
2023-11-01		16661
2023-12-01		16149
```


## Seasonal and Trend decomposition using Loess (STL)
Seasonal and Trend decomposition using Loess (STL) is a versatile and robust method for decomposing time series, while Loess is a method for estimating nonlinear relationships.

Applying the STL decomposition to the dataset:

```
import statsmodels as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition=seasonal_decompose(data['Elec_consump'],model='additive',period=12)
decomposition.plot()
plt.show()
```

Figure 1 : The electricity consumption (top) and its three additive components obtained from a STL decomposition with positive trend and fixed seasonality.

![decomposition](https://github.com/jsutthida/Forecasting-Thailand-Electricity-Consumption-by-STL/assets/160230541/c909c738-e5bc-4d15-b98f-3307ca69b2ed)

There is obviously that the dataset has positive trend and fixed seasonality.


### Robust fitting
Using robust estimation allows the model to tolerate larger errors that are visible on the bottom plot by re-weights data when estimating the LOESS. 
Next, estimated the model with and without robust weighting. The difference is minor and is most pronounced during the political crisis of 2009. The non-robust estimate places equal weights on all observations and so produces smaller errors, on average. The weights vary between 0 and 1.


```
def add_stl_plot(fig, res, legend):
    """Add 3 plots from a second STL fit"""
    axs = fig.get_axes()
    comps = ["trend", "seasonal", "resid"]
    for ax, comp in zip(axs[1:], comps):
        series = getattr(res, comp)
        if comp == "resid":
            ax.plot(series, marker="o", linestyle="none")
        else:
            ax.plot(series)
            if comp == "trend":
                ax.legend(legend, frameon=False)

stl = STL(data['Elec_consump'], period=12, robust=True)
res_robust = stl.fit()
fig = res_robust.plot()
res_non_robust = STL(data['Elec_consump'], period=12, robust=False).fit()
add_stl_plot(fig, res_non_robust, ["Robust", "Non-robust"])

fig = plt.figure(figsize=(16, 5))
lines = plt.plot(res_robust.weights, marker="o", linestyle="none")
ax = plt.gca()
xlim = ax.set_xlim(data['Elec_consump'].index[0], data['Elec_consump'].index[-1])
```

Figure 2 : The electricity consumption and its three additive components obtained from a robust STL decomposition
![Robust fitting](https://github.com/jsutthida/Forecasting-Thailand-Electricity-Consumption-by-STL/assets/160230541/7af89256-a7ea-4371-9a64-700ee0476179)

Figure 3 : Robust weights data
![Robust weight](https://github.com/jsutthida/Forecasting-Thailand-Electricity-Consumption-by-STL/assets/160230541/856a7ab6-bc00-4982-822f-75ddd51177e5)


### LOESS degree
The default configuration estimates the LOESS model with both a constant and a trend. The degree makes little difference except in the trend around the politic crisis of 2009 and nearly present.

```
stl = STL(
    data['Elec_consump'], period=12, seasonal_deg=0, trend_deg=0, low_pass_deg=0, robust=True
)
res_deg_0 = stl.fit()
fig = res_robust.plot()
add_stl_plot(fig, res_deg_0, ["Degree 1", "Degree 0"])
```

Figure 4 : The electricity consumption and its three additive components obtained from a LOESS degree STL decomposition
![LOESS degree](https://github.com/jsutthida/Forecasting-Thailand-Electricity-Consumption-by-STL/assets/160230541/57b47eed-83fa-4a00-9ac2-e0f6b991ab97)


### Forecasting
Using STLForecast function to remove seasonalities and then using a standard time-series model to forecast the trend and cyclical components.
Here we use STL to handle the seasonality and then an ARIMA(1,1,0) to model the deseasonalized data. 

```
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast

data['Elec_consump'].index.freq = data['Elec_consump'].index.inferred_freq
stlf = STLForecast(data['Elec_consump'], ARIMA, model_kwargs=dict(order=(1, 1, 0), trend="t"))
stlf_res = stlf.fit()

forecast = stlf_res.forecast(24)
plt.plot(data['Elec_consump'])
plt.plot(forecast)
plt.show()
```

Figure 5 : Forecasting of Thailand monthly electricity consumption by ARIMA model with STL.
![forecasting](https://github.com/jsutthida/Forecasting-Thailand-Electricity-Consumption-by-STL/assets/160230541/793440c3-2f8a-49c4-9b91-9d9bd2f75761)

Here, the forecasts data of Thailand monthly electricity consumption for the next 24 months.
```
2024-01-01    15791.148488
2024-02-01    15683.114398
2024-03-01    18171.121124
2024-04-01    18355.989612
2024-05-01    19404.618821
2024-06-01    18642.170283
2024-07-01    18620.275449
2024-08-01    18524.220846
2024-09-01    17750.037133
2024-10-01    17590.509849
2024-11-01    17273.640695
2024-12-01    16568.333234
2025-01-01    16226.109436
2025-02-01    16116.811941
2025-03-01    18604.920806
2025-04-01    18789.781037
2025-05-01    19838.410913
2025-06-01    19075.962321
2025-07-01    19054.067491
2025-08-01    18958.012888
2025-09-01    18183.829175
2025-10-01    18024.301891
2025-11-01    17707.432738
2025-12-01    17002.125277
Freq: MS, dtype: float64
```

#### Preference
Seasonal-Trend decomposition using LOESS (STL) : https://www.statsmodels.org/dev/examples/notebooks/generated/stl_decomposition.html
Forecasting with decomposition : https://otexts.com/fpp2/forecasting-decomposition.html
