#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from bs4 import BeautifulSoup as bs4
from datetime import datetime, date, time
import math
from matplotlib.pyplot import cm
import matplotlib.pyplot as plyplot
import os
import glob
plyplot.rcParams['figure.figsize']=[100.0,100.0]
from pylab import plt
plt.style.use('seaborn')
import seaborn as sns
import cvxopt as opt
from cvxopt import blas, solvers
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def import_html_bt(archivo):
    """ Import an htm file. First we read it """
    # Open and read file
    raw = open(archivo, 'r').read()
    # Pass the file to BeautifulSoup to be parsed
    soup = bs4(raw, 'html.parser')
    # Extract all tables from file
    tablas = soup.find_all('table')
    # From the first table extract all td tags
    tags = tablas[0].find_all('td')
            
    # Now, we transverse the first table and extract the headers
    titulo = soup.head.find('title').string.split(':')[-1].strip()
    index = 0
    cabecera = {}
    cabecera['EA'] = titulo
    while index < 28:
        if tags[index].string is None:
            index += 1
        else:
            nombre = tags[index].string
            index += 1
            valor = tags[index].string
            index += 1
            cabecera[nombre] = valor
            
    # Extract the header for the data itself (the order history)
    tags = tablas[1].find_all('td')
    header_temp = []
    num_fields = 10
    for elem in range(0, num_fields):
        header_temp.append(tags[elem].string.replace(' ', ''))
                
    # Parse the data from mt4 backtest report
    operacion = []
    operaciones = []
    elem = 0
    comienzo = len(header_temp)
    for idx in range(comienzo, len(tags)):
        temp = tags[idx].string
        if temp is not None:
            operacion.append(temp)
            elem += 1
        if (temp is None) or (elem > 9):
            operaciones.append(operacion)
            operacion = []
            elem = 0
            
    backtest_report = pd.DataFrame(data=operaciones, columns=header_temp)
    # Convert data in a dictionarly
    data = {'header': cabecera, 'report': backtest_report}
    
    # Prepare data for export in an appropiate format
    data['report']['Tiempo'] = pd.to_datetime(data['report']['Tiempo'])
    data['report']['#'] = data['report']['#'].astype('int16')
    data['report']['Tipo'] = data['report']['Tipo'].astype('str')
    data['report']['Orden'] = data['report']['Orden'].astype('int16')
    data['report']['Volumen'] = data['report']['Volumen'].astype('float')
    data['report']['Precio'] = data['report']['Precio'].astype('float')
    data['report']['S/L'] = data['report']['S/L'].astype('float')
    data['report']['T/P'] = data['report']['T/P'].astype('float')
    data['report']['Beneficios'] = data['report']['Beneficios'].astype('float')
    data['report']['Balance'] = data['report']['Balance'].astype('float')
    data['report'].set_index(data['report']['Tiempo'], inplace=True)
    
    return data


# In[ ]:


def import_csv_bt(archivo, separador):
    file = open(archivo, 'r')
    raw_data = file.read().split('\n')
    header = raw_data[0].split(separador)
    raw_data = raw_data[1:]
    data = []
    for row in raw_data:
        data.append(row.split(separador))  
        
    data = pd.DataFrame(data=data, columns=header)
    # Drop NaN values so the dates extraction can be cone
    data = data.dropna()
    # As 'Date' field is text we have to convert it into date format '%d/%m/%Y'
    fechas = [datetime.strptime(x, '%d/%m/%Y') for x in data['Date']]
    data['Date'] = fechas
    data.set_index(['Date'], inplace = True)
    data.sort_index(ascending=True, inplace=True)
    # Convert to numeric. This must be done after Date conversion
    data = data.astype(float)
    return data


# In[ ]:


def calculate_daily_ret(prices, method='lineal'):
    if method == 'lineal':
        daily_ret = ( prices - prices.shift(1) ) / prices.shift(1)
    else:
        if method == 'log':
            daily_ret = np.log(prices / prices.shift(1))
            
    return daily_ret


# In[ ]:


def calculate_cumm_daily_ret(daily_ret):
    return daily_ret.add(1).cumprod() - 1


# In[ ]:


def calculate_sharpe_ohlc(data, rf=0.04):
    periods = 252
    adj_close = data['Adj Close']
    data['DailyRet'] = calculate_daily_ret(adj_close)
    #data['DailyRet'].dropna(inplace=True)
    data['ExcessDailyRet'] = data['DailyRet'] - rf / periods
    sharpe = math.sqrt(periods) * data['ExcessDailyRet'].mean() / data['ExcessDailyRet'].std()
    return sharpe


# In[ ]:


def calculate_high_watermark(series):
    high_watermark = []
    high_watermark.append(series[0])
    for i in range(1,len(series)):
        high_watermark.append(max(series[i], high_watermark[i-1]))
    
    high_watermark = pd.Series(high_watermark, index=series.index)
    return high_watermark


# In[ ]:


def calculate_dd_from_close(close):
    # First, we calculate the cummulative compounded returns
    daily_returns = calculate_daily_ret(close)
    cumm_returns = calculate_cumm_daily_ret(daily_returns)
    high_watermark = calculate_high_watermark(cumm_returns)    
    drawdown = (high_watermark.add(1).divide(cumm_returns.add(1))-1).dropna()
    return drawdown


# In[ ]:


def calculate_dd_from_equity(equity):
    high_watermark = calculate_high_watermark(equity)
    return high_watermark - equity


# In[ ]:


def calculate_dd_duration(drawdown, holding_time=1):
    duration = drawdown
    drawdown.dropna(inplace=True)
    for i in range(0,len(drawdown)):
        if drawdown[i] == 0:
            duration[i] = 0
        else:
            drawdown[i] = duration[i-1] + 1
    
    return duration * holding_time


# In[ ]:


def calculate_profit_factor(retornos):
    positive = np.where(retornos >= 0, 1, 0)
    negative = np.where(retornos < 0, -1, 0)
    return (retornos * positive).sum() / (retornos * negative).sum()


# In[ ]:


def calculate_expected_payoff(retornos):
    total = len(retornos)
    gross_profit = (retornos * np.where(retornos >= 0, 1, 0)).sum()
    gross_loss   = (retornos * np.where(retornos < 0, -1, 0)).sum()
    return (gross_profit - gross_loss) / total


# In[ ]:


def calculate_sortino(retornos, excessRet):
    excessReturns = retornos - excessRet
    negativeExcessReturns = excessReturns * np.where(excessReturns < excessRet, 1, 0)
    excessRetNeg = math.sqrt(pd.Series([math.pow(x, 2) for x in negativeExcessReturns]).sum() / len(negativeExcessReturns))
    sortino = excessReturns.mean() / excessRetNeg
    return sortino


# In[ ]:


def calculate_alvort_coeff(retornos):
    maxDD, _ = calculate_drawdown(retornos)
    maxDD = max(maxDD)    


# In[ ]:


def calculate_total_duration(retornos):
    return 0


# In[ ]:


def parse_mt4_backtest(datos, capital_inicial=10000):
    """ Una vez están importados los datos, los convertimos a un DataFrame de Pandas cuyas columnas 
    siguen el siguiente formato: 
    [Numero Operacion, Fecha Apertura, Fecha Cierre, Precio Apertura, Precio Cierre, 
     StopLoss Inicial, TakeProfit Inicial, B/P] """
    aperturas        = np.where((datos['report']['Tipo'] == 'buy') | (datos['report']['Tipo'] == 'sell'))
    aperturas        = pd.Series(aperturas[0])
    cierres          = np.where((datos['report']['Tipo'] == 't/p') | (datos['report']['Tipo'] == 's/l') | 
                     (datos['report']['Tipo'] == 'close') | (datos['report']['Tipo'] == 'close at stop'))
    cierres          = pd.Series(cierres[0])
    modificaciones   = np.where(datos['report']['Tipo'] == 'modify')
    modificaciones   = pd.Series(modificaciones[0])
    fecha_aperturas  = np.array(datos['report']['Tiempo'][aperturas])
    fecha_cierres    = np.array(datos['report']['Tiempo'][cierres])
    precio_aperturas = np.array(datos['report']['Precio'][aperturas])
    precio_cierres   = np.array(datos['report']['Precio'][cierres])
    stop_loss        = np.array(datos['report']['S/L'][aperturas])
    take_profit      = np.array(datos['report']['T/P'][aperturas])
    beneficios       = np.array(datos['report']['Beneficios'][cierres])
    equity           = np.array(beneficios.cumsum() + capital_inicial)
    num_operations   = np.arange(0,len(equity))
    
    # Correction of initial SL/TP in case they are not sent along with the order
    if ((stop_loss == 0.0).all() | (take_profit == 0.0).all()):
        stop_loss       = np.array(datos['report']['S/L'][aperturas + 1])
        take_profit     = np.array(datos['report']['T/P'][aperturas + 1])
    
    backtest_skimmed = [fecha_aperturas, fecha_cierres, precio_aperturas, precio_cierres, stop_loss, 
             take_profit, beneficios, equity]
    columnas  = ['DateOpen', 'DateClose', 'PriceOpen', 'PriceClose', 'S/L', 'T/P', 'P&L', 'Equity']
    
    bt = {}
    for i in range(0,len(columnas)):
        bt[columnas[i]] = backtest_skimmed[i]
        
    bt = pd.DataFrame(bt, columns=columnas)
    bt.index = num_operations
        
    return bt


# In[ ]:


def align_backtests(backtests, column, index):
    """
    Aligns equities curves into the same pandas DataFrame based on index
    
    Parameters
    ----------
    backtests : Dictionary
        Contains the results from backtests performed
    column : str
        Name of the column to merge
    index : str
        Name of the index for the resulting DataFrame

    Returns
    -------
    DataFrame
        
        Merged columns
    """
    
    equities  = pd.DataFrame()
    columnas = []
    for key, value in backtests.items():
        bt       = pd.DataFrame(backtests[key][column])
        bt.index = backtests[key][index]
        equities = pd.merge(equities, bt, right_index=True, left_index=True, how='outer')
        columnas.append(key)

    equities.columns = columnas
    
    return equities


# In[ ]:


def get_filenames(prefix, directorio, extension):
    """
    Reads all the filenames in a directory and assign a unique key 
    composed of prefix + number to each of them
    
    Parameters
    ----------
    prefix     : str
        Prefix of key
    directorio : str
        Name of the directory from where too read the files
    extension : str
        Name of the extension to look for inside directorio
        
    Returns
    -------
    Dictionary
        Names of files with a unique key assigned
    """
    
    os.chdir(directorio)
    files = glob.glob(extension)

    archivos = {}
    n = 1
    for file in files:
        nombre = prefix + f'{n:04d}'
        archivos[nombre] = file
        n += 1
    
    return archivos


# In[ ]:


def parse_several_backtests(archivos, capital_inicial = 10000):
    """
    Reads all the files from Dictionary. Then it imports and parses 
    all of them into a new Dictionary
    
    Parameters
    ----------
    archivos : dict
        Key, values corresponding to files to be parsed
    capital_inicial: float64
        Initial amount of money for the backtest

    Returns
    -------
    Dictionary
        backtests parsed ready to be processed
        
    TODO
    ----
    Convert capital_inicial in a list and modify code for this
    """
    backtests = {}
    for key, value in archivos.items():
        datos          = import_html_bt(archivos[key])
        backtests[key] = parse_mt4_backtest(datos, capital_inicial)
        
    return backtests


# In[ ]:


def plot_equity_curves(equities, width, height, xlabel, ylabel, title):
    """
    Plots the equity curves from a equities DataFrame
    
    Parameters
    ----------
    equities : DataFrame
        Stored values corresponding to parsed equity curves

    Returns
    -------
    
    """
    
    colors = iter(cm.rainbow(np.linspace(0,1,len(equities.columns))))
    plyplot.figure(figsize=(width, height))
    for equity in equities:
        color = next(colors)
        plyplot.plot(equities[equity], c=color, label=equity)
    
    plyplot.xlabel(xlabel)
    plyplot.ylabel(ylabel)
    plyplot.title(title)
    plyplot.legend()


# In[ ]:


def rand_weights(n):
    """
    Produces n random weights that add up to 1
    
    Parameters
    ----------
    n : integer
        Number of weights to be generated

    Returns
    -------
       Random weights normalized to 1
    """
    k = np.random.rand(n)
    return k / sum(k)


# In[ ]:


def calculate_backtests_returns(equities):
    """
    Calculates the returns for a portfolio
    
    Parameters
    ----------
    equities : DataFrame
        Stored values corresponding to returns

    Returns
    -------
    mu : float64
        Return of portfolio
    sigma : float64
        Standard Deviation of the portfolio
    """
    
    p = np.asmatrix(equities.mean(axis=0))
    w = np.asmatrix(rand_weights(equities.shape[1]))
    C = np.asmatrix(equities.cov())
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    # This recursion reduces outliers to keep plots pretty
    #if sigma > 2:
    #    return calculate_backtests_returns(returns)
    
    return mu, sigma


# In[ ]:


def optimal_portfolio(equities):
    """
    Finds the optimal portfolio for a series of different backtests
    
    Parameters
    ----------
    equities : DataFrame
        Returns from backtests (equity curves)

    Returns
    -------
       weights : array of floats
       returns : array with expected returns from each equity
       risks   : array with the risks associated to the
    """
    n = equities.shape[1]
    #equities = np.asmatrix(equities)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.asmatrix(equities.cov()))
    pbar = opt.matrix(np.asmatrix(equities.mean(axis=0)))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']
    return np.asarray(wt), returns, risks


# In[ ]:


def portfolio_mc(backtests, num_portfolios=50000):
    """
    Computes the efficient frontier for a portfolio of returns through
    MonteCarlo simulation
    
    Parameters
    ----------
    backtests : DataFrame
        Results from trades
    
    num_portfolios : int
        Number of portfolios to be simulated
    
    Returns
    -------
    DataFrame
        Simulated Portfolios
    DataFrame    
        Portfolios with minimal drawdown and maximum Sharpe Ratio
    """
    
    # First, resample trades to be done on annual basis and calculate the mean value
    returns        = backtests.resample('A').sum()
    returns_annual = returns.mean()
    
    # Calculate the covariance between backtests
    cov_annual = returns.cov()

    port_returns = []
    port_volatility = []
    stock_weights = []
    sharpe_ratio = []
    
    selected = backtests.columns
    num_assets = len(selected)
    
    np.random.seed(89)
    
    # Perform the simulations
    for single_portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        returns = np.dot(weights, returns_annual)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
        sharpe = returns / volatility
        sharpe_ratio.append(sharpe)
        port_returns.append(returns)
        port_volatility.append(volatility)
        stock_weights.append(weights)

    # Define a dictionary with three metrics
    portfolio = {'Returns': port_returns, 'Volatility': port_volatility,'Sharpe Ratio': sharpe_ratio}
    
    # Add more columns depending on the name of columns in the backtest
    for counter,symbol in enumerate(selected):
        portfolio[symbol + ' Weight'] = [Weight[counter] for Weight in stock_weights]
        
    # Convert dictionary to DataFrame, ordering columns
    column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in selected]
    portfolio = pd.DataFrame(portfolio, columns=column_order)
    
    # Select the two special portfolios
    min_volatility = portfolio['Volatility'].min()
    max_sharpe = portfolio['Sharpe Ratio'].max()
    # use the min, max values to locate and create the two special portfolios
    sharpe_portfolio = pd.DataFrame(portfolio.loc[portfolio['Sharpe Ratio'] == max_sharpe])
    min_variance_port = pd.DataFrame(portfolio.loc[portfolio['Volatility'] == min_volatility])
    
    #optimum_portfolios = sharpe_portfolio.T.join(min_variance_port.T, lsuffix='Sharpe', rsuffix='Variance').T
    optimum_portfolios = [sharpe_portfolio.T, min_variance_port.T]
    
    return portfolio, optimum_portfolios


# In[ ]:


def draw_portfolios(portfolios, optimum_portfolios):
    """
    Draws portfolios. Assumes 'Volatility', 'Returns' and
    'Sharpe Ratio' are columns of portfolios
    
    Parameters
    ----------
    portfolios : DataFrame
        portfolios to draw
    
    Returns
    -------
    
    """
    
    plt.style.use('seaborn-dark')
    axis = portfolios.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                    cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)    
    colors = iter(cm.rainbow(np.linspace(0,1,len(optimum_portfolios))))
    hdl = []
    num = 1
    # Draw optimum_portfolios
    for pf in optimum_portfolios:
        color = next(colors)
        pf_aux = pf.T
        hdl.append(pf_aux.plot.scatter(x='Volatility', y='Returns', marker='D', s=200, ax=axis,
                                       label='Portfolio' + str(num) ))
        num+=1
        
    plt.xlabel('Volatility (Std. Deviation)')
    plt.ylabel('Expected Returns')
    plt.title('Efficient Frontier')
    axis.legend(handles=hdl)
    plt.show()

