#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

import sympy
from sympy.solvers import solve
import math
import os



def read_file (file):
    """
    Función para leer y organizar el archivo de Excel con los datos de absorbancia que vamos a utilizar
    
    Parámetros:
    file -- archivo Excel
    
    Retorna:
    Una lista con DataFrames de los datos de los diferentes días (hojas de Excel)
    """
    DF_list = list()
    sheets = (0,1,2,3)
    
    # Bucle para que trabaje con todas las páginas
    for sheet in sheets:
        df = pd.read_excel(file,decimal='.',sheet_name=sheet)
        df.set_index('uM',inplace=True)
        
        # Rellena los valores NaN con la media de las réplicas
        df_filled = df.T.fillna(df.mean(axis=1)).T  
        columns = df_filled.shape[1]
        n = int(columns/3)
        
        # Media de R1 en las tres lecturas
        mediaR1 = []
        for i in range(len(df_filled)):
            a = []
            for j in range(0,columns,n):
                a.append(df_filled.iloc[i,j])
            mediaR1.append(np.average(a))       
        # Media de R2 en las tres lecturas
        mediaR2 = []
        for i in range(len(df_filled)):
            a = []
            for j in range(1,columns,n):
                a.append(df_filled.iloc[i,j])
            mediaR2.append(np.average(a))       
        # Media de R3 en las tres lecturas
        mediaR3 = []
        for i in range(len(df_filled)):
            a = []
            for j in range(2,columns,n):
                a.append(df_filled.iloc[i,j])
            mediaR3.append(np.average(a))       
        # Genera nuevo DataFrame con las medias
        df1 = df_filled.iloc[:, :n].copy()
        df1.loc[:, "R1"] = mediaR1
        df1.loc[:, "R2"] = mediaR2
        df1.loc[:, "R3"] = mediaR3
        
        # Si solo tengo 3 columnas de réplicas para aquí
        if n == 4:
            # Media de R4 en las tres lecturas
            mediaR4 = []
            for i in range(len(df_filled)):
                a = []
                for j in range(3, columns, n):
                    a.append(df_filled.iloc[i, j])
                mediaR4.append(np.average(a))
            df1.loc[:, "R4"] = mediaR4
        
        DF_list.append(df1)
    return DF_list



def r_total (df,conc):
    """
    Función para calcular la tasa de crecimiento durante los tres días
    
    Parámetros:
    df -- lista con DataFrames de los datos de los diferentes días (resultado de read_file)
    conc -- para la concentración de fármaco que quieres calcular la tasa de crecimiento
    conc debe ser el valor numérico que indique la posición de la dosis: siendo el control 0 y la dosis más alta 7
    
    Retorna:
    Gráfica del ajuste con la tasa de crecimiento y R2 del ajuste realizado.
    """

    # Se calculan los logaritmos neperianos
    logs = []
    for i in range(0,df[0].shape[0]):
        a = []
        for j in range(0,4):
            array = df[j].iloc[i,:].values
            a.append(np.log(array))
        logs.append(a)
    
    # Se organizan en sublistas y se calcula la tasa de crecimiento para la concentración indicada
    sublists = []
    for sublist in logs:
        sub = []
        for arr in sublist:
            sub.extend(arr.flatten())
        sublists.append(sub)
    
    n = len(df[0].columns)
    tiempo = [i for i in range(4) for _ in range(n)]
    X = np.array(tiempo).reshape(-1, 1)  
    y = np.array(sublists[conc])
    
    # Se realiza el ajuste mediante regresión lineal
    regresion_lineal = LinearRegression()
    regresion_lineal.fit(X, y)
    y_pred = regresion_lineal.predict(X)
    
    # Se obtiene la pendiente de la recta y la R2
    pendiente = regresion_lineal.coef_[0]
    error = mean_squared_error(y, y_pred)

    # Representación gráfica (con las réplicas)
    fig = plt.figure(figsize=(8,5))
    
    lista = sublists[conc]
    for j in range(0,len(lista)-n,n):
        X1 = np.array(tiempo[j:j+(n*2)]).reshape(-1, 1)
        y1 = np.array(lista[j:j+(n*2)])
        plt.scatter(X1, y1)
    
    plt.plot(X,y_pred,'r',label="r = {:.2f}\n$R^2$ = {:.2f}".format(pendiente, regresion_lineal.score(X, y)))
    plt.xlabel('Tiempo (días)',fontsize=15)
    plt.ylabel('ln(N)',fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('Tasa de crecimiento de las células con {} uM de cisplatino'.format(df[0].index[conc]), fontsize=15, pad=18)
    plt.grid(True)
    plt.legend(fontsize=15)
    plt.savefig('tasa de crecimiento.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return (pendiente, error)





def r_tramos (df):
    """
    Función que calcula las tasas de crecimiento (para todas las concentraciones de fármaco)
    Las calcula tramo a tramo entre los diferentes días: r(0-1),r(1-2),r(2-3)
    
    Parámetros:
    df -- lista con DataFrames de los datos de los diferentes días (resultado de read_file)
    
    Retorna:
    3 listas con los valores de las tasas de crecimiento en: 24 horas, 48 horas y 72 horas
    1 lista con los valores de las concentraciones
    """
    
    # Se calculan los logaritmos neperianos
    logs = []
    for i in range(0,df[0].shape[0]):
        a = []
        for j in range(0,4):
            array = df[j].iloc[i,:].values
            a.append(np.log(array))
        logs.append(a)
    
    # Los datos se organizan en sublistas y se calculan las tasas de crecimiento
    sublists = []
    for sublist in logs:
        sub = []
        for arr in sublist:
            sub.extend(arr.flatten())
        sublists.append(sub)
    
    m = []
    n = len(df[0].columns)
    tiempo = [i for i in range(4) for _ in range(n)]
    for i in range(len(sublists)):
        lista = sublists[i]
        for j in range(0,len(lista)-n,n):
            X = np.array(tiempo[j:j+(n*2)]).reshape(-1, 1)
            y = np.array(lista[j:j+(n*2)])
            regresion_lineal = LinearRegression()
            regresion_lineal.fit(X, y)
            y_pred = regresion_lineal.predict(X)
            pendiente = regresion_lineal.coef_[0]
            m.append(pendiente)
            
    # Las tasas de crecimiento se organizan según los diferentes días
    r1 = m[::3]
    r2 = m[1::3] 
    r3 = m[2::3]
    c = df[0].index.tolist()
    return r1,r2,r3,c





def r_tramos_representacion(df):
    """
    Igual que r_tramos pero también representa gráficamente todos los resultados
    Función que calcula y representa las tasas de crecimiento (para todas las concentraciones de fármaco)
    Las calcula tramo a tramo entre los diferentes días: r(0-1),r(1-2),r(2-3)
    
    Parámetros:
    df -- lista con DataFrames de los datos de los diferentes días (resultado de read_file)
    
    Retorna:
    Una lista con los valores de las tasas de crecimiento en orden: tres primeras para el control, 
    tres siguientes para la primera dosis de fármaco y así sucesivamente. 
    Además guardará todas las gráficas en una nueva carpeta.
    """
    # Nombre de la carpeta donde se guardarán las figuras
    carpeta = "tasas de crecimiento a tramos"
    # Crear la carpeta si no existe
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)
    
    # Se calculan los logaritmos neperianos
    logs = []
    for i in range(0,df[0].shape[0]):
        a = []
        for j in range(0,4):
            array = df[j].iloc[i,:].values
            a.append(np.log(array))
        logs.append(a)
    
    # Los datos se organizan en sublistas y se calculan las tasas de crecimiento
    sublists = []
    for sublist in logs:
        sub = []
        for arr in sublist:
            sub.extend(arr.flatten())
        sublists.append(sub)
    
    m = []
    n = len(df[0].columns)
    tiempo = [i for i in range(4) for _ in range(n)]
    for i in range(len(sublists)):
        lista = sublists[i]
        plt.figure(figsize=(8,5))
        for j in range(0,len(lista)-n,n):
            X = np.array(tiempo[j:j+(n*2)]).reshape(-1, 1)
            y = np.array(lista[j:j+(n*2)])
            plt.scatter(X, y)
            regresion_lineal = LinearRegression()
            regresion_lineal.fit(X, y)
            y_pred = regresion_lineal.predict(X)
            pendiente = regresion_lineal.coef_[0]
            plt.plot(X,y_pred,'r',label="r = {:.2f}".format(pendiente))
            m.append(pendiente)
            #print("Tasa de crecimiento:",pendiente)
        
        plt.xlabel('Tiempo (días)',fontsize=15)
        plt.ylabel('ln(N)',fontsize=15)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title('Tasa de crecimiento de las células con {} uM de cisplatino'.format(df[0].index[i]), fontsize=15, pad=18)
        plt.legend(fontsize=15)
        plt.grid(True)
        nombre_archivo = os.path.join(carpeta, f"figura_{df[0].index[i]}.png")
        plt.savefig(nombre_archivo)
        plt.savefig('tasa de crecimiento a tramos.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    plt.show()
    r1 = m[::3]
    r2 = m[1::3] 
    r3 = m[2::3]
    c = df[0].index.tolist()
    return r1,r2,r3,c





def funcion_exponencial(x, a, b, c):
    """
    Función exponencial negativa de la forma y = a * exp(-b * x) + c

    Parámetros:
    x -- valor de entrada
    a, b, c -- parámetros de la función exponencial

    Retorna:
    El valor de la función exponencial evaluada en x
    """
    return a * np.exp(-b * x) + c 





def ic50_r0(r,c,t,*args):
    """
    Función que representa la tasa de crecimiento frente a la concentración de fármaco a un determinado tiempo
    Calcula el IC50 y la tasa de crecimiento 0
    
    Parámetros:
    r -- tasas de crecimiento
    c -- concentraciones
    t -- día para el que se calcula el IC50
    *args -- sirve para añadir los errores de las tasas de crecimiento
    
    Retorna:
    Valor de IC50
    Valor a la que la tasa de crecimiento es 0
    """
    x = np.array(c)
    y = np.array(r)

    # Función exponencial que se utiliza para el ajuste
    def funcion_exponencial(x, a, b, c):
        """
        Función exponencial negativa de la forma y = a * exp(-b * x) + c

        Parámetros:
        x -- valor de entrada
        a, b, c -- parámetros de la función exponencial

        Retorna:
        El valor de la función exponencial evaluada en x
        """
        return a * np.exp(-b * x) + c

    # Ajuste exponencial utilizando curve_fit y ajuste de los parámetros
    bounds = ([0, 0, -np.inf], [np.inf, np.inf, 0])
    popt, pcov = curve_fit(funcion_exponencial, x, y, bounds=bounds)
    a_fit, b_fit, c_fit = popt

    # Gráfico de los datos y de la función ajustada
    plt.figure(figsize=(8,5))
    plt.scatter(x, y, s=25)
    xx = np.linspace(min(x),max(x),200)
    plt.plot(xx, funcion_exponencial(xx, a_fit, b_fit, c_fit), color='red')
    plt.xlabel('concentración de cisplatino (uM)',fontsize=15)
    plt.ylabel('tasa de crecimiento (r)',fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.title('Variación de la tasa de crecimiento con la dosis de fármaco',fontsize=14,pad=18)

    # Cálculo de tasa de crecimiento "cero" despejando al ecuación exponencial
    x0 = -((np.log(-c_fit/a_fit)/b_fit))
    plt.scatter(x0, 0, color='black', marker='o', s=50, label='crecimiento 0 = {:.2f} uM'.format(x0))
    print('La tasa de crecimiento r0 resulta en un valor de concentración de:',x0,'uM')

    # Cálculo de IC50 despejando la ecuación exponencial
    valor_fijo = (np.log(0.5))/t
    r_IC50 = r[0]+valor_fijo
    x_IC50 = -((np.log((r_IC50-c_fit)/a_fit)/b_fit))
    plt.plot([x_IC50, x_IC50], [0, r_IC50], 'k:', label='IC50 = {:.2f} uM'.format(x_IC50))
    plt.plot([0, x_IC50], [r_IC50, r_IC50], 'k:')
    print('La tasa de crecimiento rIC50 es',r_IC50,',que resulta en un valor de IC5O de:',x_IC50,'uM')
    plt.legend(fontsize=15)
    plt.grid(True)
    
    # Si se dan los errores como argumento, se representan, si no se sigue con la función
    if args:
        plt.errorbar(x, y,  yerr = args, linestyle="None",  fmt="ob",  capsize=3,  ecolor="k")
        plt.savefig('tasa-vs-concentracion.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        return x0,x_IC50
    else:
        plt.savefig('tasa-vs-concentracion.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        return x0,x_IC50
    

