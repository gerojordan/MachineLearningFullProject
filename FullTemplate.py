#la finalidad de este proyecto es unificar e iniciar un procedimiento de creacion de modelo de machine learning con todas las etapas
#de pruebas de datos necesarias para validar los datos y crear nuestro modelo y probarlo.
#
#
#


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
plt.rc("font", size=14)
import seaborn as sb
sb.set(style="white")
sb.set(style="whitegrid", color_codes=True)
%matplotlib inline

#tipos de datos
#custcode        14522 non-null int64 CONTINUO cuantitativo
#Año             14522 non-null int64 CONTINUO cuantitativo
#Trimestre       14522 non-null object CATEGORICO cualitativo
#Mes             14522 non-null object CATEGORICO cualitativo
#Día             14522 non-null int64 CONTINUO cuantitativo
#-------#Tipo_Cliente    14522 non-null object CATEGORICO cualitativo
#-------#Total abono     14342 non-null float64 CONTINUO cuantitativo
#-------#declinecause    14522 non-null object CATEGORICO cualitativo
#-------#saldo           14522 non-null float64 CONTINUO cuantitativo
#Baja            14522 non-null int64 CONTINUO cuantitativo

#--------------------------------------LIMPIEZA DE DATOS----------------------------------------------------------------------
#cargamos el archivo
print("----HEAD----")#veremos de que se tratan los datos, sus columnas, tipos de datos, etc
dataframe = pd.read_csv(r"bajas_entrenamiento.csv")
dataframe.head()

dataframe.info() #datatypes y valores perdidos

# Reemplaza todas las apariciones de Not Available con numpy, no un número
dataframe = dataframe.replace({'Not Available': np.nan})

dataframe['declinecause'] = np.where(dataframe['declinecause'] =='CAMEM', 1, dataframe['declinecause'])
dataframe['declinecause'] = np.where(dataframe['declinecause'] =='INCOB', 2, dataframe['declinecause'])
dataframe['declinecause'] = np.where(dataframe['declinecause'] =='PROEC', 3, dataframe['declinecause'])
dataframe['declinecause'] = np.where(dataframe['declinecause'] =='NOUSA', 4, dataframe['declinecause'])
dataframe['declinecause'] = np.where(dataframe['declinecause'] =='CAMBI', 5, dataframe['declinecause'])
dataframe['declinecause'] = np.where(dataframe['declinecause'] =='MUDAN', 6, dataframe['declinecause'])
dataframe['declinecause'] = np.where(dataframe['declinecause'] =='AUMEN', 7, dataframe['declinecause'])
dataframe['declinecause'] = np.where(dataframe['declinecause'] =='DISCO', 8, dataframe['declinecause'])
dataframe['declinecause'] = np.where(dataframe['declinecause'] =='RAZON', 9, dataframe['declinecause'])
dataframe['declinecause'] = np.where(dataframe['declinecause'] =='CIERR', 10, dataframe['declinecause'])
dataframe['declinecause'] = np.where(dataframe['declinecause'] =='REFOR', 11, dataframe['declinecause'])
dataframe['declinecause'] = np.where(dataframe['declinecause'] =='NOINF', 12, dataframe['declinecause'])
dataframe['declinecause'] = np.where(dataframe['declinecause'] =='FACTU', 13, dataframe['declinecause'])
dataframe['declinecause'] = np.where(dataframe['declinecause'] =='EVENT', 14, dataframe['declinecause'])


dataframe['Tipo_Cliente'] = np.where(dataframe['Tipo_Cliente'] =='Comercial', 1, dataframe['Tipo_Cliente'])
dataframe['Tipo_Cliente'] = np.where(dataframe['Tipo_Cliente'] =='Residencial', 2, dataframe['Tipo_Cliente'])


#dataframe["Tipo_Cliente"].unique() #corroboramos el cambio

# Iterar a través de las columnas 
for col in list(dataframe.columns):
    # seleccionamos las columnas que deben ser numericas
    if ('Tipo_Cliente' in col or 'declinecause' in col):
        # convertir tipo de dato a float
        dataframe[col] = dataframe[col].astype(float)
        
#analizamos valores faltantes, porcentajes de los mismos por columna etc. deberiamos eliminar cualquier columna que no sea logica el agregado de datos o rompa su logica cuando sea mayor al 50% la cantidad faltante
def missing_zero_values_table(df):
        zero_val = (df == 0.00).astype(int).sum(axis=0)
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
        mz_table = mz_table.rename(
        columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})
        mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']
        mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(df)
        mz_table['Data Type'] = df.dtypes
        mz_table = mz_table[
            mz_table.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
            "There are " + str(mz_table.shape[0]) +
              " columns that have missing values.")
#         mz_table.to_excel('D:/sampledata/missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)
        return mz_table

missing_zero_values_table(dataframe)

#eliminamos anomalias basadas en la definicion de valores extremos atipicos
#VER MAS TARDE

#--------------------------------------EDA---------------------------------------------------------------------------------
#analizaremos histograma de la variable para examinar su distribucion

import matplotlib.pyplot as plt
## Histograma de baja #anotar en un informe los resultados analizados
#plt.style.use('fivethirtyeight')
#plt.hist(dataframe['Baja'].dropna(), bins = 100, edgecolor = 'k');
#plt.xlabel('Baja'); plt.ylabel('Cantidad de bajas'); 
#plt.title('Distribucion de Bajas');

#Busqueda de relaciones entre caracteristicas y objetivos
##Comenzamos con Tipo_Cliente y Baja

## Crea una lista de clientes con más de 100 medidas
#types = dataframe.dropna(subset=['Baja'])
#types = types['Tipo_Cliente'].value_counts()
#types = list(types[types.values > 100].index)

## Gráfico de distribución de baja para categorías de clientes
#figsize = (12, 10)

## Dibujar cada cliente
#for b_type in types:
#    # Selecciona el tipo de edificio
#    subset = dataframe[dataframe['Tipo_Cliente'] == b_type]
    
#    # Density plot of Energy Star scores
#    sb.kdeplot(subset['Baja'].dropna(),
#               label = b_type, shade = False, alpha = 0.8);
    
## etiquete el gráfico
#plt.xlabel('Baja', size = 20); plt.ylabel('Densidad', size = 20); 
#plt.title('Densidad de baja en base a tipo de cliente', size = 28);

##seguimos con declinecause y Baja

## Crea una lista de clientes con más de 100 medidas
#types = dataframe.dropna(subset=['Baja'])
#types = types['declinecause'].value_counts()
#types = list(types[types.values > 100].index)

## Gráfico de distribución de baja para causas de decline
#figsize = (12, 10)

## Dibujar cada cliente
#for b_type in types:
#    # Selecciona el tipo de causa
#    subset = dataframe[dataframe['declinecause'] == b_type]
    
#    # Densidad de baja
#    sb.kdeplot(subset['Baja'].dropna(),
#               label = b_type, shade = False, alpha = 0.8);
    
## etiquete el gráfico
#plt.xlabel('Baja', size = 20); plt.ylabel('Densidad', size = 20); 
#plt.title('Densidad de baja en base a causa', size = 28);

##seguimos con total abono y Baja

## Crea una lista de clientes con más de 100 medidas
#types = dataframe.dropna(subset=['Baja'])
#types = types['Total_abono'].value_counts()
#types = list(types[types.values > 100].index)

## Gráfico de distribución de baja por Total abono
#figsize = (12, 10)

## Dibujar cada cliente
#for b_type in types:
#    # Selecciona el total abono
#    subset = dataframe[dataframe['Total_abono'] == b_type]
    
#    # Densidad por total abono
#    sb.kdeplot(subset['Baja'].dropna(),
#               label = b_type, shade = False, alpha = 0.8);
    
## etiquete el gráfico
#plt.xlabel('Baja', size = 20); plt.ylabel('Densidad', size = 20); 
#plt.title('Densidad de baja en base a Total abono', size = 28);

##seguimos con total abono y Baja

## Crea una lista de clientes con más de 100 medidas
#types = dataframe.dropna(subset=['Baja'])
#types = types['Total_abono'].value_counts()
#types = list(types[types.values > 100].index)

## Gráfico de distribución de baja por Total abono
#figsize = (12, 10)

## Dibujar cada cliente
#for b_type in types:
#    # Selecciona el total abono
#    subset = dataframe[dataframe['Total_abono'] == b_type]
    
#    # Densidad por total abono
#    sb.kdeplot(subset['Baja'].dropna(),
#               label = b_type, shade = False, alpha = 0.8);
    
# etiquete el gráfico
#plt.xlabel('Baja', size = 20); plt.ylabel('Densidad', size = 20); 
#plt.title('Densidad de baja en base a Total abono', size = 28);

##terminamos con saldo y Baja

## Crea una lista de clientes con más de 100 medidas
#types = dataframe.dropna(subset=['Baja'])
#types = types['saldo'].value_counts()
#types = list(types[types.values > 100].index)

## Gráfico de distribución de baja por Total abono
#figsize = (12, 10)

## Dibujar cada cliente
#for b_type in types:
#    # Selecciona el saldo
#    subset = dataframe[dataframe['saldo'] == b_type]
    
#    # Densidad por saldo
#    sb.kdeplot(subset['Baja'].dropna(),
#               label = b_type, shade = False, alpha = 0.8);
    
# etiquete el gráfico
#plt.xlabel('Baja', size = 20); plt.ylabel('Densidad', size = 20); 
#plt.title('Densidad de baja en base a saldo', size = 28);

# CALCULAMOS Coeficiente de Correlación de Pearson para cuantificar las correlaciones entre variables. Esta es una medida de la fuerza y dirección de una relación lineal entre dos variables. Una puntuación de +1 es una relación positiva perfectamente lineal y una puntuación de -1 es una relación lineal perfectamente negativa.
#dataframe.corr()['Baja'].sort_values()

#grafico de 2 variables para visualizar relacion de 2 variables continuas grafico de dispersion y hasta coloreams por una variable categorica
#VER MAS TARDE

#sb.set(style="whitegrid", color_codes=True)

#eliminmos las columnas innecesarias
#dataframe = dataframe.drop(columns="custcode")        
#dataframe = dataframe.drop(columns="Año")              
#dataframe = dataframe.drop(columns="Trimestre")       
#dataframe = dataframe.drop(columns="Mes")             
#dataframe = dataframe.drop(columns="Día")       
#dataframe = dataframe.dropna()

# Función para calcular el coeficiente de correlación entre dos columnas
#def corr_func(x, y, **kwargs):
#    r = np.corrcoef(x, y)[0][1]
#    ax = plt.gca()
#    ax.annotate("r = {:.2f}".format(r),
#                xy=(.2, .8), xycoords=ax.transAxes,
#                size = 20)
    
# Crear el objeto pairgrid
#grid = sb.PairGrid(data = dataframe, size = 3)

# Upper es un scatter plot (es un diagrama de dispersión)
#grid.map_upper(plt.scatter, color = 'red', alpha = 0.6)

# Diagonal es un histograma
#grid.map_diag(plt.hist, color = 'red', edgecolor = 'black')

# La parte inferior es un gráfico de correlación y densidad
#grid.map_lower(corr_func);
#grid.map_lower(sb.kdeplot, cmap = plt.cm.Reds)

##############################################################################################################
#Ingenieria y seleccion de caracteristics
##############################################################################################################

#CONVERSION DE COLUMNAS CATEGORICAS A NUMERICAS
# Copia los datos originales
#features = dataframe.copy()

# Selecciona las columnas numéricas
#numeric_subset = dataframe.select_dtypes('number')

# Crea columnas con registro de las columnas numéricas
#for col in numeric_subset.columns:
    # Skip the Energy Star Score column
#    if col == 'Baja':
#        next
#    else:
#        numeric_subset['log_' + col] = np.log(numeric_subset[col])
        
# Seleccione las columnas categóricas
#categorical_subset = data[['Borough', 'Largest Property Use Type']]

# Codificación One hot
#categorical_subset = pd.get_dummies(categorical_subset)

# Une los dos dataframes usando concat
# Asegúrate de usar axis = 1 para realizar un enlace de columna
#features = pd.concat([numeric_subset, categorical_subset], axis = 1)

#SELECCION DE CARACTERISTICAS
#correlaciones de características con otras características, no de correlaciones con el objetivo
#Las características que están fuertemente correlacionadas entre sí se conocen como colineales y la eliminación de una de las variables de estos pares de características a menudo puede ayudar a un modelo machine learning a generalizarse y ser más interpretable.
#Dejaremos caer/eliminamos una de un par de características si el coeficiente de correlación entre ellas es superior a 0,6

# Eliminar cualquier columna con todos los valores na
#features  = features.dropna(axis=1, how = 'all')
#print(features.shape)(11319, 65)

#Establecer linea de base
X = np.array(dataframe.drop(['custcode','Año','Trimestre', 'Mes', 'Día','Baja'],1))
y = np.array(dataframe['Baja'])
X.shape
# Se divide en un 70% de entrenamiento y un 30% de pruebas
from sklearn.model_selection import train_test_split
validation_size = 0.30
seed = 7
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)


# Función para calcular el error absoluto medio
def mae(y_true, y_pred):
    return np.mean(abs(y_true - y_pred))

baseline_guess = np.median(y)

print('El punto de partida es una puntuación de %0.2f' % baseline_guess)
print("Desempeño de la línea de base en el equipo de prueba: MAE = %0.4f" % mae(y_test, baseline_guess))