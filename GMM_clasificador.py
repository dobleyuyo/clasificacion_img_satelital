# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 21:36:20 2020

@author: Pablo
"""
import sys
import numpy
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from osgeo import gdal
from sklearn import mixture
import itertools
from scipy import linalg
import joblib
import glob
import time


#linux
sys.path.append('/home/pabloav/Documents/Inventario_humedales/codigo/')

from funciones_procesamiento_ortomosaico import remodelamiento_ortomosaico

#############################################################################
#  CARGAR ALGORITMO ENTRENADO #
#############################################################################
        
# carga del modelo desde la carpeta
directorio='/home/pabloav/Documents/Inventario_humedales/codigo/'
ubicacion_modelo = directorio +'GMM_clasificador_5atributos_full_120_ultimo'+'.sav'
GMM_cargado = joblib.load(ubicacion_modelo)



#############################################################################
# PRE-PROCESAMIENTO DATOS para entrenamiento #
#############################################################################

#ubicacion de los rasters 
carpeta='/home/pabloav/Documents/Inventario_humedales/reconstrucciones/ort_representativos_utm21/ort_representativo_std/'
#carpeta='C:\\Pablo Machine Learning\\Inventario humedales\\reconstrucciones\\unsam\\ort_representativos_utm21\\ort representativo std\\'
# lista de los nombres de los atributos a utilizar.
# La textura calculada, el tamaño de ventana ; indices de vegetacion
lista_archivos=[]
lista_glcm=[]
texturas=['ASM','energy']
ventanas=['21x21']
indices_vegetacion=['GLI','GR','banda_B']
ext='.tif'


for atributo in range (0,len(texturas)):
    for tamaño in range (0,len(ventanas)):
        lista_glcm.append(ventanas[tamaño] +'_' +texturas[atributo])


for i in range (0, len(lista_glcm)):
    lista_archivos.append(carpeta +'ort_representativo_glcm_' +lista_glcm[i]+'_utm21s_std'+ext)
for j in range (0, len(indices_vegetacion)):
    lista_archivos.append(carpeta +'ort_representativo_' + indices_vegetacion[j] +'_utm21s_std'+ext)


# Creo una lista para guardar cada raster
# la lista esta no vacía para ser más eficiente
lista_rasters=list(range(len(lista_archivos)))
start = time.time()
for i in range(0,len(lista_archivos)):
    lista_rasters[i]=remodelamiento_ortomosaico(lista_archivos[i])
    print(len(lista_archivos)-i-1,flush=True)


# uno todos los array por columnas para alimentar al RF
raster_a_clasificar=numpy.concatenate((lista_rasters),axis=1)
raster_a_clasificar=numpy.array(raster_a_clasificar,dtype='int')


print('datos cargados', flush=True)

#############################################################################
#  USAR ALGORITMO ENTRENADO #
#############################################################################

# Podría seleccionar sólo las bandas que me intresan
# data_z=raster_a_clasificar[['banda R','banda G', 'banda B']]

#divido en DataFrames más pequeños para poder clasificar, para no saturar la
# memoria de la computadora
n_items = raster_a_clasificar.shape[0]//62 #numero de items de cada sub-DataFrame
raster_a_clasificar = [raster_a_clasificar[i:i+n_items] 
                      for i in range(0,raster_a_clasificar.shape[0],n_items)]


#creo el Dataframe para las clasificaciones obtenidas
raster_predicciones=numpy.full((lista_rasters[0].shape[0]),0,dtype='uint16')
raster_predicciones=[raster_predicciones[i:i+n_items] 
                      for i in range(0,raster_predicciones.shape[0],n_items)]



for i in range (0,len(raster_a_clasificar)):
    lista_pixel_vacios=numpy.any(raster_a_clasificar[i]==0, axis=1)
    if numpy.any(lista_pixel_vacios==False):
        raster_predicciones[i][lista_pixel_vacios==False]=GMM_cargado.predict(raster_a_clasificar[i][lista_pixel_vacios==False])
    print(len(raster_a_clasificar)-i-1,flush=True)
    


print('datos clasificados', flush=True)
# junto las clasificaciones en un solo DataFrame
raster_predicciones=numpy.concatenate(raster_predicciones)


### reconstruccion del raster ###

#información del raster
src_ds = gdal.Open(lista_archivos[0])
num_columnas = src_ds.RasterXSize
num_filas = src_ds.RasterYSize

# re-arreglo de los nuevos features o clasificaciones
raster_predicciones=raster_predicciones.reshape([num_filas,num_columnas])

outFileName= directorio +'reconstruccion_ort_representativo_GMM_full120_5_atributos_ultimo.tif'
geo_transform=src_ds.GetGeoTransform()##sets same geotransform as input
proyeccion=src_ds.GetProjection()##sets same projection as input


driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(outFileName, num_columnas,num_filas,1,gdal.GDT_UInt16) # GDT_Byte = numero entero 8 bits GDT_Byte
outdata.SetGeoTransform(src_ds.GetGeoTransform())##sets same geotransform as input
outdata.SetProjection(src_ds.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(raster_predicciones)
outdata.GetRasterBand(1).SetNoDataValue(0)##if you want these values transparent
outdata.FlushCache() ##saves to disk!!


print('raster guardado', flush=True)


outdata = None
band=None
ds=None
