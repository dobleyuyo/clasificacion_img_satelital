# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:37:26 2020

@author: Pablo
"""

import time
import numpy
from osgeo import gdal

from sklearn.preprocessing import scale
from funciones_procesamiento_ortomosaico import remodelamiento_para_indices

# ubicacion_ortomosaico = ubicacion del ortomosaico RGB .tif
# INDICE = indice a calcular: GLI, CIVE, I, GR, NGB, RB, NRB, GRAY
# ubicacion_indice_ortomosaico = nombre y ubicacion del nuevo raster .tif

# REQUISITO: importar funciones_procesamiento_ortomosaico

# calculo de distintos indices de vegetacion ( incluye escala de grises) para una imagen RGB.
# los indices son estandarizados y guardados en un raster float32

# la funcion devuelve la ubicacion de la imagn del indice calculado

# estandarizados con 4 decimales y multiplicados por 10000
def ortomosaico_computo_indices (ubicacion_ortomosaico,INDICE,ubicacion_indice_ortomosaico):
           
  
    #############################################################################
    # PRE-PROCESAMIENTO NUEVOS DATOS #
    #############################################################################
    
    #abro el raster
    #ubicacion_raster='C:\\Pablo Machine Learning\\Inventario humedales\\shapes\\ort_corte_prueba_chica.tif'
    ubicacion_raster=ubicacion_ortomosaico
    
    src_ds = gdal.Open(ubicacion_raster)
    
    raster_fuente=remodelamiento_para_indices(ubicacion_raster)
    
    #############################################################################
    # IDENTIFICACION PIXELES VACIOS #
    #############################################################################
    
    # Identificacion de pixeles vacios/en blanco (NoData=255)
    # lista de T/F de pixeles vacios
    
    lista_pixel_vacios=numpy.all(raster_fuente==255, axis=1)
    
    #############################################################################
    # CALCULO DE INDICES DE VEGETACION #
    #############################################################################
    
    #creo el array para guardar el calculo del indice
    raster_indices=numpy.zeros((raster_fuente.shape[0]),dtype='float64')
    
    # agrego el valor de los pixeles vacios al raster_indices      
    raster_indices[lista_pixel_vacios]=0
    
    # calculo del indice seleccionado para todos los pixeles con datos
    
    R = raster_fuente[lista_pixel_vacios==False][:,0]
    G = raster_fuente[lista_pixel_vacios==False][:,1]
    B = raster_fuente[lista_pixel_vacios==False][:,2]
    
    if INDICE=='GLI':
        # GLI
        raster_indices[lista_pixel_vacios==False] = (((2*G)-R-B) / (2*(G+R+B)))
    
    elif INDICE=='CIVE':
        # CIVE
        raster_indices[lista_pixel_vacios==False] =(0.441*R) -(0.881*G) + (0.385*B) + 18.78745
    
    elif INDICE=='I':
        # I
        raster_indices[lista_pixel_vacios==False] = R+G+B
    
    elif INDICE=='GR':
        # Green to red ratio
        raster_indices[lista_pixel_vacios==False] = G/R
    
    elif INDICE=='NGB':
        # Green normalized by blue (NGB)
        raster_indices[lista_pixel_vacios==False] = (G-B) / (G+B)
    
    elif INDICE=='RB':
        # Red to blue ratio
        raster_indices[lista_pixel_vacios==False] = R/B
    
    elif INDICE=='NRB':
        # Red normalized by blue (NRB)
        raster_indices[lista_pixel_vacios==False] = (R-B) / (R+B)
    
    elif INDICE=='GRAY':
        #convert raster to grayscale from RGB
        raster_indices[lista_pixel_vacios==False] = (0.2125*R) + (0.7154*G) + (0.0721*B)
            
            
    if INDICE =='GRAY':
        raster_indices=numpy.array(raster_indices,dtype='u8')
    
    #############################################################################
    #  RECONSTRUCCION DEL RASTER #
    #############################################################################
    #información del raster
    num_columnas = src_ds.RasterXSize
    num_filas = src_ds.RasterYSize
    
    # re-arreglo de las dimensiones del raster
    raster_indices=raster_indices.reshape([num_filas,num_columnas])    
    
    # Escribir y guardar el nuevo raster
    #outFileName="C:\\Pablo Machine Learning\\Inventario humedales\\reconstrucciones\\ort_corte_prueba_chica_CIVE.tif"
    outFileName=ubicacion_indice_ortomosaico
    
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outFileName, num_columnas,num_filas,1, gdal.GDT_Float64) # GDT_Byte = numero entero 8 bits GDT_Byte
    outdata.SetGeoTransform(src_ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(src_ds.GetProjection())##sets same projection as input
    
    outdata.GetRasterBand(1).WriteArray(raster_indices)
    outdata.GetRasterBand(1).SetNoDataValue(0)##if you want these values transparent
    
    outdata.FlushCache() ##saves to disk!!
    
    outdata = None
    
    
    print('new orthomosaic saved in:',ubicacion_indice_ortomosaico)
    
    return ubicacion_indice_ortomosaico
