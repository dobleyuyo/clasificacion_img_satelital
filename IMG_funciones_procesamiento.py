# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:48:59 2020

@author: Pablo
"""


### Seccion de imports ###
import numpy
from osgeo import gdal
import pandas
from sklearn.preprocessing import scale

#################################################################################
# Scrip que contiene las funciones comunes en el procesamiento del otromosaico. #
#################################################################################

## Reordena una muestra del ortomosaico en un dataframe donde cada fila es un pixel con columnas RGB.
## Ademas agrega una pimer columna con el nombre de la clase a la cual pertenece dicha muestra.
## Las columnas tienen nombre
## Elimina las filas de pixeles vacios.

def remodelamiento_muestra(clase,ubicacion_raster):
    
    #abro el raster
    src_ds = gdal.Open(ubicacion_raster)
    
    # volcado de datos en un numpy array
    data = src_ds.ReadAsArray()
    
    # con Flaten hago un vector lineal por cada banda y al transponerlo (T), 
    # obtengo una array para el Random Forest
    new_data=numpy.array([data[0].flatten(),data[1].flatten(),data[2].flatten()]).T
    
    # elimino filas de ceros
    new_data= new_data[~numpy.all(new_data == 255, axis=1)]
    
    # Convierto el array en un dataframe, agrego el nombre de las bandas 
    new_dataframe = pandas.DataFrame(data=new_data, 
                                     columns=['banda R', 'banda G', 'banda B'])
    
    # inserto la clase de pixel que es
    new_dataframe.insert(0, 'Clase', clase)
    
    return new_dataframe 



## Reordena el ortomosaico en un array donde cada fila es un pixel con columnas RGB
    
def remodelamiento_para_indices(ubicacion_raster):
    
    #abro el raster
    src_ds = gdal.Open(ubicacion_raster)
    
    # volcado de datos en un numpy array
    data = src_ds.ReadAsArray()    
    
    # con Flaten hago un vector lineal por cada banda y al transponerlo (T), 
    # con datos dtype='int16'
    new_data=numpy.array([data[0].flatten(),data[1].flatten(),data[2].flatten()],dtype='int16').T
   
    return new_data


## Reordena el ortomosaico en un array donde cada fila es un pixel y cada banda es una columna
## La primer columna es la clase y el resto las diferentes bandas
    
def remodelamiento_muestra_gps(clase,ubicacion_raster,lista_bandas):
    #abro el raster
    src_ds = gdal.Open(ubicacion_raster)
    
    # volcado de datos en un numpy array
    data = src_ds.ReadAsArray()
    
    # con Flaten hago un vector lineal por cada banda y al transponerlo (T), 
    # obtengo una  columna para del Dataframe final 
    # descarto la última columna ya que viene del raster RGB y está vacía
    new_data=numpy.zeros((data.shape[1]*data.shape[2],data.shape[0],))
    
    for banda in range (0,data.shape[0]):
        new_data[:,banda]=data[banda].flatten().T
    
    # elimino filas de ceros
    new_data= new_data[~numpy.all(new_data == 255, axis=1)]
    new_data= new_data[~numpy.all(new_data == 0, axis=1)]
    
    # Convierto el array en un dataframe, agrego el nombre de las bandas 
    new_dataframe = pandas.DataFrame(data=new_data, 
                                     columns=lista_bandas)
    
    # inserto la clase de pixel que es
    new_dataframe.insert(0, 'Clase', clase)
    
    return new_dataframe



## Reordena el ortomosaico en un dataframe donde cada fila es un pixel con las bandas en columnas.
def remodelamiento_ortomosaico(ubicacion_raster):
    #abro el raster
    src_ds = gdal.Open(ubicacion_raster)
    num_columnas = src_ds.RasterXSize
    num_filas = src_ds.RasterYSize
    num_bandas = src_ds.RasterCount
    # volcado de datos en un numpy array
    data = src_ds.ReadAsArray()
    
    # con Flaten hago un vector lineal por cada banda y al transponerlo (T), 
    # obtengo una  columna para del Dataframe final 
    if num_bandas==1:
        new_data=new_data=numpy.zeros((num_filas*num_columnas,num_bandas))
        new_data[:,0]=data.flatten().T
    else:
        new_data=numpy.zeros((num_filas*num_columnas,num_bandas))
        for banda in range (0,num_bandas):
            new_data[:,banda]=data[banda].flatten().T
       
    
    return new_data

## Estandarizacion de un raster unibanda. Se estandariza  se redondea a 4 decimales y multiplica por 
## 10.000, se guarda en Int16

def std_raster(ubicacion_raster):
    #abro el raster
    src_ds = gdal.Open(ubicacion_raster)
    num_columnas = src_ds.RasterXSize
    num_filas = src_ds.RasterYSize
    num_bandas = src_ds.RasterCount
    # volcado de datos en un numpy array
    data = src_ds.ReadAsArray()
    # con Flaten hago un vector lineal por cada banda y al transponerlo (T), 
    # obtengo una  columna 
    new_data=numpy.zeros((num_filas*num_columnas,num_bandas))
    new_data[:,0]=data.flatten().T
    new_data=numpy.round(new_data,7)
    
    # Obtengo el NoDataValue para poder excluir los pixeles vacios
    srcband = src_ds.GetRasterBand(1)
    NoData=srcband.GetNoDataValue()
    
    w_NoData=numpy.all(new_data == NoData,axis=1)
    

    # estadarizacion de los datos
    new_data[~w_NoData] = scale(new_data[~w_NoData])
    new_data[~w_NoData] = numpy.around(new_data[~w_NoData],decimals=4)*10000
    

    # pixeles vacios =255
    new_data[w_NoData]=0
#    new_data[w_0]=0
    
    #remodelacicion del array
    new_data=new_data.reshape([num_filas,num_columnas])    
    
    # Escribir y guardar el nuevo raster
    ubicacion_ortomosaico_std=ubicacion_raster.replace('.tif','_std.tif')
    
    outFileName=ubicacion_ortomosaico_std
    
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outFileName, num_columnas,num_filas,1, gdal.GDT_Int32) # GDT_Byte = numero entero 8 bits GDT_Byte
    outdata.SetGeoTransform(src_ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(src_ds.GetProjection())##sets same projection as input
    
    outdata.GetRasterBand(1).WriteArray(new_data)
    outdata.GetRasterBand(1).SetNoDataValue(0)##if you want these values transparent
    
    outdata.FlushCache() ##saves to disk!!
    
    outdata = None
    
    print('new orthomosaic saved in:',ubicacion_ortomosaico_std)
    
    return ubicacion_ortomosaico_std

# Cambia los NaN de un raster por el valor de NoData
def NaN_to_NoData(ubicacion_raster):
    #abro el raster
    src_ds = gdal.Open(ubicacion_raster)
    num_columnas = src_ds.RasterXSize
    num_filas = src_ds.RasterYSize
    num_bandas = src_ds.RasterCount
    # volcado de datos en un numpy array
    data = src_ds.ReadAsArray()
    # Obtengo el NoDataValue para poder excluir los pixeles vacios
    srcband = src_ds.GetRasterBand(1)
    NoData=srcband.GetNoDataValue()
    
    # pixeles con NaN
    array_NaN=numpy.isnan(data)
    w_NaN=numpy.where(array_NaN==True)
    
    #reemplazo los NaN con NoData    
    data[w_NaN]=NoData
    
    # Escribir y guardar el nuevo raster
    ubicacion_ortomosaico_std=ubicacion_raster.replace('.tif','_fixed.tif')
    
    outFileName=ubicacion_ortomosaico_std
    
    # directorio con los tipos de datos posibles
    NP2GDAL_CONVERSION = {'Unknown' : 0, 'Byte' : 1, 'UInt16' : 2, 'Int16' : 3,
                          'UInt32' : 4, 'Int32' : 5, 'Float32' : 6, 'Float64' : 7,
                          'CInt16' : 8, 'CInt32' : 9, 'CFloat32' : 10, 'CFloat64' : 11,
                          'TypeCount' : 12}
    
    tipo_dato=NP2GDAL_CONVERSION[gdal.GetDataTypeName(srcband.DataType)]
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(outFileName, num_columnas,num_filas,1,tipo_dato) # GDT_Byte = numero entero 8 bits GDT_Byte
    outdata.SetGeoTransform(src_ds.GetGeoTransform())##sets same geotransform as input
    outdata.SetProjection(src_ds.GetProjection())##sets same projection as input
    
    outdata.GetRasterBand(1).WriteArray(data)
    outdata.GetRasterBand(1).SetNoDataValue(0)##if you want these values transparent
    
    outdata.FlushCache() ##saves to disk!!
    
    outdata = None
    
    print('new orthomosaic saved in:',ubicacion_ortomosaico_std)
    
    return ubicacion_ortomosaico_std
    
    
    
    
## Reordena el ortomosaico en un array donde cada fila es un pixel y cada banda es una columna
## La primer columna es la clase y el resto las diferentes bandas
    
def remodelamiento_muestra_gps_v2(ubicacion_raster,lista_bandas):
    #abro el raster
    src_ds = gdal.Open(ubicacion_raster)
    
    # volcado de datos en un numpy array
    data = src_ds.ReadAsArray()
    
    # con Flaten hago un vector lineal por cada banda y al transponerlo (T), 
    # obtengo una  columna para del Dataframe final 
    # descarto la última columna ya que viene del raster RGB y está vacía
    new_data=numpy.zeros((data.shape[1]*data.shape[2],data.shape[0],))
    
    for banda in range (0,data.shape[0]):
        new_data[:,banda]=data[banda].flatten().T
    
    # elimino filas de ceros
    new_data= new_data[~numpy.all(new_data == 0, axis=1)]
    
    # Convierto el array en un dataframe, agrego el nombre de las bandas 
    new_dataframe = pandas.DataFrame(data=new_data, 
                                     columns=lista_bandas)
    
    # inserto la clase de pixel , la tomo del nombre del archivo
    new_dataframe.insert(0, 'Clase', ubicacion_raster[-5])
    
    return new_dataframe
    
