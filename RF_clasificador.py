# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 20:29:18 2020

@author: Pablo
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:10:23 2020

@author: Pablo
"""
import numpy
from osgeo import gdal
import sys
import time
import joblib


#linux
sys.path.append('/home/pabloav/Documents/Inventario_humedales/codigo/')
from funciones_procesamiento_ortomosaico import remodelamiento_ortomosaico


#############################################################################
#  CARGAR ALGORITMO ENTRENADO #
#############################################################################
        
# carga del modelo desde la carpeta
directorio='/home/pabloav/Documents/Inventario_humedales/codigo/'
ubicacion_modelo = directorio +'RF_clasificador_muestras_gps_4_atributos_clases_lq_eringium_ASM_energy' +'.sav'
RF_cargado = joblib.load(ubicacion_modelo)

#############################################################################
# PRE-PROCESAMIENTO NUEVOS DATOS #
#############################################################################



#ubicacion de los rasters 
carpeta='/home/pabloav/Documents/Inventario_humedales/reconstrucciones/ort_representativos_utm21/ort_representativo_std/'

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
        lista_glcm.append(ventanas[tamaño]+'_' +texturas[atributo])


for i in range (0, len(lista_glcm)):
    lista_archivos.append(carpeta +'ort_representativo_glcm_'+lista_glcm[i]+'_utm21s_std'+ext)
for j in range (0, len(indices_vegetacion)):
    lista_archivos.append(carpeta +'ort_representativo_'+indices_vegetacion[j]+'_utm21s_std'+ext)


# Creo una lista para guardar cada raster
# la lista esta no vacía para ser más eficiente
lista_rasters=list(range(len(lista_archivos)))
start = time.time()
for i in range(0,len(lista_archivos)):
    lista_rasters[i]=remodelamiento_ortomosaico(lista_archivos[i])
    print(len(lista_archivos)-i-1,flush=True)


# uno todos los array por columnas para alimentar al RF
raster_a_clasificar=numpy.concatenate(lista_rasters,axis=1)
print(raster_a_clasificar.shape)

print('remodelamiento listo', flush=True)

#############################################################################
#  USAR ALGORITMO ENTRENADO #
#############################################################################
#
# Podría seleccionar sólo las bandas que me intresan
# data_z=raster_a_clasificar[['banda R','banda G', 'banda B']]

#divido en DataFrames más pequeños para poder clasificar, para no saturar la
# memoria de la computadora
n_items = raster_a_clasificar.shape[0]//62 #numero de items de cada sub-DataFrame
raster_a_clasificar = [raster_a_clasificar[i:i+n_items] 
                      for i in range(0,raster_a_clasificar.shape[0],n_items)]


#creo el Dataframe para las clasificaciones obtenidas
raster_predicciones=numpy.full((lista_rasters[0].shape[0]),'0',dtype='str')
raster_predicciones=[raster_predicciones[i:i+n_items] 
                      for i in range(0,raster_predicciones.shape[0],n_items)]



for i in range (0,len(raster_a_clasificar)):
    lista_pixel_vacios=numpy.any(raster_a_clasificar[i]==0,axis=1)
    if numpy.any(lista_pixel_vacios==False):
        raster_predicciones[i][lista_pixel_vacios==False]=RF_cargado.predict(raster_a_clasificar[i][lista_pixel_vacios==False])
    print(len(raster_a_clasificar)-i-1,flush=True)

#junto las clasificaciones en un solo DataFrame
raster_predicciones=numpy.concatenate(raster_predicciones)

#############################################################################
#  RECONSTRUCCION DEL RASTER #
#############################################################################

W=numpy.where(raster_predicciones=='l')
raster_predicciones[W]='1'

W=numpy.where(raster_predicciones=='m')
raster_predicciones[W]='3'

W=numpy.where(raster_predicciones=='n')
raster_predicciones[W]='6'

W=numpy.where(raster_predicciones=='o')
raster_predicciones[W]='7'

W=numpy.where(raster_predicciones=='p')
raster_predicciones[W]='8'

W=numpy.where(raster_predicciones=='q')
raster_predicciones[W]='9'

print('conversion lista', flush=True)      


#información del raster
src_ds = gdal.Open(lista_archivos[0])
num_columnas = src_ds.RasterXSize
num_filas = src_ds.RasterYSize

# re-arreglo de los nuevos features o clasificaciones
raster_predicciones=raster_predicciones.reshape([num_filas,num_columnas])


outFileName= directorio +'reconstruccion_ort_representativo_clases_lq_4_atributos_eringium_ASM_energy.tif'
geo_transform=src_ds.GetGeoTransform()##sets same geotransform as input
proyeccion=src_ds.GetProjection()##sets same projection as input


driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(outFileName, num_columnas,num_filas,1,gdal.GDT_Byte) # GDT_Byte = numero entero 8 bits GDT_Byte
outdata.SetGeoTransform(src_ds.GetGeoTransform())##sets same geotransform as input
outdata.SetProjection(src_ds.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(raster_predicciones)
outdata.GetRasterBand(1).SetNoDataValue(0)##if you want these values transparent
outdata.FlushCache() ##saves to disk!!

outdata = None
band=None
ds=None