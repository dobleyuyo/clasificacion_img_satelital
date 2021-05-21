# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 20:45:46 2020

@author: Pablo
"""


import numpy as np
from osgeo import gdal
from skimage.feature import greycomatrix, greycoprops
from sklearn.preprocessing import scale

# grey_image =ubicacion del raster en escala de grises
# window_size =tamaño de la ventana
# textures =lista con las texturas a calcular ['ASM','contrast','correlation','energy','homogeneity']
# directorio_guardado =carpeta de guardado
# texture_image = nombre generico para las nuevas imagenes de textura

# Crea rasters con los atributos de textura estandarizados.
# con formato float32 y con 5 decimales.


def glcm_ortomosaico_multiple(grey_image,window_size,textures,directorio_guardado,texture_image):
    
    
    #abro el raster
    src_ds = gdal.Open(grey_image)
    raster_img= src_ds.ReadAsArray()
    raster_img=np.array(raster_img,dtype='uint8')
    
    #testrraster es donde sea crea la imagen con textura
    testraster = np.full(shape=(len(textures),raster_img.shape[0],raster_img.shape[1]),
                         fill_value=0,dtype='float64')
    
    # bordes de la imagen para que las ventanas entren completas
    borde_fila_min=window_size//2
    borde_fila_max=(raster_img.shape[0] - (window_size//2) +1)
        
    borde_columna_min=window_size//2
    borde_columna_max=(raster_img.shape[1] - (window_size//2) +1)
    
    # lista de pixeles CON DATO
    pixel_data_list=np.where(raster_img!=0)
    
    # recorro la imagen con la ventana , teniendo en cuenta los bordes
    for i in range(len(pixel_data_list[0])):
        if pixel_data_list[0][i] <borde_fila_min or pixel_data_list[1][i]<borde_columna_min:
            continue
        elif pixel_data_list[0][i]>borde_fila_max  or pixel_data_list[1][i]>borde_columna_max:
            continue
        else:
            glcm_window = raster_img[pixel_data_list[0][i]-(window_size//2) : pixel_data_list[0][i]+(window_size//2)+1 ,
                                     pixel_data_list[1][i]-(window_size//2) : pixel_data_list[1][i] +(window_size//2)+1]    
        
        if np.any(glcm_window==0)==True:
            continue
        else:
            glcm = greycomatrix(glcm_window, [1], [0,np.pi/4,np.pi/2,3*np.pi/4],  symmetric = True, normed = True, levels=256)
            #Calculo las propiedades y reemplazo el pixel del centro
            for k in range(len(textures)):
                texture_prop = greycoprops(glcm,textures[k])
                average_texture=np.average(texture_prop, axis=1, weights=None, returned=False)
                testraster[k,pixel_data_list[0][i],pixel_data_list[1][i]]= average_texture
                
                
        print(len(pixel_data_list[0])-i, flush=True)
            
    #Escribir y guardar el nuevo raster
    num_columnas = src_ds.RasterXSize
    num_filas = src_ds.RasterYSize
    
    for k in range(len(textures)):
        outFileName=directorio_guardado + texture_image +'_'+ textures[k] +'_x'+ str(window_size) +'.tif'
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(outFileName, num_columnas,num_filas,1, gdal.GDT_Float64) # GDT_Byte = numero entero 8 bits GDT_Byte
        outdata.SetGeoTransform(src_ds.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(src_ds.GetProjection())##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(testraster[k])
        outdata.GetRasterBand(1).SetNoDataValue(0)##if you want these values transparent
    
        outdata.FlushCache() ##saves to disk!!
        outdata = None    
        print(textures[k],'ready')
    
    print('files saved in',directorio_guardado)
    