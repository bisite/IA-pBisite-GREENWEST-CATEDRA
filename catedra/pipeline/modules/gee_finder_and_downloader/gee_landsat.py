import os
import numpy as np
from datetime import datetime
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import geemap
import ee
import re
from tqdm import tqdm
import datetime
from halo import Halo
from print_color import print
from pprint import pprint
from functools import partial
import traceback
from pyboxen import boxen

# Initialize Google Earth Engine
ee.Authenticate()
ee.Initialize(project=os.environ["GEE_PROJECT"])  # Google Cloud project with Earth Engine enabled

# Cloud masking (sin cambios)
def mask_clouds(image):
    qa = image.select('QA_PIXEL')
    cloud_shadow_bit = 1 << 3
    cloud_bit = 1 << 5
    mask = qa.bitwiseAnd(cloud_shadow_bit).eq(0).And(
            qa.bitwiseAnd(cloud_bit).eq(0))
    return image.updateMask(mask)

# Apply scale factors (sin cambios)
def apply_scale_factors(image):
    optical_bands = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    # thermal_bands = image.select('ST_B6').multiply(0.00341802).add(149.0)
    return image.addBands(optical_bands, None, True)
    # return image.addBands(optical_bands, None, True).addBands(
    #     thermal_bands, None, True)

# --- NUEVAS FUNCIONES OPTIMIZADAS ---

def add_all_indices(image, dataset_name):

    if '7' in dataset_name or '5' in dataset_name:
        """Añade todas las bandas de índices de vegetación a la vez."""
        ndvi = image.normalizedDifference(['SR_B4', 'SR_B3']).rename('NDVI')
        ndii = image.normalizedDifference(['SR_B4', 'SR_B5']).rename('NDII')
        gndvi = image.normalizedDifference(['SR_B4', 'SR_B2']).rename('GNDVI')
        
        # EVI usa una expresión. Asegúrate de que las bandas de entrada existen.
        evi = image.expression(
            '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
            {
                'NIR': image.select('SR_B4'),
                'RED': image.select('SR_B3'),
                'BLUE': image.select('SR_B1')
            }).rename('EVI')
        
    elif '8' in dataset_name:
        """Añade todas las bandas de índices de vegetación a la vez."""
        ndvi = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        ndii = image.normalizedDifference(['SR_B5', 'SR_B6']).rename('NDII')
        gndvi = image.normalizedDifference(['SR_B5', 'SR_B3']).rename('GNDVI')
        
        # EVI usa una expresión. Asegúrate de que las bandas de entrada existen.
        evi = image.expression(
            '2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1)',
            {
                'NIR': image.select('SR_B5'),
                'RED': image.select('SR_B4'),
                'BLUE': image.select('SR_B2')
            }).rename('EVI')
        
    else:
        raise ValueError('Dataset name must contain either "7" or "8"')
    
    return image.addBands([ndvi, ndii, gndvi, evi])
    

def calculate_vegetation_stats(image_collection, indices_list):
    """
    Calcula todas las estadísticas (min, max, median, mean, stdDev) 
    para todos los índices en una sola pasada.
    """
    if not isinstance(indices_list, list):
        indices_list = [indices_list]

    # 1. Combinar todos los reductores
    reducer = ee.Reducer.min().combine(
        ee.Reducer.max(), '', True
    ).combine(
        ee.Reducer.median(), '', True
    ).combine(
        ee.Reducer.mean(), '', True
    ).combine(
        ee.Reducer.stdDev(), '', True
    )
    
    # 2. Aplicar el reductor combinado a la colección
    # Seleccionamos solo las bandas de índices para reducir
    stats_image = image_collection.select(indices_list).reduce(reducer)
    
    # El resultado es una imagen con bandas como 'NDVI_min', 'NDVI_max', 'NDII_min', etc.
    return stats_image

def get_terrain_image(region):
    """
    Obtiene una única imagen multibanda con elevación, pendiente y aspecto.
    """
    # No es necesario clip(region) aquí, reduceRegions solo muestrea los puntos.
    dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
    slope = ee.Terrain.slope(dem).rename('SLO')
    aspect = ee.Terrain.aspect(dem).rename('ASP')
    elevation = dem.rename('ELE')
    
    # Concatenar en una sola imagen
    return ee.Image.cat([elevation, slope, aspect])

# Funciones de selección de dataset (sin cambios)
def select_dataset(year):
    dataset_names = {'Landsat5': 'LANDSAT/LT05/C02/T1_L2',
                     'Landsat7': 'LANDSAT/LE07/C02/T1_L2',
                     'Landsat8': 'LANDSAT/LC08/C02/T1_L2'}
    operation_years = {'Landsat5': [1983, 2011],
                       'Landsat7': [2000, 2023],
                       'Landsat8': [2013, datetime.datetime.now().year-1]}
    
    # TODO: Meter LANDSAT8 con comprobación de años, cálculo de las bandas y aplicación de correción  (distintos respecto a otros landat)
    
    if operation_years['Landsat5'][0] <= year <= (operation_years['Landsat7'][0] - 1):
        return dataset_names['Landsat5']
    elif operation_years['Landsat7'][0] <= year <= operation_years['Landsat8'][0] - 1:
        return dataset_names['Landsat7']
    elif operation_years['Landsat8'][0] <= year <= operation_years['Landsat8'][1]:
        return dataset_names['Landsat8']
    else:
        raise ValueError(f"Año fuera de rango. Debe estar entre {operation_years['Landsat5'][0]} y {datetime.datetime.now().year - 1}")

def select_dataset_name(year):
    operation_years = {'Landsat5': [1983, 2011],
                       'Landsat7': [2000, 2023],
                       'Landsat8': [2013, datetime.datetime.now().year-1]}
    
    if operation_years['Landsat5'][0] <= year <= (operation_years['Landsat7'][0] - 1):
        return 'Landsat5'
    elif operation_years['Landsat7'][0] <= year <= operation_years['Landsat7'][1] - 1:
        return 'Landsat7'
    elif operation_years['Landsat8'][0] <= year <= operation_years['Landsat8'][1]:
        return 'Landsat8'
    else:
        raise ValueError(f"Año fuera de rango. Debe estar entre {operation_years['Landsat5'][0]} y {datetime.datetime.now().year - 1}")
    
def pivot_to_medium_wide(df_largo):
    """
    Transforma un DataFrame "largo" de GEE en un formato "medio-ancho".

    Agrupa por 'parcela_id', 'ano' y 'season', creando columnas
    para cada métrica (ej. 'NDVI_mean', 'ASP_value').
    """
    
    # 1. Pivota la tabla para agrupar por estación
    #    'season' se pone en el índice para que se mantenga como una fila.
    df_medio_ancho = df_largo.pivot_table(
        index=['parcela_id', 'ano', 'latitude', 'longitude', 'season'],
        columns=['index', 'statistic'], # Crea nuevas columnas (ej. 'NDVI', 'mean')
        values='value'
    )

    # 2. Aplana las cabeceras MultiIndex ('NDVI', 'mean') -> 'NDVI_mean'
    df_medio_ancho.columns = df_medio_ancho.columns.map('_'.join)

    # 3. Resetea el índice para que 'parcela_id', 'ano', 'season', etc.
    #    vuelvan a ser columnas normales.
    df_medio_ancho = df_medio_ancho.reset_index()

    # 4. Renombra 'season' a 'estacion_id' según tu petición
    df_medio_ancho = df_medio_ancho.rename(columns={'season': 'estacion_id'})

    return df_medio_ancho

gaul = ee.FeatureCollection('WM/geoLab/geoBoundaries/600/ADM0')
spain = gaul.filter(ee.Filter.eq('shapeName', 'Spain'))

class GeeDownloadLandsatData():
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.num_elements_per_chunk = 6000 # Nota: tu código actual no usa chunking

    def download_data(self, start_year, batch_size=500, verbose=False, callback_function=None, return_type='dict'):
        
        # --- Listas de control (descomenta para la versión final) ---
        satellite_indices = ['NDVI', 'NDII', 'GNDVI', 'EVI']
        terrain_indices = ['ELE', 'ASP', 'SLO']
        # satellite_indices = ['NDVI'] # Para pruebas
        # terrain_indices = ['ELE']   # Para pruebas

        stats_list = ['min', 'max', 'median', 'mean', 'stdDev']
        
        # Lista para guardar todas las bandas que queremos muestrear
        all_images_to_sample = []

        # --- Determinar Año ---
        try:
            # year = start_year
            
            # print(f"Processing year: {year}", tag='GEE', tag_color='blue', color='white') if verbose else None
            # satellite_dataset_name = select_dataset(year)

            # # --- 1. Preparar Puntos de Muestreo ---
            # # (Esto es lo mismo que tu código original)
            # points = []
            # for index, row in self.coordinates.iterrows():
            #     lat = row['latitude']
            #     lon = row['longitude']
            #     parcela_id = row['parcela_id']
            #     point = ee.Feature(ee.Geometry.Point([lon, lat]), {'parcela_id': parcela_id, 'ano': int(year)})
            #     points.append(point)
            
            # # Crear la FeatureCollection UNA VEZ
            # points_fc = ee.FeatureCollection(points)

            # # --- 2. Preparar Capas de Terreno (DEM) ---
            # if terrain_indices:
            #     print("Preparing terrain layers (ELE, SLO, ASP)...", tag='GEE', tag_color='blue', color='white') if verbose else None
            #     terrain_image = get_terrain_image(spain).select(terrain_indices)
            #     all_images_to_sample.append(terrain_image)

            # # --- 3. Preparar Capas de Vegetación (Estacional) ---
            # if satellite_indices:
            #     # Definir las temporadas y sus prefijos
            #     seasons_to_process = {
            #         'yearly': (f'{year}-01-01', f'{year}-12-31'),
            #         'fall': (f'{year}-09-21', f'{year}-12-21'),
            #         'winter': (f'{year-1}-12-21', f'{year}-03-21'),
            #         'spring': (f'{year}-03-21', f'{year}-06-21'),
            #         'summer': (f'{year}-06-21', f'{year}-09-21'),
            #     }
                
            #     # Descomenta las temporadas que necesites
            #     # (Para la prueba, solo 'yearly' como en tu código original)
            #     # active_seasons = ['yearly'] 
            #     active_seasons = ['yearly', 'fall', 'winter', 'spring', 'summer'] # Versión final

            #     # --- LÓGICA DEL CALLBACK (INICIO) ---
            #     total_seasons = len(active_seasons)
            #     current_season_step = 0
            #     # --- FIN LÓGICA ---

            #     for season_name, (s_date, e_date) in seasons_to_process.items():
                    
            #         if season_name not in active_seasons:
            #             continue

            #         # --- 1. LLAMADA AL CALLBACK ---
            #         if callback_function:
            #             # Escalamos el progreso de esta fase para que ocupe el primer 40%
            #             percentage = (current_season_step / total_seasons) * 40
            #             callback_function(percentage, f"Preparing {season_name} layers...")
                        
            #         print(f"Preparing {season_name} vegetation stats...", tag='GEE', tag_color='blue', color='white') if verbose else None
                    
            #         add_indices_to_collection = partial(add_all_indices, dataset_name=satellite_dataset_name)

            #         # Cargar, filtrar y mapear la colección base
            #         base_collection = ee.ImageCollection(satellite_dataset_name) \
            #             .filterDate(s_date, e_date) \
            #             .filterBounds(spain) \
            #             .map(apply_scale_factors) \
            #             .map(mask_clouds) \
            #             .map(add_indices_to_collection) # Añadir todos los índices a la vez
                    
            #         # Calcular todas las estadísticas para todos los índices
            #         stats_image = calculate_vegetation_stats(base_collection, satellite_indices)
                    
            #         # Renombrar las bandas para incluir el prefijo de la temporada
            #         # p.ej. 'NDVI_min' -> 'yearly_NDVI_min'
            #         def rename_band(b):
            #             return ee.String(season_name).cat('_').cat(b)
                    
            #         new_band_names = stats_image.bandNames().map(rename_band)
            #         stats_image_renamed = stats_image.rename(new_band_names)
                    
            #         all_images_to_sample.append(stats_image_renamed)
            #         current_season_step += 1

            # # --- 4. Combinar todas las imágenes en una sola ---
            # if not all_images_to_sample:
            #     print("No data to process. Check satellite_indices or terrain_indices.", tag='GEE', tag_color='red', color='white')
            #     return pd.DataFrame()

            # print("Combining all layers into a single image...", tag='GEE', tag_color='blue', color='white') if verbose else None
            # final_image = ee.Image.cat(all_images_to_sample)

            # if callback_function:
            #     # Asignamos un 10% del total a esta tarea
            #     callback_function(40, "Extracting all data from GEE (this may take a moment)...")
            
            # extracted_features = final_image.unmask(-999).reduceRegions(
            #     collection=points_fc,
            #     reducer=ee.Reducer.mean(), # Usamos mean para muestrear el píxel en el punto
            #     scale=30,
            #     tileScale=4
            # )
            
            # # --- 6. Convertir a DataFrame (una sola vez) ---
            # results_wide_df = geemap.ee_to_df(extracted_features)

            # print(results_wide_df)

            year = start_year
    
            if verbose:
                print(f"\n{'='*70}", tag='LANDSAT', tag_color='blue', color='white')
                print(f"🛰️  LANDSAT DATA EXTRACTION | Year: {year}", tag='LANDSAT', tag_color='blue', color='white', format='bold')
                print(f"📊 Dataset: {select_dataset_name(year)}", tag='LANDSAT', tag_color='blue', color='white')
                print(f"{'='*70}", tag='LANDSAT', tag_color='blue', color='white')
            satellite_dataset_name = select_dataset(year)

            # --- 1. Preparar Capas (Omitimos la creación de 'points_fc' aquí) ---
            all_images_to_sample = [] # Mover esto aquí

            # --- 2. Preparar Capas de Terreno (DEM) ---
            if terrain_indices:
                if verbose:
                    print(f"\n🏞️  Preparing TERRAIN layers (Elevation, Slope, Aspect)...", tag='LANDSAT', tag_color='blue', color='white')
                terrain_image = get_terrain_image(spain).select(terrain_indices)
                all_images_to_sample.append(terrain_image)

            # --- 3. Preparar Capas de Vegetación (Estacional) ---
            if satellite_indices:
                # Definir las temporadas y sus prefijos
                seasons_to_process = {
                    'yearly': (f'{year}-01-01', f'{year}-12-31'),
                    'fall': (f'{year}-09-21', f'{year}-12-21'),
                    'winter': (f'{year-1}-12-21', f'{year}-03-21'),
                    'spring': (f'{year}-03-21', f'{year}-06-21'),
                    'summer': (f'{year}-06-21', f'{year}-09-21'),
                }
                
                active_seasons = ['yearly', 'fall', 'winter', 'spring', 'summer'] 

                # --- LÓGICA DEL CALLBACK (INICIO) ---
                total_seasons = len(active_seasons)
                current_season_step = 0
                # --- FIN LÓGICA ---

                for season_name, (s_date, e_date) in seasons_to_process.items():
                    
                    if season_name not in active_seasons:
                        continue

                    # --- 1. LLAMADA AL CALLBACK ---
                    if callback_function:
                        # Escalamos el progreso de esta fase (preparación de imagen) 
                        # para que ocupe el primer 40%
                        percentage = (current_season_step / total_seasons) * 40
                        callback_function(percentage, f"Preparing {season_name} layers for {year}...")
                        
                    if verbose:
                        season_icons = {'yearly': '🌍', 'winter': '❄️', 'spring': '🌱', 'summer': '🌞', 'fall': '🍁'}
                        icon = season_icons.get(season_name, '🌿')
                        print(f"{icon} Computing {season_name.upper()} vegetation indices (NDVI, EVI, NDII, GNDVI)...", tag='LANDSAT', tag_color='blue', color='white')
                    
                    add_indices_to_collection = partial(add_all_indices, dataset_name=satellite_dataset_name)

                    # Cargar, filtrar y mapear la colección base
                    base_collection = ee.ImageCollection(satellite_dataset_name) \
                        .filterDate(s_date, e_date) \
                        .filterBounds(spain) \
                        .map(apply_scale_factors) \
                        .map(mask_clouds) \
                        .map(add_indices_to_collection)
                    
                    # Calcular todas las estadísticas para todos los índices
                    stats_image = calculate_vegetation_stats(base_collection, satellite_indices)
                    
                    # Renombrar las bandas
                    def rename_band(b):
                        return ee.String(season_name).cat('_').cat(b)
                    
                    new_band_names = stats_image.bandNames().map(rename_band)
                    stats_image_renamed = stats_image.rename(new_band_names)
                    
                    all_images_to_sample.append(stats_image_renamed)
                    current_season_step += 1

            # --- 4. Combinar todas las imágenes en una sola ---
            if not all_images_to_sample:
                print(f"No data to process for {year}. Check indices.", tag='GEE', tag_color='red', color='white')
                # Continuar al siguiente año si esto estuviera en un bucle de años
                return pd.DataFrame() 

            if verbose:
                print(f"\n🧩 Combining all spectral layers for {year}...", tag='LANDSAT', tag_color='blue', color='white')
            final_image = ee.Image.cat(all_images_to_sample)

            if callback_function:
                callback_function(40)

            # --- 5. y 6. EXTRAER DATOS POR LOTES Y CONVERTIR A DATAFRAME ---
            # (Esta es la lógica de tu segundo script, adaptada)
            
            # ¡DEBES DEFINIR batch_size en algún lugar! 
            # (p.ej., como parámetro de la función o self.batch_size)

            # Calculate number of batches
            num_points = len(self.coordinates)
            num_batches = int(np.ceil(num_points / batch_size))
            
            # List to store the results DataFrame from each batch
            results_list = []
            
            if verbose:
                print(f"\n📍 Processing {num_points} coordinates → {num_batches} batches (batch size: {batch_size})", tag='LANDSAT', tag_color='blue', color='white')
            
            # Usar tqdm para la barra de progreso
            for i in tqdm(range(num_batches), desc=f"Processing batches for {year}", disable=not verbose):
                # 5a. Obtener el lote (chunk) actual
                start_index = i * batch_size
                end_index = min((i + 1) * batch_size, num_points)
                df_batch = self.coordinates.iloc[start_index:end_index].copy() # .copy() para evitar warnings
                
                # 5b. Convertir este lote a ee.FeatureCollection
                # (Aquí replicamos la lógica de añadir el año que tenías antes)
                try:
                    df_batch['ano'] = int(year) # Añadir el año a los datos del batch
                    
                    # geemap transfiere todas las columnas del df (lat, lon, parcela_id, y la nueva 'ano')
                    points_fc_batch = geemap.pandas_to_ee(df_batch, latitude='latitude', longitude='longitude')
                    
                except Exception as e:
                    print(f"Error: Batch {i} failed. Ensure DataFrame has 'latitude' and 'longitude'.")
                    print(f"Detail: {e}")
                    continue # Saltar este lote y continuar con el siguiente

                # 5c. Extraer datos (reduceRegions) para este lote
                try:
                    extracted_data_fc = final_image.unmask(-999).reduceRegions(
                        collection=points_fc_batch,
                        reducer=ee.Reducer.mean(), 
                        scale=30, # ¡Importante! Usar la escala de 30m (o 10m) de las imágenes satelitales
                        tileScale=4 # tileScale alto (4, 8, 16) ayuda a evitar errores de memoria GEE
                    )
                except Exception as e:
                    print(f"Error: Batch {i} failed during GEE reduceRegions. Skipping.")
                    print(f"Detail: {e}")
                    continue

                # 5d. Convertir resultados del lote a DataFrame
                df_results_batch = geemap.ee_to_df(extracted_data_fc)
                
                # 5e. Añadir a la lista
                results_list.append(df_results_batch)
                
                # --- 2. LLAMADA AL CALLBACK ---
                if callback_function:
                    # Escalamos el progreso de esta fase (batching) del 40% al 100%
                    percentage = 40 + ((i + 1) / num_batches) * 60 # 60% restante del progreso
                    callback_function(int(percentage), f"Extracted batch {i+1}/{num_batches} for {year}...")


            if verbose:
                print(f"\n✅ Year {year} extraction complete! Merging {num_batches} batch results...", tag='LANDSAT', tag_color='green', color='white', format='bold')

            if not results_list:
                print(f"No results generated for {year}. All batches failed.", tag='GEE', tag_color='red', color='white')
                return pd.DataFrame()

            # Combinar todos los DataFrames de los lotes en uno solo
            results_wide_df = pd.concat(results_list, ignore_index=True)

            if callback_function:
                callback_function(100, f"Finished processing {year}!")

            if return_type == 'dict':
                # --- 4. LLAMADA AL CALLBACK ---
                if callback_function:
                    callback_function(50, "GEE data extracted. Formatting results...")
                # --- FIN CALLBACK ---
                
                # --- 7. REFORMATEAR A DICCIONARIO ANIDADO ---
                if verbose:
                    print(f"\n📋 Formatting data into nested structure...", tag='LANDSAT', tag_color='blue', color='white')

                # 7.1. Añadir latitud y longitud desde el DF original
                # coords_to_merge = self.coordinates[['parcela_id', 'latitude', 'longitude']].drop_duplicates(subset=['parcela_id'])
                # df_with_coords = pd.merge(results_wide_df, coords_to_merge, on='parcela_id', how='left')
                df_with_coords = results_wide_df.copy()

                # 7.2. Convertir de "ancho" a "largo"
                #id_vars = ['parcela_id', 'ano', 'latitude', 'longitude', 'inventario_id']
                id_vars = ['parcela_id', 'latitude', 'longitude', 'ano']
                value_vars = [col for col in df_with_coords.columns if col not in id_vars]
                
                if not value_vars:
                    print("Extraction returned no data columns.", tag='GEE', tag_color='red', color='white')
                    return {}

                long_df = df_with_coords.melt(
                    id_vars=id_vars,
                    value_vars=value_vars,
                    var_name='metric_full',
                    value_name='value'
                )

                # 7.3. Reemplazar -999 (unmasked) con NaN
                long_df['value'] = long_df['value'].replace(-999, np.nan)

                # 7.4. Parsear la columna 'metric_full' para obtener 'season', 'index', 'statistic'
                
                # Regex para vegetación: (season)_(index)_(statistic)
                regex_veg = r'^(yearly|fall|winter|spring|summer)_(' + '|'.join(satellite_indices) + ')_(' + '|'.join(stats_list) + ')$'
                parts = long_df['metric_full'].str.extract(regex_veg)
                parts.columns = ['season', 'index', 'statistic']
                parsed_df = pd.concat([long_df, parts], axis=1)

                # Manejar datos de terreno (que no coincidieron con el regex)
                is_terrain = parsed_df['season'].isnull() & parsed_df['metric_full'].isin(terrain_indices)
                parsed_df.loc[is_terrain, 'season'] = 'yearly' # Asignar terreno a la temporada 'yearly'
                parsed_df.loc[is_terrain, 'index'] = parsed_df.loc[is_terrain, 'metric_full']
                parsed_df.loc[is_terrain, 'statistic'] = 'value' # Usar 'value' como clave de estadística genérica

                # Manejar cualquier métrica no reconocida
                is_unparsed = parsed_df['season'].isnull() & ~is_terrain
                if is_unparsed.any():
                    unparsed_metrics = parsed_df.loc[is_unparsed, 'metric_full'].unique()
                    # print(f"Advertencia: Métricas no parseadas encontradas: {unparsed_metrics}", tag='GEE', tag_color='yellow', color='white')
                    parsed_df.loc[is_unparsed, 'season'] = 'yearly'
                    parsed_df.loc[is_unparsed, 'index'] = parsed_df.loc[is_unparsed, 'metric_full']
                    parsed_df.loc[is_unparsed, 'statistic'] = 'value'

                # 7.5. Construir el diccionario anidado final
                final_nested_dict = {}

                # --- LÓGICA DEL CALLBACK (INICIO) ---
                groups = parsed_df.groupby('parcela_id')
                total_parcels = len(groups)
                current_parcel_step = 0
                
                # Agrupar por 'parcela_id' para procesar cada parcela individualmente
                for parcela_id, group in parsed_df.groupby('parcela_id'):

                    # --- 5. LLAMADA AL CALLBACK ---
                    if callback_function:
                        # Escalamos esta fase para que ocupe del 50% al 100%
                        percentage = 50 + ((current_parcel_step / total_parcels) * 50)
                        callback_function(percentage, f"Formatting parcel {current_parcel_step+1}/{total_parcels}")
                    # --- FIN CALLBACK ---
                    
                    # Inicializar el diccionario para esta parcela
                    parcela_data = {}
                    
                    # Añadir metadatos (son los mismos para todas las filas de este grupo)
                    first_row = group.iloc[0]
                    parcela_data['metadata'] = {
                        'latitude': first_row['latitude'],
                        'longitude': first_row['longitude'],
                        'ano': int(start_year), #int(first_row['ano']),
                        'parcela_id': parcela_id,
                        'start_year': start_year, # Añadir fechas de la consulta
                        'dataset_name': satellite_dataset_name
                    }
                    
                    parcela_data['data'] = {} # Aquí irán los datos anidados

                    # Agrupar por 'season'
                    for season, season_group in group.groupby('season'):
                        parcela_data['data'][season] = {}

                        # Agrupar por 'index'
                        for index, index_group in season_group.groupby('index'):
                            
                            # Si la estadística es 'value' (datos de terreno), es un valor único
                            if (index_group['statistic'] == 'value').all():
                                parcela_data['data'][season][index] = float(index_group['value'].iloc[0])
                            else:
                                # Si no, es un índice de vegetación con múltiples estadísticas
                                # Crear el diccionario {estadística: valor}
                                stats_dict = index_group.set_index('statistic')['value'].to_dict()
                                parcela_data['data'][season][index] = stats_dict
                    
                    # Asignar el diccionario completo de la parcela a la clave 'parcela_id'
                    final_nested_dict[parcela_id] = parcela_data
                    salida_final = final_nested_dict

                    # --- 6. INCREMENTO DEL CALLBACK ---
                    current_parcel_step += 1
                    # --- FIN CALLBACK ---

                # --- 7. LLAMADA FINAL AL CALLBACK ---
                if callback_function:
                    callback_function(100, "Formatting complete.")
                # --- FIN CALLBACK ---

            elif return_type == 'dataframe':
                # 2. Definir las columnas que identifican cada observación
                #    (Ahora solo 'ano' y nuestro ID)
                id_vars = ['parcela_id', 'ano', 'ASP', 'ELE', 'SLO', 'latitude', 'longitude']
                # 3. "Derretir" (melt) el DataFrame
                df_long = results_wide_df.melt(
                    id_vars=id_vars,
                    var_name='variable_completa',  # 'fall_EVI_mean', etc.
                    value_name='valor'
                )

                # 4. Dividir la variable en 'estacion_id' y 'metrica'
                #    'fall_EVI_mean' -> ['fall', 'EVI_mean']
                #    'yearly_NDII_stdDev' -> ['yearly', 'NDII_stdDev']
                df_long[['estacion_id', 'metrica']] = df_long['variable_completa'].str.split('_', n=1, expand=True)

                # 5. Pivotar la tabla para obtener las columnas de métricas
                df_final = df_long.pivot(
                    index=id_vars + ['estacion_id'],
                    columns='metrica',
                    values='valor'
                )

                # 6. Limpiar el DataFrame resultante
                df_final = df_final.reset_index()      # Aplanar el índice
                df_final.columns.name = None           # Eliminar el nombre 'metrica' del índice

                df_final.columns = df_final.columns.str.lower().str.replace('stddev', 'std', regex=False)
                df_final.replace(-999, np.nan, inplace=True)
                salida_final = df_final.copy()
            else:
                raise ValueError(f"Invalid return_type: {return_type}")

            
            return salida_final


        except Exception as e:
            print(f"An error occurred: {e}", tag='GEE', tag_color='red', color='white')
            print(traceback.format_exc())
            print("----------------------------------------\n")
            return pd.DataFrame() # Devolver DF vacío en caso de error
