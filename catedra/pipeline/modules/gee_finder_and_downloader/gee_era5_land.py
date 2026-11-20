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
from sqlalchemy import create_engine, MetaData, Table
import time
from pprint import pprint

# Initialize Google Earth Engine
ee.Authenticate()
ee.Initialize(project=os.environ["GEE_PROJECT"])  # Google Cloud project with Earth Engine enabled

spain_boundary = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017") \
                   .filter(ee.Filter.eq('country_co', 'SP'))

class GeeDownloadEra5Data():

    def __init__(self, 
                 coordinates):
        
        self.coordinates = coordinates


    def add_precipitation_statistics(self, start_date, end_date, batch_size=500, verbose=False):
        """
        Calculates precipitation statistics (sum, mean, min, max, std)
        for a given year, season, and DataFrame of coordinates.
        
        Processes points in batches to display a progress bar.
        
        Args:
            self.coordinates (pd.DataFrame): DataFrame that MUST contain
                                    'latitude' and 'longitude' columns.
            year (int or str): The year for which to calculate statistics.
            season (str): The season to calculate. 
                        Options: 'invierno', 'primavera', 'verano', 'otoño', 'yearly'.
            batch_size (int): How many points to process in each call to GEE.
                            A smaller number gives a smoother progress bar
                            but might be slightly slower overall.
            
        Returns:
            pd.DataFrame: The original DataFrame with 5 new columns added
                        for the seasonal/annual precipitation statistics in mm.
        """
        
        if verbose:
            print(f"\n{'='*70}", tag='ERA5', tag_color='cyan', color='white')
            print(f"🌧️  PRECIPITATION STATISTICS | Period: {start_date} to {end_date}", tag='ERA5', tag_color='cyan', color='white', format='bold')
            print(f"{'='*70}", tag='ERA5', tag_color='cyan', color='white')
        
        # --- 2. Prepare the Multi-Band Image in GEE (One-time calculation) ---
        # (This section remains unchanged, it will use the dates from above)
        band = 'total_precipitation_sum'
        
        seasonal_dataset = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
                        .filterDate(f'{start_date}', f'{end_date}') \
                        .select(band)

        # Calculate temporal statistics
        img_sum = seasonal_dataset.sum()
        img_mean = seasonal_dataset.mean()
        img_min = seasonal_dataset.min()
        img_max = seasonal_dataset.max()
        img_std = seasonal_dataset.reduce(ee.Reducer.stdDev())

        # Define future column names (in mm)
        band_names = [
            f'pr_sum', 
            f'pr_mean', 
            f'pr_min', 
            f'pr_max', 
            f'pr_std'
        ]

        # Combine into a multi-band image, rename, and convert to mm
        combined_image = img_sum.rename(band_names[0]) \
                    .addBands(img_mean.rename(band_names[1])) \
                    .addBands(img_min.rename(band_names[2])) \
                    .addBands(img_max.rename(band_names[3])) \
                    .addBands(img_std.rename(band_names[4])) \
                    .multiply(1000) # Convert from meters to mm

        # --- 3. Process in Batches with Progress Bar ---
        # (This section remains unchanged)
        
        # Calculate number of batches
        num_points = len(self.coordinates)
        num_batches = int(np.ceil(num_points / batch_size))
        
        # List to store the results DataFrame from each batch
        results_list = []
        
        if verbose:
            print(f"📍 Processing {num_points} coordinates → {num_batches} batches (size: {batch_size})", tag='ERA5', tag_color='cyan', color='white')
        
        era5_scale = 11132 # Native ERA5-LAND scale
        
        # Use tqdm to create the progress bar
        for i in tqdm(range(num_batches), desc=f"Processing period from {start_date} to {end_date}", disable=not verbose):
            # 3a. Get the current batch (chunk) from the DataFrame
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, num_points)
            df_batch = self.coordinates.iloc[start_index:end_index]
            
            # 3b. Convert this batch to an ee.FeatureCollection
            try:
                points_fc_batch = geemap.pandas_to_ee(df_batch, latitude='latitude', longitude='longitude')
            except Exception as e:
                print(f"Error: Batch {i} failed. Ensure your DataFrame has 'latitude' and 'longitude'.")
                print(f"Detail: {e}")
                continue # Skip this batch and continue with the next

            # 3c. Extract data (reduceRegions) for this batch
            extracted_data_fc = combined_image.unmask(-999).reduceRegions(
                collection=points_fc_batch,
                reducer=ee.Reducer.mean(), 
                scale=era5_scale,
                tileScale=4 # High tileScale (4, 8, 16) helps avoid GEE memory errors
            )

            # 3d. Convert batch results to a DataFrame
            df_results_batch = geemap.ee_to_df(extracted_data_fc)
            
            # 3e. Add to our list
            results_list.append(df_results_batch)

        if verbose:
            print(f"✅ Extraction complete! Merging batch results...", tag='ERA5', tag_color='green', color='white', format='bold')
        
        # --- 4. Combine all results ---
        # (This section remains unchanged)
        if not results_list:
            print("No results were processed.")
            return self.coordinates # Return original if everything failed
            
        df_final_results = pd.concat(results_list)
        
        # --- 5. Clean up and Return ---
        # (This section remains unchanged)
        original_columns = list(self.coordinates.columns)
        final_columns = original_columns + band_names + ['estacion']
        
        present_columns = [col for col in final_columns if col in df_final_results.columns]
        df_final_results = df_final_results[present_columns].reset_index(drop=True)
        df_final_results['start_date'] = start_date
        # Reorder columns to match original + new,
        # and reset the index which gets jumbled by pd.concat
        return df_final_results

    def add_temperature_statistics(self, start_date, end_date, batch_size=500, verbose=False):
        """
        Calculates seasonal statistics (mean, min, max, std) for a
        predefined list of 14 daily temperature bands from ERA5-LAND.
        
        AUTOMATICALLY converts data from Kelvin to Celsius before stats.
        
        The output columns will be, for example:
        - 't2m_min_mean': The mean of all daily minimums in the season.
        - 't2m_min_min': The minimum of all daily minimums (abs. min).
        - 't2m_max_max': The maximum of all daily maximums (abs. max).
        
        Args:
            self.coordinates (pd.DataFrame): DataFrame that MUST contain
                                    'latitude' and 'longitude' columns.
            year (int or str): The year for which to calculate statistics.
            season (str): The season to calculate. 
                        Options: 'invierno', 'primavera', 'verano', 'otoño', 'yearly'.
            batch_size (int): How many points to process in each call to GEE.
            
        Returns:
            pd.DataFrame: The original DataFrame with 56 new columns added
                        (14 bands * 4 statistics) for the seasonal
                        statistics in Celsius (°C).
        """
        
        if verbose:
            print(f"\n{'='*70}", tag='ERA5', tag_color='yellow', color='white')
            print(f"🌡️  TEMPERATURE STATISTICS | 18 Bands (2m, Skin, Soil L1-4)", tag='ERA5', tag_color='yellow', color='white', format='bold')
            print(f"📅 Period: {start_date} → {end_date}", tag='ERA5', tag_color='yellow', color='white')
            print(f"{'='*70}", tag='ERA5', tag_color='yellow', color='white')

        
        # --- 2. Define Bands and Prefixes ---
        # (Band Name in GEE, Output Column Prefix)
        bands_to_process = [
            ('temperature_2m', 't2m'),
            ('temperature_2m_min', 't2m_min'),
            ('temperature_2m_max', 't2m_max'),
            ('skin_temperature', 'skt'),
            ('skin_temperature_min', 'skt_min'),
            ('skin_temperature_max', 'skt_max'),
            ('soil_temperature_level_1', 'stl1'),
            ('soil_temperature_level_1_min', 'stl1_min'),
            ('soil_temperature_level_1_max', 'stl1_max'),
            ('soil_temperature_level_2', 'stl2'),
            ('soil_temperature_level_2_min', 'stl2_min'),
            ('soil_temperature_level_2_max', 'stl2_max'),
            ('soil_temperature_level_3', 'stl3'),
            ('soil_temperature_level_3_min', 'stl3_min'),
            ('soil_temperature_level_3_max', 'stl3_max'),
            ('soil_temperature_level_4', 'stl4'),
            ('soil_temperature_level_4_min', 'stl4_min'),
            ('soil_temperature_level_4_max', 'stl4_max')
        ]
        
        # Get the list of GEE band names to select
        gee_band_names = [b[0] for b in bands_to_process]

        # --- 3. Prepare the Multi-Band Image in GEE ---

        # 3a. Load the raw seasonal dataset and select all 14 bands
        raw_dataset = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
                        .filterDate(f'{start_date}', f'{end_date}') \
                        .select(gee_band_names)

        # 3b. Apply unit conversion (Kelvin to Celsius) to all bands
        def convert_k_to_c(img):
            # Subtract 273.15 from all bands in the image
            return img.subtract(273.15).copyProperties(img, ['system:time_start'])
        
        converted_dataset = raw_dataset.map(convert_k_to_c)
            
        # 3c. Calculate temporal statistics (mean, min, max, std)
        # This is done once for all 14 bands efficiently
        img_mean = converted_dataset.mean()
        img_min = converted_dataset.min()
        img_max = converted_dataset.max()
        img_std = converted_dataset.reduce(ee.Reducer.stdDev())

        # 3d. Generate lists of new column names
        # (e.g., 't2m_min_mean', 't2m_min_min', ...)
        mean_names = [f'{b[1]}_mean' for b in bands_to_process]
        min_names = [f'{b[1]}_min' for b in bands_to_process]
        max_names = [f'{b[1]}_max' for b in bands_to_process]
        std_names = [f'{b[1]}_std' for b in bands_to_process]
        
        # GEE's stdDev reducer renames bands to 'band_stdDev'
        std_input_names = [f'{b}_stdDev' for b in gee_band_names]

        # 3e. Rename bands in the reduced images to our desired names
        img_mean = img_mean.rename(mean_names)
        img_min = img_min.rename(min_names)
        img_max = img_max.rename(max_names)
        img_std = img_std.select(std_input_names).rename(std_names) # Select and rename for std

        # 3f. Combine all 56 bands (14 * 4 stats) into one Image
        combined_image = ee.Image.cat([img_mean, img_min, img_max, img_std])
        
        # This is the master list of all new columns
        band_names = mean_names + min_names + max_names + std_names

        # --- 4. Process in Batches with Progress Bar ---
        
        num_points = len(self.coordinates)
        num_batches = int(np.ceil(num_points / batch_size))
        results_list = []
        
        if verbose:
            print(f"📍 Processing {num_points} coordinates → {num_batches} batches", tag='ERA5', tag_color='yellow', color='white')
        
        era5_scale = 11132 # Native ERA5-LAND scale
        
        for i in tqdm(range(num_batches), desc=f"Processing Temps from {start_date} to {end_date}", disable=not verbose):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, num_points)
            df_batch = self.coordinates.iloc[start_index:end_index]
            
            try:
                points_fc_batch = geemap.pandas_to_ee(df_batch, latitude='latitude', longitude='longitude')
            except Exception as e:
                print(f"Error: Batch {i} failed. Ensure 'latitude'/'longitude'.")
                print(f"Detail: {e}")
                continue 

            # Extract all 56 values at once
            extracted_data_fc = combined_image.unmask(-999).reduceRegions(
                collection=points_fc_batch,
                reducer=ee.Reducer.mean(), 
                scale=era5_scale,
                tileScale=4 
            )

            df_results_batch = geemap.ee_to_df(extracted_data_fc)
            results_list.append(df_results_batch)

        if verbose:
            print(f"✅ Temperature extraction complete! Merging results...", tag='ERA5', tag_color='green', color='white', format='bold')
        
        # --- 5. Combine all results ---
        if not results_list:
            print("No results were processed.")
            return self.coordinates
            
        df_final_results = pd.concat(results_list)
        
        # --- 6. Clean up and Return ---
        original_columns = list(self.coordinates.columns)
        # 'band_names' now contains all 56 new column names
        all_columns = original_columns + band_names
        
        present_columns = [col for col in all_columns if col in df_final_results.columns]
        
        df_clean = df_final_results[present_columns].reset_index(drop=True)
        df_clean['start_date'] = start_date

        return df_clean
    

    def get_data(self, start_year, batch_size=500, verbose=False, progress_callback=None, return_type='dict'):
        """
        Obtiene datos de precipitación y temperatura para un año dado Y
        todas sus estaciones (invierno, primavera, verano, otoño).
        Formatea todo en un único diccionario anidado por parcela.
        
        La estructura final es:
        parcela_id -> metadata
        parcela_id -> data -> 'yearly'  -> index -> statistic
                           -> 'winter'  -> index -> statistic
                           -> 'spring'  -> index -> statistic
                           -> 'summer'  -> index -> statistic
                           -> 'fall'    -> index -> statistic
        """

        
        # 1. Definir todos los períodos a procesar
        # Nota: El invierno 'winter' de {year} abarca desde {year-1} a {year}
        seasons = {
            'yearly': (f'{start_year}-01-01', f'{start_year}-12-31'),
            'winter': (f'{start_year-1}-12-21', f'{start_year}-03-20'),
            'spring': (f'{start_year}-03-21', f'{start_year}-06-20'),
            'summer': (f'{start_year}-06-21', f'{start_year}-09-20'),
            'fall': (f'{start_year}-09-21', f'{start_year}-12-20')
        }
        
        

        total_steps = len(seasons)
        current_step = 0

        df_final = pd.DataFrame()

        # 3. Iterar sobre cada período, obtener datos y poblar el diccionario
        for season_name, (start_date, end_date) in seasons.items():
            current_step += 1
            
            if verbose:
                season_icons = {'yearly': '🌍', 'winter': '❄️', 'spring': '🌸', 'summer': '☀️', 'fall': '🍂'}
                icon = season_icons.get(season_name, '📊')
                print(f"\n{'─'*70}", tag='ERA5', tag_color='magenta', color='white')
                print(f"{icon} {season_name.upper()} SEASON | {start_date} → {end_date}", tag='ERA5', tag_color='magenta', color='white', format='bold')
                print(f"{'─'*70}", tag='ERA5', tag_color='magenta', color='white')

            # 3.1. Obtener datos de precipitación para este período
            df_precip = self.add_precipitation_statistics(start_date, end_date, batch_size=batch_size)
            df_precip['estacion_id'] = season_name
            df_precip['start_date'] = start_date
            df_precip['end_date'] = end_date
            
            if df_precip.empty:
                print(f"Skipping {season_name}: Error al obtener datos de precipitación.", tag='GEE', tag_color='red', color='white')
                continue

            # 3.2. Obtener datos de temperatura para este período
            df_temp = self.add_temperature_statistics(start_date, end_date, batch_size=batch_size)
            # df_temp['estacion_id'] = season_name
            if df_temp.empty:
                print(f"Skipping {season_name}: Error al obtener datos de temperatura.", tag='GEE', tag_color='red', color='white')
                continue

            # 3.3. Combinar los DataFrames del período actual
            if verbose:
                print(f"🔄 Merging precipitation & temperature data for {season_name}...", tag='ERA5', tag_color='magenta', color='white')
            metadata_cols = list(self.coordinates.columns) + ['start_date']
            data_cols_temp = [col for col in df_temp.columns if col not in metadata_cols and col != 'parcela_id']
            
            merged_df = pd.merge(
                df_precip,
                df_temp[['parcela_id'] + data_cols_temp],
                on='parcela_id',
                how='left'
            )

            df_final = pd.concat([df_final, merged_df], ignore_index=True)
            df_final['ano'] = start_year
            


        if return_type == 'dict':
            # 2. Inicializar el diccionario final que contendrá todo
            final_nested_dict = {}

            # 3.4. Convertir de "ancho" a "largo"
            id_vars = [col for col in metadata_cols if col in merged_df.columns]+ ['estacion_id', 'end_date']
            value_vars = [col for col in merged_df.columns if col not in id_vars] 
            
            if not value_vars:
                print(f"Skipping {season_name}: No data columns found after merge.", tag='GEE', tag_color='yellow', color='white')
                

            long_df = merged_df.melt(
                id_vars=id_vars,
                value_vars=value_vars,
                var_name='metric_full',
                value_name='value'
            )

            # 3.5. Parsear 'metric_full' en 'index' y 'statistic'
            long_df['value'] = long_df['value'].replace(-999, np.nan)
            parts = long_df['metric_full'].str.rsplit('_', n=1, expand=True)
            parts.columns = ['index', 'statistic']
            parsed_df = pd.concat([long_df, parts], axis=1)

            # 3.6. Construir/Actualizar el diccionario anidado final
            for parcela_id, group in parsed_df.groupby('parcela_id'):
                
                # --- Inicializar la parcela (metadata) si es la primera vez que la vemos ---
                if parcela_id not in final_nested_dict:
                    first_row = group.iloc[0]
                    final_nested_dict[parcela_id] = {
                        'metadata': {
                            'latitude': float(first_row['latitude']),
                            'longitude': float(first_row['longitude']),
                            'parcela_id': parcela_id,
                            'year': start_year  # Usamos 'year' en lugar de start/end date
                        },
                        'data': {}  # Inicializar el contenedor de datos
                    }
                    # Añadir cualquier otra columna de metadatos del CSV original
                    for meta_col in self.coordinates.columns:
                        if meta_col not in ['latitude', 'longitude', 'parcela_id']:
                            final_nested_dict[parcela_id]['metadata'][meta_col] = first_row[meta_col]

                # --- Poblar los datos para la ESTACIÓN (season_name) actual ---
                season_data = {}
                for index, index_group in group.groupby('index'):
                    stats_dict = {
                        row['statistic']: float(row['value']) for _, row in index_group.iterrows() if pd.notna(row['value'])
                    }
                    if stats_dict:
                        season_data[index] = stats_dict
                
                # Añadir el diccionario de esta estación a los datos de la parcela
                if season_data:
                    final_nested_dict[parcela_id]['data'][season_name] = season_data

            # 2. ----- INFORMA DEL PROGRESO -----
            # Calcula el porcentaje completado
            percentage = ((current_step) / total_steps) * 100
            
            # Si la función "callback" existe, llámala con el porcentaje
            if progress_callback:
                progress_callback(percentage)
        
            if verbose:
                print(f"\n{'='*70}", tag='ERA5', tag_color='green', color='white')
                print(f"🎉 ALL SEASONS COMPLETED SUCCESSFULLY!", tag='ERA5', tag_color='green', color='white', format='bold')
                print(f"{'='*70}\n", tag='ERA5', tag_color='green', color='white')

            if progress_callback:
                progress_callback(100)

            salida_final = final_nested_dict

              
        elif return_type == 'dataframe':
            salida_final = df_final
        else:
            raise ValueError(f"Invalid return_type: {return_type}")

        
        return salida_final # df_final
