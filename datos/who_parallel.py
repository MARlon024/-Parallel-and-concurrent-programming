import multiprocessing as mp
import pandas as pd
import os
from functools import partial

def calculate_most_polluted_cities(df, region):
    # Seleccionamos los datos de la región
    region_data = df[df['who_region'] == region]

    # Calculamos la media de la concentración de PM10 por ciudad
    city_means = region_data.groupby(['country_name', 'city'])['pm10_concentration'].mean()

    # Ordenamos las ciudades por la media de la concentración de PM10 en orden descendente
    sorted_cities = city_means.sort_values(ascending=False)

    # Seleccionamos las 3 ciudades con la mayor media de concentración de PM10
    most_polluted_cities = sorted_cities.head(3)

    # Convert the Series to a list of dictionaries and return it
    return most_polluted_cities.reset_index().to_dict('records')



def main():
    archivo_original = 'who_ambient.csv'
    archivo_nuevo = 'who_new_country.csv'

    if not os.path.exists(archivo_nuevo):
        df = pd.read_csv(archivo_original, encoding='ISO-8859-1')
        columnas = ['who_region','country_name', 'city', 'year', 'pm10_concentration', 'pm25_concentration', 'no2_concentration', 'pm10_tempcov', 'pm25_tempcov', 'no2_tempcov', 'latitude', 'longitude']
        df = df[columnas]
        df.to_csv(archivo_nuevo, index=False)

    df = pd.read_csv(archivo_nuevo)

    regions = {
        '1_Afr': 'African region',
        '2_Amr': 'Region of the Americas',
        '3_Sear': 'South-East Asian region',
        '4_Eur': 'European region',
        '5_Emr': 'Eastern Mediterranean region',
        '6_Wpr': 'Western Pacific region',
        '7_NonMS': 'Non-member state'
    }

    results = {}

    # Creamos una nueva función que tiene df como un argumento preestablecido
    func = partial(calculate_most_polluted_cities, df)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results_list = pool.map(func, regions.keys())

    # Convert each list of dictionaries in results_list into a DataFrame and concatenate them
    df_results = pd.concat([pd.DataFrame(result) for result in results_list])

    # Add a 'who_region' column to df_results
    df_results['who_region'] = df_results['country_name'].map(regions)

    return df_results  # Return the DataFrame instead of printing it

if __name__ == '__main__':
    df_results = main()
    print(df_results)