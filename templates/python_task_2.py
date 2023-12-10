import pandas as pd
import numpy as np
from datetime import time

# Load the dataset
df3 = pd.read_csv('../datasets/dataset-3.csv')

# Task 2: Question 1
def calculate_distance_matrix(df):
    distance_matrix = pd.pivot_table(df, values='distance', index='id_start', columns='id_end', aggfunc=np.sum, fill_value=0)
    distance_matrix = distance_matrix.add(distance_matrix.T, fill_value=0)
    np.fill_diagonal(distance_matrix.values, 0)
    return distance_matrix

# Task 2: Question 2
def unroll_distance_matrix(distance_matrix):
    unrolled_df = distance_matrix.unstack().reset_index(name='distance')
    unrolled_df = unrolled_df[unrolled_df['id_start'] != unrolled_df['id_end']]
    return unrolled_df

# Task 2: Question 3
def find_ids_within_ten_percentage_threshold(df, reference_value):
    avg_distance = df[df['id_start'] == reference_value]['distance'].mean()
    threshold_min = avg_distance * 0.9
    threshold_max = avg_distance * 1.1
    within_threshold_ids = df[(df['id_start'] != reference_value) & (df['distance'] >= threshold_min) & (df['distance'] <= threshold_max)]['id_start'].unique()
    return sorted(within_threshold_ids)

# Task 2: Question 4
def calculate_toll_rate(df):
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}
    
    for vehicle_type, coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * coefficient
    
    return df

# Task 2: Question 5
def calculate_time_based_toll_rates(df):
    weekday_discounts = [(time(0, 0, 0), time(10, 0, 0), 0.8),
                         (time(10, 0, 0), time(18, 0, 0), 1.2),
                         (time(18, 0, 0), time(23, 59, 59), 0.8)]
    
    weekend_discount = 0.7
    
    df['start_day'] = df['start_time'].dt.day_name()
    df['end_day'] = df['end_time'].dt.day_name()
    
    for _, discount_start, discount_end, discount_factor in weekday_discounts:
        mask = ((df['start_day'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])) & 
                (df['start_time'] >= discount_start) & (df['start_time'] <= discount_end))
        df.loc[mask, ['moto', 'car', 'rv', 'bus', 'truck']] *= discount_factor
    
    weekend_mask = df['start_day'].isin(['Saturday', 'Sunday'])
    df.loc[weekend_mask, ['moto', 'car', 'rv', 'bus', 'truck']] *= weekend_discount
    
    return df[['id_start', 'id_end', 'start_day', 'start_time', 'end_day', 'end_time', 'moto', 'car', 'rv', 'bus', 'truck']]

# Example usage:
distance_matrix = calculate_distance_matrix(df3)
print("Task 2: Question 1 Result:")
print(distance_matrix)

unrolled_df = unroll_distance_matrix(distance_matrix)
print("\nTask 2: Question 2 Result:")
print(unrolled_df)

reference_value = unrolled_df['id_start'].iloc[0]
within_threshold_ids = find_ids_within_ten_percentage_threshold(unrolled_df, reference_value)
print("\nTask 2: Question 3 Result:")
print(within_threshold_ids)

toll_rate_df = calculate_toll_rate(unrolled_df)
print("\nTask 2: Question 4 Result:")
print(toll_rate_df)

time_based_toll_rates_df = calculate_time_based_toll_rates(unrolled_df)
print("\nTask 2: Question 5 Result:")
print(time_based_toll_rates_df)
