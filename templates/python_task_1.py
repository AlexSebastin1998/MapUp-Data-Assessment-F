import pandas as pd
import numpy as np

# Task 1: Question 1
def generate_car_matrix(df):
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)
    np.fill_diagonal(car_matrix.values, 0)
    return car_matrix

# Task 1: Question 2
def get_type_count(df):
    df['car_type'] = pd.cut(df['car'], bins=[-np.inf, 15, 25, np.inf], labels=['low', 'medium', 'high'])
    type_count = df['car_type'].value_counts().to_dict()
    type_count = dict(sorted(type_count.items()))
    return type_count

# Task 1: Question 3
def get_bus_indexes(df):
    mean_bus = df['bus'].mean()
    bus_indexes = df[df['bus'] > 2 * mean_bus].index.tolist()
    return sorted(bus_indexes)

# Task 1: Question 4
def filter_routes(df):
    avg_truck_by_route = df.groupby('route')['truck'].mean()
    routes_filtered = avg_truck_by_route[avg_truck_by_route > 7].index.tolist()
    return sorted(routes_filtered)

# Task 1: Question 5
def multiply_matrix(df):
    modified_matrix = df.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25).round(1)
    return modified_matrix

# Task 1: Question 6
def check_time_completeness(df):
    df['start_time'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end_time'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    df['day_of_week'] = df['start_time'].dt.day_name()
    
    completeness_check = df.groupby(['id', 'id_2']).apply(lambda group: check_timestamps(group)).reset_index(level=[0, 1], drop=True)
    
    return completeness_check

def check_timestamps(group):
    time_range = pd.date_range(start='00:00:00', end='23:59:59', freq='T')
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    expected_timestamps = pd.MultiIndex.from_product([days_of_week, time_range])
    actual_timestamps = pd.MultiIndex.from_arrays([group['day_of_week'], group['start_time'].dt.time])
    
    return not actual_timestamps.equals(expected_timestamps)

# Example usage:
df1 = pd.read_csv('../datasets/dataset-1.csv')
result_df1 = generate_car_matrix(df1)
print("Task 1: Question 1 Result:")
print(result_df1)

result_dict2 = get_type_count(df1)
print("\nTask 1: Question 2 Result:")
print(result_dict2)

result_list3 = get_bus_indexes(df1)
print("\nTask 1: Question 3 Result:")
print(result_list3)

result_list4 = filter_routes(df1)
print("\nTask 1: Question 4 Result:")
print(result_list4)

result_df5 = multiply_matrix(result_df1)
print("\nTask 1: Question 5 Result:")
print(result_df5)

df2 = pd.read_csv('../datasets/dataset-2.csv')
result_bool_series6 = check_time_completeness(df2)
print("\nTask 1: Question 6 Result:")
print(result_bool_series6)
