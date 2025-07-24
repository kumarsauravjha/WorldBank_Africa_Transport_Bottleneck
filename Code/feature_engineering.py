#%%
import pandas as pd
import json
import pycountry
import pycountry_convert as pc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
# %%
df = pd.read_csv("imputed_full_matrix_at_centroid.csv")
# %%
df.shape
# %%
# Source: CIA World Factbook & UN geodata
landlocked_iso = [
    'AFG', 'AND', 'ARM', 'AUT', 'AZE', 'BOL', 'BWA', 'BFA', 'BDI', 'CAF', 'CHE',
    'CZE', 'ETH', 'HUN', 'KAZ', 'KGZ', 'LAO', 'LSO', 'LIE', 'LUX', 'MKD', 'MLI',
    'MNE', 'MWI', 'NER', 'NPL', 'RWA', 'SMR', 'SSD', 'SWZ', 'TCD', 'TJK', 'TKM',
    'UGA', 'UZB', 'VAT', 'ZMB', 'ZWE', 'SRB'
]

island_iso = [
    'AUS', 'JPN', 'ISL', 'NZL', 'MDV', 'MUS', 'FJI', 'VUT', 'TWN', 'SGP', 'CYP',
    'PHL', 'IDN', 'SYC', 'COM', 'CPV', 'GRD', 'LCA', 'KIR', 'NRU', 'PLW', 'SLB',
    'STP', 'TON', 'WSM', 'TUV', 'BRB', 'BHS', 'ATG', 'DMA', 'MTQ', 'CYM', 'MHL',
    'NIU', 'NCL', 'NFK', 'REU', 'GUM', 'MYT', 'GLP', 'COK', 'ASM', 'VIR', 'VGB',
    'SXM', 'WLF'
]

#%%
# Classification function
def classify_origin(iso):
    if iso in landlocked_iso:
        return 'landlocked'
    elif iso in island_iso:
        return 'island'
    else:
        return 'coastal'

df['origin_type'] = df['origin_ISO'].apply(classify_origin)

#%%
df['destination_type'] = df['destination_ISO'].apply(classify_origin)
# %%
df['origin_type'].value_counts()
# %%
df.groupby('origin_type')['origin_ISO'].nunique()

#%% CONTINENT CLASSIFICATION
manual_continent_map = {
    'TLS': 'AS',
    'SXM': 'NA',
    'ESH': 'AF'
}

# alpha-3 to alpha-2
iso3_to_iso2_map = {c.alpha_3: c.alpha_2 for c in pycountry.countries}

def iso3_to_continent(iso3):
    if iso3 in manual_continent_map:
        return manual_continent_map[iso3]
    
    iso2 = iso3_to_iso2_map.get(iso3.upper(), None)
    try:
        return pc.country_alpha2_to_continent_code(iso2)
    except:
        return None

df['origin_continent'] = df['origin_ISO'].apply(iso3_to_continent)
df['destination_continent'] = df['destination_ISO'].apply(iso3_to_continent)

#%%
df[df['origin_continent'].isna()]['origin_ISO'].unique()


# %%
df['origin_continent'].value_counts()
# %%
df['destination_continent'].value_counts()

#%% TRADE TYPE

df['trade_type'] = df.apply(
    lambda row: 'continental' if row['origin_continent'] == row['destination_continent'] else 'overseas',
    axis=1
)
#%%
#interregional data creation
# Define region mapping
iso_to_region = {
    # Africa
    'DZA': 'North Africa', 'EGY': 'North Africa', 'MAR': 'North Africa', 'TUN': 'North Africa', 'LBY': 'North Africa', 'SDN': 'North Africa',
    'NGA': 'West Africa', 'GHA': 'West Africa', 'CIV': 'West Africa', 'SEN': 'West Africa', 'MLI': 'West Africa', 'BFA': 'West Africa',
    'BEN': 'West Africa', 'TGO': 'West Africa', 'GNB': 'West Africa', 'GMB': 'West Africa', 'NER': 'West Africa', 'LBR': 'West Africa',
    'SLE': 'West Africa', 'CPV': 'West Africa',
    'CMR': 'Central Africa', 'COD': 'Central Africa', 'CAF': 'Central Africa', 'GNQ': 'Central Africa', 'TCD': 'Central Africa',
    'GAB': 'Central Africa', 'COG': 'Central Africa', 'STP': 'Central Africa',
    'ETH': 'East Africa', 'KEN': 'East Africa', 'UGA': 'East Africa', 'RWA': 'East Africa', 'BDI': 'East Africa',
    'SOM': 'East Africa', 'TZA': 'East Africa', 'SSD': 'East Africa', 'ERI': 'East Africa', 'DJI': 'East Africa',
    'ZAF': 'Southern Africa', 'BWA': 'Southern Africa', 'NAM': 'Southern Africa', 'MOZ': 'Southern Africa',
    'ZMB': 'Southern Africa', 'ZWE': 'Southern Africa', 'SWZ': 'Southern Africa', 'LSO': 'Southern Africa',
    'AGO': 'Southern Africa', 'MWI': 'Southern Africa',

    # Asia
    'CHN': 'East Asia', 'JPN': 'East Asia', 'KOR': 'East Asia', 'MNG': 'East Asia', 'TWN': 'East Asia', 'HKG': 'East Asia', 'MAC': 'East Asia',
    'IDN': 'Southeast Asia', 'MYS': 'Southeast Asia', 'SGP': 'Southeast Asia', 'THA': 'Southeast Asia', 'VNM': 'Southeast Asia',
    'PHL': 'Southeast Asia', 'KHM': 'Southeast Asia', 'LAO': 'Southeast Asia', 'MMR': 'Southeast Asia', 'BRN': 'Southeast Asia',
    'TLS': 'Southeast Asia',
    'IND': 'South Asia', 'PAK': 'South Asia', 'BGD': 'South Asia', 'LKA': 'South Asia', 'NPL': 'South Asia',
    'BTN': 'South Asia', 'MDV': 'South Asia', 'AFG': 'South Asia',
    'KAZ': 'Central Asia', 'UZB': 'Central Asia', 'TJK': 'Central Asia', 'KGZ': 'Central Asia', 'TKM': 'Central Asia',
    'TUR': 'West Asia', 'IRN': 'West Asia', 'IRQ': 'West Asia', 'SAU': 'West Asia', 'ARE': 'West Asia', 'QAT': 'West Asia',
    'OMN': 'West Asia', 'BHR': 'West Asia', 'KWT': 'West Asia', 'YEM': 'West Asia', 'SYR': 'West Asia', 'JOR': 'West Asia',
    'LBN': 'West Asia', 'PSE': 'West Asia', 'ISR': 'West Asia',

    # Europe
    'DEU': 'Western Europe', 'FRA': 'Western Europe', 'NLD': 'Western Europe', 'BEL': 'Western Europe',
    'LUX': 'Western Europe', 'CHE': 'Western Europe', 'AUT': 'Western Europe',
    'GBR': 'Northern Europe', 'IRL': 'Northern Europe', 'DNK': 'Northern Europe', 'FIN': 'Northern Europe',
    'SWE': 'Northern Europe', 'NOR': 'Northern Europe', 'ISL': 'Northern Europe',
    'EST': 'Northern Europe', 'LVA': 'Northern Europe', 'LTU': 'Northern Europe',
    'ITA': 'Southern Europe', 'ESP': 'Southern Europe', 'PRT': 'Southern Europe', 'GRC': 'Southern Europe',
    'SVN': 'Southern Europe', 'HRV': 'Southern Europe', 'MKD': 'Southern Europe', 'MLT': 'Southern Europe',
    'ALB': 'Southern Europe', 'BIH': 'Southern Europe', 'SRB': 'Southern Europe', 'MNE': 'Southern Europe',
    'POL': 'Eastern Europe', 'CZE': 'Eastern Europe', 'SVK': 'Eastern Europe', 'HUN': 'Eastern Europe',
    'ROU': 'Eastern Europe', 'BGR': 'Eastern Europe', 'BLR': 'Eastern Europe', 'UKR': 'Eastern Europe',
    'MDA': 'Eastern Europe', 'RUS': 'Eastern Europe'
}

#%%

iso_to_region.update({
    'AND': 'Southern Europe',
    'FRO': 'Northern Europe',
    'GIB': 'Southern Europe',
    'SMR': 'Southern Europe'
})

#%%
iso_to_region.update({
    # AF corrections
    'GIN': 'West Africa',
    'MDG': 'Southern Africa',
    'MUS': 'Southern Africa',
    'COM': 'East Africa',
    'SYC': 'East Africa',
    'MRT': 'West Africa',
    'SHN': 'Southern Africa',
    'ESH': 'North Africa',

    # AS corrections
    'AZE': 'West Asia',
    'ARM': 'West Asia',
    'GEO': 'West Asia',
    'CYP': 'West Asia',
    'CXR': 'Southeast Asia',
    'PRK': 'East Asia'
})

#%%
# Apply mapping
df['origin_region'] = df['origin_ISO'].map(iso_to_region)
df['origin_region'] = df['origin_region'].fillna(df['origin_continent'])

#%%
df['destination_region'] = df['destination_ISO'].map(iso_to_region)
df['destination_region'] = df['destination_region'].fillna(df['destination_continent'])
#%%
df.isna().sum()

#%%
df[(df['origin_continent'] == 'AF') & (df['trade_type'] == 'overseas')
   & (df['destination_ISO'] == 'KOR')]['Mode_name'].value_counts()
# %%
# df.to_csv("wb_data_all_fields.csv", index=False)

# #%%
# df.to_csv("wb_data_all_fields_updated.csv", index=False)

#%%
df[df['origin_continent'] == 'OC']['origin_ISO'].unique()

# %%
# Define function to assign range
def assign_distance_range(km):
    if km <= 500:
        return '0-500'
    elif km <= 1000:
        return '500-1k'
    elif km <= 5000:
        return '1k-5k'
    elif km <= 10000:
        return '5k-10k'
    elif km <= 50000:
        return '10k-50k'
    elif km <= 100000:
        return '50k-100k'
    else:
        return '100k+'

# Apply function to create new column
df['Distance Range (km)'] = df['distance(km)'].apply(assign_distance_range)

#%%
# df.to_csv("wb_data_all_fields_updated_dist_range.csv", index=False)
# #%%
# df = pd.read_csv("wb_data_all_fields_updated_dist_range.csv")

#%%
# cols_to_fix = ['origin_continent', 'destination_continent', 'origin_region', 'destination_region']
# df[cols_to_fix] = df[cols_to_fix].astype(str)
# df.to_csv("D:/STUDY/MS/world bank projects/after grad/WorldBank_Africa_Transport_Bottleneck/Data/wb_data_all_fields.csv", index=False)
#%%
# df_dist = pd.read_csv("D:/STUDY/MS/world bank projects/after grad/WorldBank_Africa_Transport_Bottleneck/Data/wb_data_all_fields.csv", keep_default_na=False)
# %%
df2 = pd.read_csv("wb_data_all_fields_with_income_group.csv")
# %%
merged_df = df.merge(df2[['income_group']], left_index=True, right_index=True)

# %%
# merged_df['origin_continent'] = merged_df['origin_continent'].fillna('NA')
# merged_df['destination_continent'] = merged_df['destination_continent'].fillna('NA')
# merged_df['origin_region'] = merged_df['origin_region'].fillna('NA')
# merged_df['destination_region'] = merged_df['destination_region'].fillna('NA')
# %%
merged_df.to_csv("D:/STUDY/MS/world bank projects/after grad/WorldBank_Africa_Transport_Bottleneck/Data/wb_data_all_fields.csv", index=False)
# %%
