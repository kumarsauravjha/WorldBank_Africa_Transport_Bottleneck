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
df = pd.read_csv("../Data/imputed_full_matrix_at_centroid.csv")
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

#%%
# Full ISO3 to income group mapping
income_map = {
    'AFG': 'Low income', 'ALB': 'Upper middle income', 'DZA': 'Upper middle income', 'ASM': 'High income',
    'AND': 'High income', 'AGO': 'Lower middle income', 'ATG': 'High income', 'ARG': 'Upper middle income',
    'ARM': 'Upper middle income', 'ABW': 'High income', 'AUS': 'High income', 'AUT': 'High income',
    'AZE': 'Upper middle income', 'BHS': 'High income', 'BHR': 'High income', 'BGD': 'Lower middle income',
    'BRB': 'High income', 'BLR': 'Upper middle income', 'BEL': 'High income', 'BLZ': 'Upper middle income',
    'BEN': 'Lower middle income', 'BMU': 'High income', 'BTN': 'Lower middle income', 'BOL': 'Lower middle income',
    'BIH': 'Upper middle income', 'BWA': 'Upper middle income', 'BRA': 'Upper middle income', 'VGB': 'High income',
    'BRN': 'High income', 'BGR': 'High income', 'BFA': 'Low income', 'BDI': 'Low income', 'CPV': 'Lower middle income',
    'KHM': 'Lower middle income', 'CMR': 'Lower middle income', 'CAN': 'High income', 'CYM': 'High income',
    'CAF': 'Low income', 'TCD': 'Low income', 'CHI': 'High income', 'CHL': 'High income', 'CHN': 'Upper middle income',
    'COL': 'Upper middle income', 'COM': 'Lower middle income', 'COD': 'Low income', 'COG': 'Lower middle income',
    'CRI': 'Upper middle income', 'CIV': 'Lower middle income', 'HRV': 'High income', 'CUB': 'Upper middle income',
    'CUW': 'High income', 'CYP': 'High income', 'CZE': 'High income', 'DNK': 'High income', 'DJI': 'Lower middle income',
    'DMA': 'Upper middle income', 'DOM': 'Upper middle income', 'ECU': 'Upper middle income', 'EGY': 'Lower middle income',
    'SLV': 'Upper middle income', 'GNQ': 'Upper middle income', 'ERI': 'Low income', 'EST': 'High income',
    'SWZ': 'Lower middle income', 'ETH': 'Low income', 'FRO': 'High income', 'FJI': 'Upper middle income',
    'FIN': 'High income', 'FRA': 'High income', 'PYF': 'High income', 'GAB': 'Upper middle income',
    'GMB': 'Low income', 'GEO': 'Upper middle income', 'DEU': 'High income', 'GHA': 'Lower middle income',
    'GIB': 'High income', 'GRC': 'High income', 'GRL': 'High income', 'GRD': 'Upper middle income',
    'GUM': 'High income', 'GTM': 'Upper middle income', 'GIN': 'Lower middle income', 'GNB': 'Low income',
    'GUY': 'High income', 'HTI': 'Lower middle income', 'HND': 'Lower middle income', 'HKG': 'High income',
    'HUN': 'High income', 'ISL': 'High income', 'IND': 'Lower middle income', 'IDN': 'Upper middle income',
    'IRN': 'Upper middle income', 'IRQ': 'Upper middle income', 'IRL': 'High income', 'IMN': 'High income',
    'ISR': 'High income', 'ITA': 'High income', 'JAM': 'Upper middle income', 'JPN': 'High income',
    'JOR': 'Lower middle income', 'KAZ': 'Upper middle income', 'KEN': 'Lower middle income', 'KIR': 'Lower middle income',
    'PRK': 'Low income', 'KOR': 'High income', 'XKX': 'Upper middle income', 'KWT': 'High income',
    'KGZ': 'Lower middle income', 'LAO': 'Lower middle income', 'LVA': 'High income', 'LBN': 'Lower middle income',
    'LSO': 'Lower middle income', 'LBR': 'Low income', 'LBY': 'Upper middle income', 'LIE': 'High income',
    'LTU': 'High income', 'LUX': 'High income', 'MAC': 'High income', 'MDG': 'Low income', 'MWI': 'Low income',
    'MYS': 'Upper middle income', 'MDV': 'Upper middle income', 'MLI': 'Low income', 'MLT': 'High income',
    'MHL': 'Upper middle income', 'MRT': 'Lower middle income', 'MUS': 'Upper middle income', 'MEX': 'Upper middle income',
    'FSM': 'Lower middle income', 'MDA': 'Upper middle income', 'MCO': 'High income', 'MNG': 'Upper middle income',
    'MNE': 'Upper middle income', 'MAR': 'Lower middle income', 'MOZ': 'Low income', 'MMR': 'Lower middle income',
    'NAM': 'Upper middle income', 'NRU': 'High income', 'NPL': 'Lower middle income', 'NLD': 'High income',
    'NCL': 'High income', 'NZL': 'High income', 'NIC': 'Lower middle income', 'NER': 'Low income',
    'NGA': 'Lower middle income', 'MKD': 'Upper middle income', 'MNP': 'High income', 'NOR': 'High income',
    'OMN': 'High income', 'PAK': 'Lower middle income', 'PLW': 'High income', 'PAN': 'High income',
    'PNG': 'Lower middle income', 'PRY': 'Upper middle income', 'PER': 'Upper middle income',
    'PHL': 'Lower middle income', 'POL': 'High income', 'PRT': 'High income', 'PRI': 'High income',
    'QAT': 'High income', 'ROU': 'High income', 'RUS': 'High income', 'RWA': 'Low income',
    'WSM': 'Lower middle income', 'SMR': 'High income', 'STP': 'Lower middle income', 'SAU': 'High income',
    'SEN': 'Lower middle income', 'SRB': 'Upper middle income', 'SYC': 'High income', 'SLE': 'Low income',
    'SGP': 'High income', 'SXM': 'High income', 'SVK': 'High income', 'SVN': 'High income', 'SLB': 'Lower middle income',
    'SOM': 'Low income', 'ZAF': 'Upper middle income', 'SSD': 'Low income', 'ESP': 'High income',
    'LKA': 'Lower middle income', 'KNA': 'High income', 'LCA': 'Upper middle income', 'MAF': 'High income',
    'VCT': 'Upper middle income', 'SDN': 'Low income', 'SUR': 'Upper middle income', 'SWE': 'High income',
    'CHE': 'High income', 'SYR': 'Low income', 'TWN': 'High income', 'TJK': 'Lower middle income',
    'TZA': 'Lower middle income', 'THA': 'Upper middle income', 'TLS': 'Lower middle income',
    'TGO': 'Low income', 'TON': 'Upper middle income', 'TTO': 'High income', 'TUN': 'Lower middle income',
    'TUR': 'Upper middle income', 'TKM': 'Upper middle income', 'TCA': 'High income', 'TUV': 'Upper middle income',
    'UGA': 'Low income', 'UKR': 'Upper middle income', 'ARE': 'High income', 'GBR': 'High income',
    'USA': 'High income', 'URY': 'High income', 'UZB': 'Lower middle income', 'VUT': 'Lower middle income',
    'VNM': 'Lower middle income', 'VIR': 'High income', 'PSE': 'Lower middle income', 'YEM': 'Low income',
    'ZMB': 'Lower middle income', 'ZWE': 'Lower middle income'
}

income_map.update({
    'VEN': 'Upper middle income',
    'AIA': 'High income',
    'MSR': 'High income',
    'NIU': 'High income',
    'GLP': 'High income',
    'COK': 'High income',
    'BES': 'High income',
    'CXR': 'High income',
    'FLK': 'High income',
    'NFK': 'High income',
    'BLM': 'High income',
    'SHN': 'Upper middle income',
    'SPM': 'High income',
    'TKL': 'High income',
    'ESH': 'Lower middle income'
})

# Apply mapping
df['income_group'] = df['origin_ISO'].map(income_map)


# %%
# df2 = pd.read_csv("wb_data_all_fields_with_income_group.csv")
# %%
# merged_df = df.merge(df2[['income_group']], left_index=True, right_index=True)

# %%
# merged_df['origin_continent'] = merged_df['origin_continent'].fillna('NA')
# merged_df['destination_continent'] = merged_df['destination_continent'].fillna('NA')
# merged_df['origin_region'] = merged_df['origin_region'].fillna('NA')
# merged_df['destination_region'] = merged_df['destination_region'].fillna('NA')
# %%
# merged_df.to_csv("D:/STUDY/MS/world bank projects/after grad/WorldBank_Africa_Transport_Bottleneck/Data/wb_data_all_fields.csv", index=False)
# %%
df.to_csv("../Data/wb_data_all_fields_final.csv", index=False)