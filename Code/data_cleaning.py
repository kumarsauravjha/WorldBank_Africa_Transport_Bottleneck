#%%
import pandas as pd
import numpy as np
import json
import pycountry
import pycountry_convert as pc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
#%%
pd.set_option('display.max_columns', None)
#%%
df = pd.read_csv("wb_data_all_fields_final.csv", keep_default_na = False)
# %%
df.shape
# %%
'''getting rid of the empty values'''
df_filtered = df[
    (df['flow(tonne)'] > 0) &
    # (df['distance(km)'] > 0) &
    (df['Unit logistics costs ($/ton)'] > 0)
]
df_filtered.shape

#%%
df_filtered2 = df[
    (df['flow(tonne)'] > 0) &
    # (df['distance(km)'] > 0) &
    (df['Unit logistics costs ($/ton)'] > 0)
]
df_filtered2.shape
# %%
df_filtered.isna().sum()
# %%
# df_filtered[['origin_continent', 'destination_continent', 'origin_region', 'destination_region']] = \
# df_filtered[['origin_continent', 'destination_continent', 'origin_region', 'destination_region']].fillna('NA')

# df_filtered.isna().sum()
# %%
print(df_filtered.columns)
# %%
df_filtered['IFM_HS'].unique()
# %%
len(df_filtered['IFM_HS'].unique())
# %%
len(df_filtered['commodity_index'].unique())
# %%
df_filtered['Total_logistics_cost($)'] = df_filtered['Unit logistics costs ($/ton)'] * df_filtered['flow(tonne)']

# %%
df_filtered['Total_logistics_cost($)'].isna().sum()
# %%
df_filtered['Logistics_cost_per_km'] = df_filtered['Total_logistics_cost($)'] / df_filtered['distance(km)']

#%%
len(df_filtered['Model'].unique())
# %%
'''So Model field not needed for analysis, next '''
df_filtered['Mode_name'].value_counts()
# %%
'''Some Mode analysis in general'''

df_filtered[df_filtered['Mode_name'] == 'Air']['Unit logistics costs ($/ton)'].describe()
# %%
import matplotlib.pyplot as plt

# Extract just the Air‐mode unit costs
air_costs = df_filtered[df_filtered['Mode_name']=='Air']['Unit logistics costs ($/ton)']

# Plot histogram
plt.figure()
plt.hist(air_costs, bins=50)
plt.xlabel('Unit logistics cost ($/ton)')
plt.ylabel('Frequency')
plt.title('Distribution of Unit Logistics Costs for Air Shipments')
plt.tight_layout()
plt.show()

# %%
'''seperating into anomalies and normal data'''
clf = IsolationForest(contamination=0.01, random_state=0)
df_filtered['anomaly'] = clf.fit_predict(df_filtered[['Unit logistics costs ($/ton)']])

# 2) Split into “weird” vs. “normal”
anomalies   = df_filtered[df_filtered['anomaly'] == -1]
clean_data  = df_filtered[df_filtered['anomaly'] ==  1]
# %%
print('Number of anomalies: ',anomalies.shape[0])
print('Number of normal data points: ',clean_data.shape[0])

#%%
#Anomalies plots

#%%
# Step 1: Compute Q1 and Q3 for unit logistics cost
Q1 = clean_data['Unit logistics costs ($/ton)'].quantile(0.25)
Q3 = clean_data['Unit logistics costs ($/ton)'].quantile(0.75)
IQR = Q3 - Q1

# Step 2: Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 3: Filter out the outliers
tier_normal_clean = clean_data[
    (clean_data['Unit logistics costs ($/ton)'] >= lower_bound) &
    (clean_data['Unit logistics costs ($/ton)'] <= upper_bound)
]

# Optional: summary
print(f"Original rows: {clean_data.shape[0]:,}")
print(f"Cleaned rows: {tier_normal_clean.shape[0]:,}")

#%%
'''cross verifying fom the flow presentations'''

print('Average transport cost when Africa is exporting = ',df_filtered[df_filtered['origin_continent'] == 'AF']['Unit logistics costs ($/ton)'].mean())
#%%
print('Average flow(tonne) when Africa is exporting = ',df_filtered[df_filtered['origin_continent'] == 'AF']['flow(tonne)'].mean())
#%%
print('Average transport cost when Europe is exporting = ',df_filtered[df_filtered['origin_continent'] == 'EU']['Unit logistics costs ($/ton)'].mean())

#%%
print('Average flow(tonne) when Europe is exporting = ',df_filtered[df_filtered['origin_continent'] == 'EU']['flow(tonne)'].mean())

#%%
print('Average transport cost when North-America is exporting = ',df_filtered[df_filtered['origin_continent'] == 'NA']['Unit logistics costs ($/ton)'].mean())

#%%
print('Average flow(tonne) when North-America is exporting = ',df_filtered[df_filtered['origin_continent'] == 'NA']['flow(tonne)'].mean())
#%%
print('Average transport cost when South-America is exporting = ',df_filtered[df_filtered['origin_continent'] == 'SA']['Unit logistics costs ($/ton)'].mean())

#%%
print('Average flow(tonne) when South-America is exporting = ',df_filtered[df_filtered['origin_continent'] == 'SA']['flow(tonne)'].mean())

#%%
print('Average transport cost when Oceania is exporting = ',df_filtered[df_filtered['origin_continent'] == 'OC']['Unit logistics costs ($/ton)'].mean())

#%%
print('Average flow(tonne) when Oceania is exporting = ',df_filtered[df_filtered['origin_continent'] == 'OC']['flow(tonne)'].mean())

#%%
print('Shipments count by different modes, when Africa is exporting:')
df_filtered[df_filtered['origin_continent'] == 'AF']['Mode_name'].value_counts()

#%%
print('Shipments count by different modes, when Africa is importing:')
df_filtered[df_filtered['destination_continent'] == 'AF']['Mode_name'].value_counts()

#%%
# Define the continents of interest
continents = ['AF', 'EU', 'NA', 'SA', 'OC']

# Calculate averages
avg_costs = [df_filtered[df_filtered['origin_continent'] == cont]['Unit logistics costs ($/ton)'].mean() for cont in continents]
avg_flows = [df_filtered[df_filtered['origin_continent'] == cont]['flow(tonne)'].mean() for cont in continents]

# Prepare plotting positions
x = np.arange(len(continents))
width = 0.35

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

# Plot bars
bars1 = ax1.bar(x - width/2, avg_costs, width, label='Avg Transport Cost ($/ton)', color='steelblue')
bars2 = ax2.bar(x + width/2, avg_flows, width, label='Avg Flow (tonnes)', color='orange')

# Axis labels and title
ax1.set_xlabel('Exporting Continent')
ax1.set_ylabel('Transport Cost ($/ton)', color='steelblue')
ax2.set_ylabel('Flow (tonnes)', color='orange')
ax1.set_title('Average Transport Cost and Flow by Exporting Continent')

# X-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(continents)

# Legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Layout
plt.tight_layout()
plt.show()

#%%
# Calculate averages
avg_costs_import = [df_filtered[df_filtered['destination_continent'] == cont]['Unit logistics costs ($/ton)'].mean() for cont in continents]
avg_costs_import = [df_filtered[df_filtered['destination_continent'] == cont]['flow(tonne)'].mean() for cont in continents]

# Prepare plotting positions
x = np.arange(len(continents))
width = 0.35

# Create figure and axes
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

# Plot bars
bars1 = ax1.bar(x - width/2, avg_costs_import, width, label='Avg Transport Cost ($/ton)', color='steelblue')
bars2 = ax2.bar(x + width/2, avg_costs_import, width, label='Avg Flow (tonnes)', color='orange')

# Axis labels and title
ax1.set_xlabel('Exporting Continent')
ax1.set_ylabel('Transport Cost ($/ton)', color='steelblue')
ax2.set_ylabel('Flow (tonnes)', color='orange')
ax1.set_title('Average Transport Cost and Flow by Exporting Continent')

# X-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(continents)

# Legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Layout
plt.tight_layout()
plt.show()
# %%
clean_data[clean_data['Mode_name'] == 'Air']['Unit logistics costs ($/ton)'].describe()
# %%

# original & cleaned Air costs
air_orig  = df_filtered[df_filtered['Mode_name']=='Air']['Unit logistics costs ($/ton)']
air_clean = clean_data[df_filtered['Mode_name']=='Air']['Unit logistics costs ($/ton)']

# set up a 2×2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1) Histogram before
axes[0,0].hist(air_orig, bins=50)
axes[0,0].set_title('Air Costs: Original')
axes[0,0].set_xlabel('$ per ton')
axes[0,0].set_ylabel('Freq')

# 2) Histogram after
axes[0,1].hist(air_clean, bins=50, color='orange')
axes[0,1].set_title('Air Costs: Cleaned')
axes[0,1].set_xlabel('$ per ton')

# 3) Boxplot before
axes[1,0].boxplot(air_orig, vert=False)
axes[1,0].set_title('Boxplot: Original')

# 4) Boxplot after
axes[1,1].boxplot(air_clean, vert=False)
axes[1,1].set_title('Boxplot: Cleaned')

plt.tight_layout()
plt.show()

# %%
clean_data[clean_data['Mode_name'] == 'Air'].sort_values(by='Unit logistics costs ($/ton)', ascending=False).head(10)

#%%

clean_data[clean_data['Mode_name'] == 'Sea'].sort_values(by='Unit logistics costs ($/ton)', ascending=False).head(10)

#%%
clean_data[clean_data['Mode_name'] == 'Rail'].sort_values(by='Unit logistics costs ($/ton)', ascending=False).head(10)

#%%
clean_data[clean_data['Mode_name'] == 'Road'].sort_values(by='Unit logistics costs ($/ton)', ascending=False).head(10)

# %%
anomalies['Mode_name'].value_counts()
#%%
clean_data['Mode_name'].value_counts()
# %%
# # 1) Grab your cleaned Air set (anomaly=1)
# air_clean = clean_data[clean_data['Mode_name']=='Air'].copy()

# # 2) Compute percentiles
# p95 = air_clean['Unit logistics costs ($/ton)'].quantile(0.95)
# p99 = air_clean['Unit logistics costs ($/ton)'].quantile(0.99)

# # 3) Split into three tiers
# tier_normal   = air_clean[air_clean['Unit logistics costs ($/ton)'] <= p95]
# tier_moderate = air_clean[(air_clean['Unit logistics costs ($/ton)'] > p95) &
#                           (air_clean['Unit logistics costs ($/ton)'] <= p99)]
# tier_extreme  = air_clean[air_clean['Unit logistics costs ($/ton)'] > p99]

# print("Normal:",   tier_normal.shape[0], "rows")
# print("Moderate:", tier_moderate.shape[0], "rows")
# print("Extreme:",  tier_extreme.shape[0], "rows")

# # %%
# clean_data2 = clean_data.copy()

# # 2) Compute percentiles
# p95 = clean_data2['Unit logistics costs ($/ton)'].quantile(0.95)
# p99 = clean_data2['Unit logistics costs ($/ton)'].quantile(0.99)

# # 3) Split into three tiers
# tier_normal   = clean_data2[clean_data2['Unit logistics costs ($/ton)'] <= p95]
# tier_moderate = clean_data2[(clean_data2['Unit logistics costs ($/ton)'] > p95) &
#                           (clean_data2['Unit logistics costs ($/ton)'] <= p99)]
# tier_extreme  = clean_data2[clean_data2['Unit logistics costs ($/ton)'] > p99]

# print("Normal:",   tier_normal.shape[0], "rows")
# print("Moderate:", tier_moderate.shape[0], "rows")
# print("Extreme:",  tier_extreme.shape[0], "rows")
# %%
mode_quants = (
    clean_data
     .groupby('Mode_name')['Unit logistics costs ($/ton)']
     .quantile([0.95, 0.99])
     .unstack(level=1)
     .rename(columns={0.95:'p95', 0.99:'p99'})
)
mode_quants
# %%
# 2) broadcast them back onto every row
clean_data = clean_data.join(
    mode_quants,
    on='Mode_name'
)

# %%
def assign_tier(row):
    v, p95, p99 = row['Unit logistics costs ($/ton)'], row['p95'], row['p99']
    if   v <= p95:   return 'normal'
    elif v <= p99:   return 'moderate'
    else:            return 'extreme'

clean_data['tier'] = clean_data.apply(assign_tier, axis=1)
# %%
clean_data['tier'].value_counts()
# %%
summary_table = clean_data.groupby(['Mode_name', 'tier']).size().reset_index(name='count')
summary_pivot = summary_table.pivot(index='Mode_name', columns='tier', values='count').fillna(0).astype(int)

# Plot stacked bar chart
summary_pivot.plot(kind='bar', stacked=True, figsize=(10, 6))

plt.title('Shipment Tier Distribution by Mode')
plt.xlabel('Mode of Transport')
plt.ylabel('Number of Shipments')
plt.legend(title='Tier')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
# %%
tier_normal = clean_data[clean_data['tier'] == 'normal']
tier_normal.shape
# %%
tier_normal[tier_normal['Mode_name'] == 'Air' ]['Unit logistics costs ($/ton)'].describe()
# %%
tier_normal_air = tier_normal[tier_normal['Mode_name'] == 'Air']['Unit logistics costs ($/ton)']

plt.figure()
plt.hist(tier_normal_air, bins=20)
plt.xlabel('Unit logistics cost ($/ton)')
# plt.xscale('log')
plt.ylabel('Frequency')
plt.title('Distribution of Unit Logistics Costs for Normal-tier air shipments')
plt.tight_layout()
plt.show()
# %%
tier_normal_sea = tier_normal[tier_normal['Mode_name'] == 'Road']['Unit logistics costs ($/ton)']

plt.figure()
plt.hist(tier_normal_sea, bins=20)
plt.xlabel('Unit logistics cost ($/ton)')
plt.ylabel('Frequency')
plt.title('Distribution of Unit Logistics Costs for Normal-tier sea shipments')
plt.tight_layout()
plt.show()
# %%
# Group by binned distance and mode, then calculate mean cost
tier_normal['Distance_bin'] = pd.cut(tier_normal['distance(km)'], bins=50)

grouped = (
    tier_normal
    .groupby(['Distance_bin', 'Mode_name'])['Unit logistics costs ($/ton)']
    .mean()
    .reset_index()
)

# Extract bin midpoints for plotting
grouped['Distance_mid'] = grouped['Distance_bin'].apply(lambda x: x.mid)

# Plot line graph
plt.figure(figsize=(10, 6))
for mode in grouped['Mode_name'].unique():
    subset = grouped[grouped['Mode_name'] == mode]
    plt.plot(subset['Distance_mid'], subset['Unit logistics costs ($/ton)'], label=mode)

plt.xlabel('Distance (km)')
plt.ylabel('Average Unit Logistics Cost ($/ton)')
plt.title('Distance vs Average Unit Logistics Cost by Mode')
plt.legend(title='Mode')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

grouped = (
    tier_normal
    .groupby(['Mode_name', 'Distance_bin'])['Logistics_cost_per_km']
    .mean()
    .reset_index()
)
grouped['Distance_mid'] = grouped['Distance_bin'].apply(lambda x: x.mid)

#%%
plt.figure(figsize=(10,6))
for mode in grouped['Mode_name'].unique():
    dfm = grouped[grouped['Mode_name']==mode]
    plt.plot(dfm['Distance_mid'], dfm['Logistics_cost_per_km'], label=mode)

plt.xlabel('Distance (km)')
plt.ylabel('Avg Cost per km ($/km)')
plt.title('Distance vs Cost per km by Mode')
plt.legend()
plt.xscale('log')            # Optional: stretch out the long tail
plt.tight_layout()
plt.show()

# %%
tier_normal[tier_normal['origin_continent'] == 'AF']['trade_type'].value_counts()
# %%
'''Analysis of the anomalies data'''
anomalies.shape
# %%
anomalies['Mode_name'].value_counts()
# %%
anomalies['origin_continent'].value_counts()
# %%
anomalies['destination_continent'].value_counts()
# %%
anomalies[anomalies['origin_continent'] == 'AF'].sort_values(by='Unit logistics costs ($/ton)', ascending=False).head(10)
# %%
anomalies[anomalies['origin_continent'] == 'AF']['trade_type'].value_counts()

#%%
anomalies[anomalies['origin_continent'] == 'AF']['Mode_name'].value_counts()
# %%
anomalies[(anomalies['origin_continent'] == 'AF') & (anomalies['Mode_name'] == 'Rail')]
# %%
anomalies[(anomalies['origin_continent'] == 'AF') & (anomalies['Mode_name'] == 'Air')].sort_values(by='Unit logistics costs ($/ton)', ascending=False).head(20)
# %%
anomalies[(anomalies['origin_continent'] == 'AF') & (anomalies['Mode_name'] == 'Sea')].sort_values(by='Unit logistics costs ($/ton)', ascending=False).head(10)
# %%
# Group by country pairs and count anomalies
corridor_counts = (
    anomalies[anomalies['origin_continent'] == 'AF']
    .groupby(['origin_ISO', 'destination_ISO'])
    .agg(anomaly_count=('anomaly', 'size'),
         total_cost=('Total_logistics_cost($)', 'sum'))
    .reset_index()
    .sort_values(by='anomaly_count', ascending=False)
    .head(20)
)

pivot_corridors = corridor_counts.pivot(index='origin_ISO', columns='destination_ISO', values='anomaly_count')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_corridors, cmap='Reds', annot=True, fmt='g')
plt.title("Top 20 Anomalous Corridors (Count of Anomalies)")
plt.show()

# %%
tier_normal['distance(km)'].describe()
# %%
# Filter anomalies where origin is Africa
an_af = anomalies[anomalies['origin_continent'] == 'AF']

# Count top commodities
hotspot_counts = (
    an_af['IFM_HS']
    .value_counts()
    .head(10)
    .sort_values(ascending=True)  # for a clean horizontal bar plot
)

# Plot
plt.figure(figsize=(10, 6))
hotspot_counts.plot(kind='barh', color='darkred')
plt.xlabel('Number of Anomalies')
plt.title('Top 10 Anomalous Commodities (Origin: Africa)')
plt.tight_layout()
plt.show()

# %%

# Use seaborn style
sns.set(style='whitegrid')

# Data: same as before
an_af = anomalies[anomalies['origin_continent'] == 'AF']
hotspot_counts = (
    an_af['IFM_HS']
    .value_counts()
    .head(10)
    .sort_values(ascending=False)
)

# Plot
plt.figure(figsize=(12, 7))
sns.barplot(x=hotspot_counts.values, y=hotspot_counts.index, palette='rocket')

# Add data labels
for i, v in enumerate(hotspot_counts.values):
    plt.text(v + 100, i, str(v), va='center', fontweight='bold')

# Labels and title
plt.xlabel('Number of Anomalous Shipments', fontsize=12)
plt.ylabel('Commodity Group (IFM_HS)', fontsize=12)
plt.title('Top 10 Anomalous Commodities from Africa', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# %%
# Data: same as before
an_af = anomalies[anomalies['destination_continent'] == 'AF']
hotspot_counts = (
    an_af['IFM_HS']
    .value_counts()
    .head(10)
    .sort_values(ascending=False)
)

# Plot
plt.figure(figsize=(12, 7))
sns.barplot(x=hotspot_counts.values, y=hotspot_counts.index, palette='rocket')

# Add data labels
for i, v in enumerate(hotspot_counts.values):
    plt.text(v + 100, i, str(v), va='center', fontweight='bold')

# Labels and title
plt.xlabel('Number of Anomalous Shipments', fontsize=12)
plt.ylabel('Commodity Group (IFM_HS)', fontsize=12)
plt.title('Top 10 Anomalous Commodities imported to Africa', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
# %%
counts = anomalies['origin_continent'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Share of Anomalous Shipments by Origin Continent')
plt.axis('equal')
plt.show()

# %%
anomalies.columns
# %%
anomalies['trade_type'].value_counts()
# %%
# Filter anomalies where origin is Africa
an_af = anomalies[anomalies['origin_continent'] == 'AF']

# Count transport modes
mode_counts = an_af['Mode_name'].value_counts()

# Plot pie chart
plt.figure(figsize=(7, 7))
plt.pie(mode_counts, labels=mode_counts.index, autopct='%1.1f%%', startangle=90,
        colors=sns.color_palette('pastel'))

plt.title('Share of Transport Modes in Anomalies from Africa')
plt.axis('equal')  # ensures pie is circular
plt.tight_layout()
plt.show()

# %%
anomalies[anomalies['origin_continent'] == 'AF']['Mode_name'].value_counts()
# %%
# Count anomalies by income group
income_counts = anomalies[anomalies['origin_continent'] == 'AF']['income_group'].value_counts()
labels = income_counts.index
sizes = income_counts.values
colors = sns.color_palette('pastel')

# Create donut plot
plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=90,
    counterclock=False,
    wedgeprops={'width': 0.4},  # Makes it a donut
    colors=colors,
    textprops={'fontsize': 12}
)

# Add center text
plt.text(0, 0, 'Anomalies\nby Income', ha='center', va='center', fontsize=14, fontweight='bold')

# Add title
plt.title('Distribution of Anomalous Shipments by Income Group: Exported from Africa', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
anomalies[anomalies['origin_continent'] == 'AF']['origin_type'].value_counts()
# %%

# Filter for anomalies originating from Africa
an_af = anomalies[anomalies['origin_continent'] == 'AF']

# Count by origin_type
origin_counts = an_af['origin_type'].value_counts()

# Count by destination_type
dest_counts = an_af['destination_type'].value_counts()

# Plot side-by-side bars
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(x=origin_counts.index, y=origin_counts.values, ax=axes[0], palette='Blues')
axes[0].set_title('Origin Type (Africa)')
axes[0].set_ylabel('Number of Anomalies')
axes[0].set_xlabel('Origin Type')

sns.barplot(x=dest_counts.index, y=dest_counts.values, ax=axes[1], palette='Greens')
axes[1].set_title('Destination Type')
axes[1].set_ylabel('')
axes[1].set_xlabel('Destination Type')

plt.suptitle('Anomalous Shipments by Geography Type (Origin: Africa)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
# Step 1: Run anomaly detection on distance
clf_distance = IsolationForest(contamination=0.01, random_state=0)
df_filtered['anomaly_distance'] = clf_distance.fit_predict(df_filtered[['distance(km)']])

# Step 2: Split into anomalous and normal sets based on distance
distance_anomalies = df_filtered[df_filtered['anomaly_distance'] == -1]
distance_normal = df_filtered[df_filtered['anomaly_distance'] == 1]

# Step 3: Print counts
print("Number of distance anomalies:", distance_anomalies.shape[0])
print("Number of normal distance rows:", distance_normal.shape[0])

# %%
plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=distance_anomalies,
    x='distance(km)',
    y='Unit logistics costs ($/ton)',
    hue='Mode_name',
    alpha=0.6,
    palette='Set2'
)

plt.title('Distance vs Unit Cost for Distance Anomalies')
plt.xlabel('Distance (km)')
plt.ylabel('Unit Logistics Cost ($/ton)')
plt.legend(title='Mode', loc='upper right')
plt.tight_layout()
plt.show()

# %%
distance_anomalies['Mode_name'].value_counts()
# %%
# Create distance bins
distance_anomalies['Distance_bin'] = pd.cut(distance_anomalies['distance(km)'], bins=[0, 1000, 5000, 10000, 20000, 50000, 100000, 200000])

# Count by bin and mode
bin_mode_counts = distance_anomalies.groupby(['Distance_bin', 'Mode_name']).size().reset_index(name='count')

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=bin_mode_counts, x='Distance_bin', y='count', hue='Mode_name', palette='pastel')

plt.title('Distance Anomalies by Mode and Distance Range')
plt.xlabel('Distance Range (km)')
plt.ylabel('Anomaly Count')
plt.xticks(rotation=45)
plt.legend(title='Mode')
plt.tight_layout()
plt.show()

# %%

origin_counts = clean_data['origin_type'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(
    origin_counts.values,
    labels=origin_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette('pastel'),
    wedgeprops={'width': 0.4}
)
plt.text(0, 0, 'Normal\nOrigin Type', ha='center', va='center', fontsize=14, fontweight='bold')
plt.title('Distribution of Normal Shipments by Origin Type')
plt.tight_layout()
plt.show()

# %%
origin_counts = clean_data[clean_data['origin_continent'] == "AF"]['origin_type'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(
    origin_counts.values,
    labels=origin_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette('pastel'),
    wedgeprops={'width': 0.4}
)
plt.text(0, 0, 'Normal\nOrigin Type', ha='center', va='center', fontsize=14, fontweight='bold')
plt.title('Distribution of Normal Shipments by Origin Type')
plt.tight_layout()
plt.show()
# %%
# 1. Grouped average costs
avg_costs = (
    clean_data
    .groupby(['trade_type', 'origin_type', 'Mode_name'])['Unit logistics costs ($/ton)']
    .mean()
    .reset_index()
)

# 2. Use catplot for column-based faceting
g = sns.catplot(
    data=avg_costs,
    kind='bar',
    x='Mode_name',
    y='Unit logistics costs ($/ton)',
    hue='origin_type',
    col='trade_type',
    palette='Set2',
    height=5,
    aspect=1
)

# 3. Add titles and polish
g.set_titles(col_template='{col_name} Trade')
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle('Average Logistics Cost by Mode, Origin Type, and Trade Type', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()


# %%
origin_counts = clean_data[clean_data['origin_continent'] == "AF"]['origin_type'].value_counts()

plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    origin_counts.values,
    labels=origin_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=sns.color_palette('pastel'),
    wedgeprops={'width': 0.4},
    textprops={'fontsize': 14}  # label font size
)

# Set font size for % labels
for autotext in autotexts:
    autotext.set_fontsize(14)
    autotext.set_fontweight('bold')

# Center text
plt.text(0, 0, 'Normal\nOrigin Type', ha='center', va='center', fontsize=18, fontweight='bold')

# Title
plt.title('Distribution of Normal Shipments by Origin Type', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
# Assuming your DataFrame is called df and has the relevant fields
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_filtered,
    x='distance(km)',
    y='Unit logistics costs ($/ton)',
    hue='IFM_HS',  # or your actual column for commodity names/types
    palette='tab10',
    alpha=0.7,
    edgecolor='w',
    linewidth=0.5
)

plt.title('Unit Logistics Cost vs Distance by Commodity')
plt.xlabel('Distance (km)')
plt.ylabel('Unit Logistics Cost ($/ton)')
plt.legend(title='Commodity', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Define the mapping
supergroup_map = {
    'Food': 'Agriculture & Food',
    'Livestock': 'Agriculture & Food',
    'Rice_crops': 'Agriculture & Food',
    'Other_agriculture': 'Agriculture & Food',

    'Other_manufacturing': 'Manufactured Goods',
    'Textile': 'Manufactured Goods',
    'Transport_equipment': 'Manufactured Goods',
    'Electronic_devices': 'Manufactured Goods',
    'Metal_products': 'Manufactured Goods',

    'Coal': 'Raw Materials (Dry Bulk/Mining)',
    'Crude oil': 'Raw Materials (Dry Bulk/Mining)',
    'Gas': 'Raw Materials (Dry Bulk/Mining)',
    'Other_mining': 'Raw Materials (Dry Bulk/Mining)',
    'Other_minerals': 'Raw Materials (Dry Bulk/Mining)',

    'Iron_steel': 'Metals & Construction',
    'Other_metals': 'Metals & Construction',
    'Paper_wood': 'Metals & Construction',

    'Chemicals_plastic': 'Chemicals & Refinery Products',
    'Refined_oil': 'Chemicals & Refinery Products'
}

# Create new column
df_filtered['commodity_group'] = df_filtered['IFM_HS'].map(supergroup_map)

# %%
clean_data['commodity_group'].value_counts()
# %%
# Define the sampling fractions for each group
fractions = {
    'Manufactured Goods': 0.10,
    'Agriculture & Food': 0.10,
    'Raw Materials (Dry Bulk/Mining)': 0.20,
    'Metals & Construction': 0.20,
    'Chemicals & Refinery Products': 0.30
}

# Sample each group accordingly
sampled_df = clean_data.groupby('commodity_group', group_keys=False).apply(lambda x: x.sample(frac=fractions[x.name], random_state=42))

#%%
sampled_df_overseas = sampled_df[sampled_df['trade_type'] == 'overseas']

#%%
anomalies_overseas = anomalies[anomalies['trade_type'] == 'overseas']
#%%
sampled_df_overseas.shape
# %%
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=anomalies_overseas,
    x='distance(km)',
    y='Unit logistics costs ($/ton)',
    hue='IFM_HS',
    palette='Set2',
    alpha=0.7
)
plt.title('Unit Logistics Cost vs Distance (Sampled by Commodity Group)')
plt.xlabel('Distance (km)')
plt.ylabel('Unit Logistics Cost ($/ton)')
plt.legend(title='Commodity Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
