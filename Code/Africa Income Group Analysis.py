#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the full dataset
file_path = r"wb_data_all_fields_final.csv"
df_filtered = pd.read_csv(file_path, keep_default_na = False)
df_filtered.shape

#%%
#in case origin and destination continent columns have NaN values
# df_filtered['origin_continent'] = df_filtered['origin_continent'].fillna('NA')
# df_filtered['destination_continent'] = df_filtered['destination_continent'].fillna('NA')
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
df_filtered['income_group'] = df_filtered['origin_ISO'].map(income_map)

#%%
# Check for missing mappings
missing = df_filtered[df_filtered['income_group'].isna()]['origin_ISO'].unique()
print("Missing ISO codes:", missing)

# Preview the mapped column
print(df_filtered[['origin_ISO', 'income_group']].drop_duplicates().sort_values('origin_ISO').head(20))

# # Save to new CSV
# output_path = r"C:\Users\bsiva\Downloads\wb_data_all_fields_with_income_group.csv"
# df_filtered.to_csv(output_path, index=False)
# print(f"Updated file saved to: {output_path}")

df_filtered_africa = df_filtered[df_filtered['origin_continent'] == 'AF'].copy()

# Standardize income group labels (optional but helpful)
df_filtered_africa['income_group'] = df_filtered_africa['income_group'].replace({
    'Low income': 'Low',
    'Lower middle income': 'Lower-middle',
    'Upper middle income': 'Upper-middle',
    'High income': 'High'
})

# Use existing 'destination_continent' column and map to friendly names
continent_map = {
    'AF': 'Africa',
    'EU': 'Europe',
    'AS': 'Asia',
    'NA': 'North America',
    'SA': 'South America',
    'OC': 'Oceania'
}
#%%
df_filtered_africa['destination_region'] = df_filtered_africa['destination_continent'].map(continent_map)


# %%

# Set seaborn style
sns.set(style="whitegrid")

# Reorder income groups
income_order = ['Low', 'Lower-middle', 'Upper-middle', 'High']

#%%
# ----------- FIXED: 1. Total Export Volume by Income Group -----------
export_volume = df_filtered_africa.groupby('income_group')['flow(tonne)'].sum().reindex(income_order)

plt.figure(figsize=(8, 4))
ax = sns.barplot(x=export_volume.index, y=export_volume.values, palette='viridis')

# Add value labels
for i, v in enumerate(export_volume.values):
    ax.text(i, v + 0.05 * max(export_volume.values), f'{int(v):,}', ha='center', fontsize=9)

plt.title("Total Export Volume by Income Group")
plt.ylabel("Total Volume (tonnes)")
plt.xlabel("Income Group")
plt.tight_layout()
plt.show()

#%%
# ----------- 2. Average Unit Logistics Cost by Income Group -----------
avg_cost = df_filtered_africa.groupby('income_group')['Unit logistics costs ($/ton)'].mean().reindex(income_order)
plt.figure(figsize=(8, 4))
sns.barplot(x=avg_cost.index, y=avg_cost.values, palette='crest')
plt.title("Average Unit Logistics Cost ($/ton)")
plt.ylabel("Average Cost ($/ton)")
plt.xlabel("Income Group")
plt.tight_layout()
plt.show()

#%%
# ----------- 3. Mode of Transport Distribution (with labels) -----------
import matplotlib.pyplot as plt

mode_dist = pd.crosstab(df_filtered_africa['income_group'], df_filtered_africa['Mode_name'], normalize='index') * 100
mode_dist = mode_dist.reindex(income_order)

# Plot
ax = mode_dist.plot(kind='bar', stacked=True, figsize=(10, 5), colormap='Set3')

# Add labels to each segment
for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', label_type='center', fontsize=8)

plt.title("Mode of Transport Distribution by Income Group")
plt.ylabel("Percentage of Shipments (%)")
plt.xlabel("Income Group")
plt.legend(title='Transport Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


 #%%
# # ----------- 4. Top 5 Commodities per Income Group -----------
# top_commodities = (
#     df_filtered_africa.groupby(['income_group', 'IFM_HS'])['flow(tonne)']
#     .sum()
#     .reset_index()
#     .sort_values(['income_group', 'flow(tonne)'], ascending=[True, False])
# )
# for group in income_order:
#     subset = top_commodities[top_commodities['income_group'] == group].head(5)
#     plt.figure(figsize=(8, 4))
#     sns.barplot(x='flow(tonne)', y='IFM_HS', data=subset, palette='flare')
#     plt.title(f"Top 5 Commodities in {group} Income Countries")
#     plt.xlabel("Flow (tonnes)")
#     plt.ylabel("Commodity (IFM_HS)")
#     plt.tight_layout()
#     plt.show()

#%%
# ----------- 5. Top Destination Regions by Income Group -----------
top_dest_regions = (
    df_filtered_africa
    .groupby(['income_group', 'destination_continent'])['flow(tonne)']
    .sum()
    .reset_index()
    .sort_values(['income_group', 'flow(tonne)'], ascending=[True, False])
)
for group in income_order:
    subset = top_dest_regions[top_dest_regions['income_group'] == group]
    plt.figure(figsize=(8, 4))
    sns.barplot(x='flow(tonne)', y='destination_continent', data=subset, palette='pastel')
    plt.title(f"Top Destination Continents for {group} Income Countries")
    plt.xlabel("Flow (tonnes)")
    plt.ylabel("Destination Continent")
    plt.tight_layout()
    plt.show()
# %%

# Count unique countries by income group
country_counts = df_filtered_africa.groupby('income_group')['origin_ISO'].nunique().reindex(income_order)

# Print or use in a plot
print("Number of Countries per Income Group:\n", country_counts)

# %%

plt.figure(figsize=(6, 4))
sns.barplot(x=country_counts.index, y=country_counts.values, palette='pastel')
plt.title("Number of African Countries by Income Group")
plt.ylabel("Number of Unique Countries")
plt.xlabel("Income Group")
plt.tight_layout()
plt.show()

# %%
# # Get unique countries per income group
# country_table = df_filtered_africa[['origin_ISO', 'income_group']].drop_duplicates().sort_values('income_group')

# # View table
# print(country_table.to_string(index=False))
#%% 

# Set your target income group (e.g., "Lower-middle", "Upper-middle", "Low", "High")
selected_group = "Lower-middle"

#  Filter the dataframe
df_filtered_group = df_filtered_africa[df_filtered_africa['income_group'] == selected_group]

#  Calculate % share per commodity
grouped = df_filtered_group.groupby('IFM_HS')['flow(tonne)'].sum()
percentage = (100 * grouped / grouped.sum()).reset_index()
percentage.columns = ['IFM_HS', 'percentage']

#  Get top 5 commodities
top_5 = percentage.sort_values('percentage', ascending=False).head(5)

#  Plot
plt.figure(figsize=(8, 5))
sns.barplot(data=top_5, x='percentage', y='IFM_HS', palette='Spectral')
plt.title(f"Top 5 Commodities as % of Total Exports ({selected_group} Income)")
plt.xlabel("Share of Total Exports (%)")
plt.ylabel("Commodity (IFM_HS)")
plt.tight_layout()
plt.show()

# %%
#  Set your target income group (e.g., "Lower-middle", "Upper-middle", "Low", "High")
selected_group = "Upper-middle"

#  Filter the dataframe
df_filtered_group = df_filtered_africa[df_filtered_africa['income_group'] == selected_group]

#  Calculate % share per commodity
grouped = df_filtered_group.groupby('IFM_HS')['flow(tonne)'].sum()
percentage = (100 * grouped / grouped.sum()).reset_index()
percentage.columns = ['IFM_HS', 'percentage']

#  Get top 5 commodities
top_5 = percentage.sort_values('percentage', ascending=False).head(5)

#  Plot
plt.figure(figsize=(8, 5))
sns.barplot(data=top_5, x='percentage', y='IFM_HS', palette='Spectral')
plt.title(f"Top 5 Commodities as % of Total Exports ({selected_group} Income)")
plt.xlabel("Share of Total Exports (%)")
plt.ylabel("Commodity (IFM_HS)")
plt.tight_layout()
plt.show()
# %%
#  Set your target income group (e.g., "Lower-middle", "Upper-middle", "Low", "High")
selected_group = "Low"

#  Filter the dataframe
df_filtered_group = df_filtered_africa[df_filtered_africa['income_group'] == selected_group]

#  Calculate % share per commodity
grouped = df_filtered_group.groupby('IFM_HS')['flow(tonne)'].sum()
percentage = (100 * grouped / grouped.sum()).reset_index()
percentage.columns = ['IFM_HS', 'percentage']

#  Get top 5 commodities
top_5 = percentage.sort_values('percentage', ascending=False).head(5)

#  Plot
plt.figure(figsize=(8, 5))
sns.barplot(data=top_5, x='percentage', y='IFM_HS', palette='Spectral')
plt.title(f"Top 5 Commodities as % of Total Exports ({selected_group} Income)")
plt.xlabel("Share of Total Exports (%)")
plt.ylabel("Commodity (IFM_HS)")
plt.tight_layout()
plt.show()
# %%
#  Set your target income group (e.g., "Lower-middle", "Upper-middle", "Low", "High")
selected_group = "High"

#  Filter the dataframe
df_filtered_group = df_filtered_africa[df_filtered_africa['income_group'] == selected_group]

#  Calculate % share per commodity
grouped = df_filtered_group.groupby('IFM_HS')['flow(tonne)'].sum()
percentage = (100 * grouped / grouped.sum()).reset_index()
percentage.columns = ['IFM_HS', 'percentage']

#  Get top 5 commodities
top_5 = percentage.sort_values('percentage', ascending=False).head(5)

#  Plot
plt.figure(figsize=(8, 5))
sns.barplot(data=top_5, x='percentage', y='IFM_HS', palette='Spectral')
plt.title(f"Top 5 Commodities as % of Total Exports ({selected_group} Income)")
plt.xlabel("Share of Total Exports (%)")
plt.ylabel("Commodity (IFM_HS)")
plt.tight_layout()
plt.show()
# %%

# Set income groups in order
income_groups = ['Low', 'Lower-middle', 'Upper-middle', 'High']

# Create 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Loop through each income group and plot
for i, group in enumerate(income_groups):
    df_filtered_group = df_filtered_africa[df_filtered_africa['income_group'] == group]
    grouped = df_filtered_group.groupby('IFM_HS')['flow(tonne)'].sum()
    percentage = (100 * grouped / grouped.sum()).reset_index()
    percentage.columns = ['IFM_HS', 'percentage']
    top_5 = percentage.sort_values('percentage', ascending=False).head(5)

    sns.barplot(data=top_5, x='percentage', y='IFM_HS', ax=axes[i], palette='Spectral')
    axes[i].set_title(f"{group} Income")
    axes[i].set_xlabel("Share of Total Exports (%)")
    axes[i].set_ylabel("Commodity (IFM_HS)")

plt.suptitle("Top 5 Commodities as % of Total Exports by Income Group", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")
income_order = ['Low', 'Lower-middle', 'Upper-middle', 'High']

# ---- PREP DATA ----
# Export volume
volume = df_filtered_africa.groupby('income_group')['flow(tonne)'].sum().reindex(income_order)

# Cost per ton
cost = df_filtered_africa.groupby('income_group')['Unit logistics costs ($/ton)'].mean().reindex(income_order)

# Mode distribution (%)
mode_dist = pd.crosstab(df_filtered_africa['income_group'], df_filtered_africa['Mode_name'], normalize='index') * 100
mode_dist = mode_dist.reindex(income_order)

# Country count
country_counts = df_filtered_africa.groupby('income_group')['origin_ISO'].nunique().reindex(income_order)

# ---- PLOT ALL 4 ----
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# 1. Total Export Volume
sns.barplot(x=volume.index, y=volume.values, ax=axes[0], palette='viridis')
axes[0].set_title("Total Export Volume by Income Group")
axes[0].set_ylabel("Tonnes")
for i, v in enumerate(volume.values):
    axes[0].text(i, v + 0.05 * max(volume.values), f'{int(v):,}', ha='center', fontsize=9)

# 2. Average Cost per Ton
sns.barplot(x=cost.index, y=cost.values, ax=axes[1], palette='crest')
axes[1].set_title("Average Unit Logistics Cost ($/ton)")
axes[1].set_ylabel("USD")

# 3. Mode Distribution (stacked bar)
mode_dist.plot(kind='bar', stacked=True, ax=axes[2], colormap='Set3', legend=False)
axes[2].set_title("Mode of Transport Distribution")
axes[2].set_ylabel("Percentage")

# 4. Country Count
sns.barplot(x=country_counts.index, y=country_counts.values, ax=axes[3], palette='pastel')
axes[3].set_title("Number of Countries per Income Group")
axes[3].set_ylabel("Countries")

# Title and layout
plt.suptitle("Export Logistics Insights by Income Group", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Reorder income groups
income_order = ['Low', 'Lower-middle', 'Upper-middle', 'High']

# Grouped data
export_volume = df_filtered_africa.groupby('income_group')['flow(tonne)'].sum().reindex(income_order)
avg_cost = df_filtered_africa.groupby('income_group')['Unit logistics costs ($/ton)'].mean().reindex(income_order)

# Combine into single DataFrame
combo_df = pd.DataFrame({
    'Export Volume (Million Tonnes)': export_volume / 1e6,
    'Avg Logistics Cost ($/ton)': avg_cost
}).reset_index().rename(columns={'income_group': 'Income Group'})

# Plot
sns.set(style="whitegrid")
fig, ax1 = plt.subplots(figsize=(10, 6))

# Barplot for export volume
sns.barplot(data=combo_df, x='Income Group', y='Export Volume (Million Tonnes)', ax=ax1, color='steelblue', label='Export Volume')
ax1.set_ylabel("Export Volume (Million Tonnes)", color='steelblue')
ax1.tick_params(axis='y', labelcolor='steelblue')

# Twin axis for cost
ax2 = ax1.twinx()
sns.barplot(data=combo_df, x='Income Group', y='Avg Logistics Cost ($/ton)', ax=ax2, color='tomato', alpha=0.7, label='Avg Logistics Cost')
ax2.set_ylabel("Avg Logistics Cost ($/ton)", color='tomato')
ax2.tick_params(axis='y', labelcolor='tomato')

# Title & Legends
plt.title("Export Volume vs. Average Logistics Cost by Income Group")
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.tight_layout()
plt.show()

# %%
