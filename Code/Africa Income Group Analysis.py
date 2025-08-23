#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the full dataset
file_path = r"../Data/wb_data_all_fields_final.csv"
df_filtered = pd.read_csv(file_path, keep_default_na = False)
df_filtered.shape
# # Save to new CSV
# output_path = r"C:\Users\bsiva\Downloads\wb_data_all_fields_with_income_group.csv"
# df_filtered.to_csv(output_path, index=False)
# print(f"Updated file saved to: {output_path}")

#%%
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
