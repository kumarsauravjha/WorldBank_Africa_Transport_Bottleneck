#%%
import pandas as pd
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
df = pd.read_csv("wb_data_all_fields_final.csv", keep_default_na=False)
# %%
#africa as an exporter when trade is continental and overseas

df_africa_cont = df[(df['origin_continent'] == 'AF') & (df['trade_type'] == 'continental')]
df_africa_overseas = df[(df['origin_continent'] == 'AF') & (df['trade_type'] == 'overseas')]

features = ['Unit logistics costs ($/ton)', 'distance(km)', 'flow(tonne)','IFM_HS_encoded']
# %%
# Label Encoding
le = LabelEncoder()
df_africa_cont.loc[:, 'IFM_HS_encoded'] = le.fit_transform(df_africa_cont['IFM_HS'])

# Select Features
features = ['Unit logistics costs ($/ton)', 'distance(km)', 'flow(tonne)', 'IFM_HS_encoded']
X = df_africa_cont[features]

# Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
df_africa_cont.loc[:, 'anomaly_score'] = iso_forest.fit_predict(X)

# Filter anomalies
anomalies = df_africa_cont[df_africa_cont['anomaly_score'] == -1]

# Show Top 10 Anomalies
print(anomalies[['origin_ISO', 'destination_ISO', 'Unit logistics costs ($/ton)', 'IFM_HS', 'distance(km)', 'flow(tonne)']].sort_values(by='Unit logistics costs ($/ton)', ascending=False).head(10))

#%%
print(anomalies[['origin_ISO', 'destination_ISO', 'Unit logistics costs ($/ton)', 'IFM_HS', 'distance(km)', 'flow(tonne)']].sort_values(by='flow(tonne)', ascending=False).head(10))

#%%
anomalies[anomalies['flow(tonne)'] > 0.00000].shape
# %%
sns.scatterplot(data=df_africa_cont, x='distance(km)', y='Unit logistics costs ($/ton)', hue='IFM_HS')
plt.title('Anomalies: Distance vs. Unit Logistics Cost')
plt.xscale('log')
plt.yscale('log')
plt.show()
# %%
'''Now for overseas trade'''
# Label Encoding
le = LabelEncoder()
df_africa_overseas.loc[:, 'IFM_HS_encoded'] = le.fit_transform(df_africa_overseas['IFM_HS'])

# Select Features
features = ['Unit logistics costs ($/ton)', 'distance(km)', 'flow(tonne)', 'IFM_HS_encoded']
X = df_africa_overseas[features]

# Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
df_africa_overseas.loc[:, 'anomaly_score'] = iso_forest.fit_predict(X)

# Filter anomalies
anomalies2 = df_africa_overseas[df_africa_overseas['anomaly_score'] == -1]

# Show Top 10 Anomalies
print(anomalies2[['origin_ISO', 'destination_ISO', 'Unit logistics costs ($/ton)', 'IFM_HS', 'distance(km)', 'flow(tonne)']].sort_values(by='Unit logistics costs ($/ton)', ascending=False).head(10))

#%%
print(anomalies2[['origin_ISO', 'destination_ISO', 'Unit logistics costs ($/ton)', 'IFM_HS', 'distance(km)', 'flow(tonne)']].sort_values(by='flow(tonne)', ascending=False).head(10))

#%%
anomalies2[anomalies['flow(tonne)'] > 0.00000].shape

#%%
df_africa_overseas.loc[anomalies2.index].head(10)

# %%
valid_anomalies = anomalies2[anomalies2['flow(tonne)'] > 0.0]

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=valid_anomalies,
    x='distance(km)',
    y='Unit logistics costs ($/ton)',
    hue='IFM_HS',
    # style='ship_type'
)
plt.title('Anomalous Overseas Trades (flow > 0)')
plt.xlabel('Distance (km)')
plt.ylabel('Unit Logistics Cost ($/ton)')
plt.yscale('log')
plt.xscale('log')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# plt.savefig("overseas_anomalies_plot.png", dpi=300, bbox_inches='tight')
# %%
'''Now Africa as an importer when trade is continental and overseas'''
df_africa_overseas_import = df[(df['destination_continent'] == 'AF') & (df['trade_type'] == 'overseas')]
# %%
# Label Encoding
le = LabelEncoder()
df_africa_overseas_import.loc[:, 'IFM_HS_encoded'] = le.fit_transform(df_africa_overseas_import['IFM_HS'])

# Select Features
features = ['Unit logistics costs ($/ton)', 'distance(km)', 'flow(tonne)', 'IFM_HS_encoded']
X = df_africa_overseas_import[features]

# Isolation Forest
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
df_africa_overseas_import.loc[:, 'anomaly_score'] = iso_forest.fit_predict(X)

# Filter anomalies
anomalies_import = df_africa_overseas_import[df_africa_overseas_import['anomaly_score'] == -1]

# Show Top 10 Anomalies
print(anomalies_import[['origin_ISO', 'destination_ISO', 'Unit logistics costs ($/ton)', 'IFM_HS', 'distance(km)', 'flow(tonne)']].sort_values(by='Unit logistics costs ($/ton)', ascending=False).head(10))

#%%
print(anomalies_import[['origin_ISO', 'destination_ISO', 'Unit logistics costs ($/ton)', 'IFM_HS', 'distance(km)', 'flow(tonne)']].sort_values(by='flow(tonne)', ascending=False).head(10))

#%%
anomalies_import[anomalies_import['flow(tonne)'] > 0.00000].shape
# %%
valid_anomalies_import = anomalies_import[anomalies_import['flow(tonne)'] > 0.0]

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=valid_anomalies_import,
    x='distance(km)',
    y='Unit logistics costs ($/ton)',
    hue='IFM_HS',
    # style='ship_type'
)
plt.title('Anomalous Overseas Trades (flow > 0)')
plt.xlabel('Distance (km)')
plt.ylabel('Unit Logistics Cost ($/ton)')
plt.yscale('log')
plt.xscale('log')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
# %%
