#%%
import pandas as pd
import numpy as np
import json
import math
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import pickle
#%%
pd.set_option('display.max_columns', None)
#%%
df = pd.read_csv("wb_data_all_fields_final.csv")
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
# %%
df_filtered.columns
# %%
df_filtered['Mode_name'].value_counts()

#%%
Q1 = df_filtered['Unit logistics costs ($/ton)'].quantile(0.25)
Q3 = df_filtered['Unit logistics costs ($/ton)'].quantile(0.75)
IQR = Q3 - Q1

# Step 2: Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Step 3: Filter out the outliers
clean_df = df_filtered[
    (df_filtered['Unit logistics costs ($/ton)'] >= lower_bound) &
    (df_filtered['Unit logistics costs ($/ton)'] <= upper_bound)
]

# Optional: summary
print(f"Original rows: {df_filtered.shape[0]:,}")
print(f"Cleaned rows: {clean_df.shape[0]:,}")
# %%

with open('centroid_json/country-centroids.json','r') as f:
    centroids = json.load(f)
iso_centroids = {c['alpha3']:(c['latitude'],c['longitude']) for c in centroids}


# 2. Vectorized haversine function
def haversine_np(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    dφ = np.radians(lat2 - lat1)
    dλ = np.radians(lon2 - lon1)
    a = np.sin(dφ/2)**2 + np.cos(φ1) * np.cos(φ2) * np.sin(dλ/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# Create route identifier
df['origin_destination'] = df['origin_ISO'] + "_" + df['destination_ISO']

# 3. Apply mapping
final_df = clean_df.copy()
# final_df = df.copy()
# Add origin_destination route identifier
final_df['origin_destination'] = final_df['origin_ISO'] + "_" + final_df['destination_ISO']

final_df['lat1'] = final_df['origin_ISO'].map(lambda x: iso_centroids.get(x, (np.nan, np.nan))[0])
final_df['lon1'] = final_df['origin_ISO'].map(lambda x: iso_centroids.get(x, (np.nan, np.nan))[1])
final_df['lat2'] = final_df['destination_ISO'].map(lambda x: iso_centroids.get(x, (np.nan, np.nan))[0])
final_df['lon2'] = final_df['destination_ISO'].map(lambda x: iso_centroids.get(x, (np.nan, np.nan))[1])

# 4. Compute and store distance
final_df['distance_km'] = haversine_np(final_df['lat1'], final_df['lon1'], final_df['lat2'], final_df['lon2'])
final_df.drop(columns=['lat1', 'lon1', 'lat2', 'lon2'], inplace=True)

#%%
# Mode ship_type per origin-destination pair
shiptype_lookup = final_df.groupby(['origin_ISO', 'destination_ISO'])['ship_type'] \
                          .agg(lambda x: x.mode().iloc[0]) \
                          .to_dict()

#%%
# Median distance for each origin-destination pair
distance_lookup = final_df.groupby(['origin_ISO', 'destination_ISO'])['distance(km)'] \
                          .median().to_dict()

#%%

X = final_df[['origin_destination','flow(tonne)','IFM_HS','distance(km)','ship_type']]

le = LabelEncoder()
y = le.fit_transform(final_df['Mode_name'])

# Split first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# # Target encode IFM_HS
# te = TargetEncoder(smoothing=0.3)
# X_tr['IFM_HS_encoded'] = te.fit_transform(X_tr['IFM_HS'], y_tr)
# X_te['IFM_HS_encoded'] = te.transform(X_te['IFM_HS'])
# #%%
# te_ship = TargetEncoder()
# X_tr['ship_type_encoded'] = te_ship.fit_transform(X_tr['ship_type'], y_tr)
# X_te['ship_type_encoded'] = te_ship.transform(X_te['ship_type'])

#%%
# Initialize encoders
enc_route = TargetEncoder()
enc_ship  = TargetEncoder()
enc_hs    = TargetEncoder()

# Encode origin_destination
X_train['origin_destination_encoded'] = enc_route.fit_transform(X_train[['origin_destination']], y_train)
X_test['origin_destination_encoded']  = enc_route.transform(X_test[['origin_destination']])

# Encode ship_type
X_train['ship_type_encoded'] = enc_ship.fit_transform(X_train[['ship_type']], y_train)
X_test['ship_type_encoded']  = enc_ship.transform(X_test[['ship_type']])

# Encode IFM_HS
X_train['IFM_HS_encoded'] = enc_hs.fit_transform(X_train[['IFM_HS']], y_train)
X_test['IFM_HS_encoded']  = enc_hs.transform(X_test[['IFM_HS']])
#%%
X_train_final = X_train[['distance(km)', 'flow(tonne)', 
                         'origin_destination_encoded', 
                         'ship_type_encoded', 
                         'IFM_HS_encoded']].copy()

X_test_final = X_test[['distance(km)', 'flow(tonne)', 
                       'origin_destination_encoded', 
                       'ship_type_encoded', 
                       'IFM_HS_encoded']].copy()

#%%
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05
)
model.fit(X_train_final, y_train)

#%%
y_pred = model.predict(X_test_final)
print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred)))
print(confusion_matrix(le.inverse_transform(y_test), le.inverse_transform(y_pred)))

#%%
# sample_weights = compute_sample_weight(class_weight='balanced', y=y_tr)
# # Fit model directly
# pipeline.fit(X_tr_final, y_tr, clf__sample_weight=sample_weights)
#%%
# 6. Evaluate
# pred = pipeline.predict(X_te_final)
# decoded_preds = le.inverse_transform(pred)

# print(pd.Series(decoded_preds).value_counts())

# print(classification_report(y_te,pred))
# print(confusion_matrix(y_te,pred))

#%%
# grid.best_params_
#%%
# 7. Save
joblib.dump(best,'mode_predictor_xgb.pkl')

#%%
# # Example input
# user_input = pd.DataFrame([{
#     'origin_ISO': 'KEN',
#     'destination_ISO': 'UGA',
#     'flow(tonne)': 0.06,
#     'IFM_HS': '1001'  # Example HS code: Wheat
# }])

def predict_mode(user_input_df):
    # Step 1: Compute distance
    user_input_df['lat1'] = user_input_df['origin_ISO'].map(lambda x: iso_centroids.get(x, (np.nan, np.nan))[0])
    user_input_df['lon1'] = user_input_df['origin_ISO'].map(lambda x: iso_centroids.get(x, (np.nan, np.nan))[1])
    user_input_df['lat2'] = user_input_df['destination_ISO'].map(lambda x: iso_centroids.get(x, (np.nan, np.nan))[0])
    user_input_df['lon2'] = user_input_df['destination_ISO'].map(lambda x: iso_centroids.get(x, (np.nan, np.nan))[1])
    # user_input_df['distance_km'] = haversine_np(user_input_df['lat1'], user_input_df['lon1'], user_input_df['lat2'], user_input_df['lon2'])
    user_input_df.drop(columns=['lat1', 'lon1', 'lat2', 'lon2'], inplace=True)

    # Step 2: Add origin_destination and ship_type from lookup
    user_input_df['origin_destination'] = user_input_df['origin_ISO'] + "_" + user_input_df['destination_ISO']
    user_input_df['ship_type'] = user_input_df.apply(
        lambda row: shiptype_lookup.get((row['origin_ISO'], row['destination_ISO']), 'Container'), axis=1
    )
    # Instead of Haversine, use lookup
    user_input_df['distance(km)'] = user_input_df.apply(
        lambda row: distance_lookup.get((row['origin_ISO'], row['destination_ISO']), np.nan), axis=1
    )

    # Step 3: Apply encoders
    user_input_df['IFM_HS_encoded'] = enc_hs.transform(user_input_df[['IFM_HS']])
    user_input_df['ship_type_encoded'] = enc_ship.transform(user_input_df[['ship_type']])
    user_input_df['origin_destination_encoded'] = enc_route.transform(user_input_df[['origin_destination']])

    # Step 4: Ensure correct column order to match training
    cols = ['distance(km)', 'flow(tonne)', 
                         'origin_destination_encoded', 
                         'ship_type_encoded', 
                         'IFM_HS_encoded']
    user_input_final = user_input_df[cols]

    # Step 4: Predict probabilities
    prob_array = model.predict_proba(user_input_final)[0]
    class_labels = le.inverse_transform(np.arange(len(prob_array)))

    # Step 5: Build response dictionary
    result = {label: f"{prob:.2%}" for label, prob in zip(class_labels, prob_array)}
    return result



result = predict_mode(pd.DataFrame([{
    'origin_ISO': 'KEN',
    'destination_ISO': 'UGA',
    'flow(tonne)': 0.06,
    'IFM_HS': '1001'
}]))
print("Predicted:", result)




# %%
# Save models and encoders
joblib.dump(model, r'final presentation\gradio\model.pkl')
#%%
joblib.dump(enc_hs, r'final presentation\gradio\enc_hs.pkl')
joblib.dump(enc_ship, r'final presentation\gradio\enc_ship.pkl')
joblib.dump(enc_route, r'final presentation\gradio\enc_route.pkl')

# Save lookup dicts
with open(r'final presentation\gradio\shiptype_lookup.pkl', 'wb') as f:
    pickle.dump(shiptype_lookup, f)

with open(r'final presentation\gradio\distance_lookup.pkl', 'wb') as f:
    pickle.dump(distance_lookup, f)
# %%
joblib.dump(le, r'final presentation\gradio\label_encoder.pkl')
# %%
# after you load your training DataFrame `df_train` with columns 
# ['origin_ISO','TransportMode'] you do:
modes_by_origin = (
    final_df
    .groupby("origin_ISO")["Mode_name"]
    .unique()
    .to_dict()
)

#%%
# e.g. modes_by_origin["USA"] == array(["Air","Sea","Road"], dtype=object
with open(r'final presentation\gradio\modes_by_origin.pkl',"wb") as f:
    pickle.dump(modes_by_origin, f)

# %%
