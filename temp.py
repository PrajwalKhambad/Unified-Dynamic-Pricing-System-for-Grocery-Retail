import joblib

model = joblib.load('fmcg_model.pkl')
print(model.get_xgb_params())
