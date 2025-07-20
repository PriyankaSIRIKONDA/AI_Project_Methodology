import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

model = joblib.load('models/rf_telco_churn.pkl')
df = pd.read_csv('data/processed/featured_telco.csv')
y = df.pop('Churn')

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
X = X_test

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print("Type of shap_values:", type(shap_values))
print("shap_values.shape:", shap_values.shape)

# Fix here: select class 1 SHAP values (samples, features)
shap_vals_to_use = shap_values[:, :, 1]
print("Selected shap_vals_to_use.shape:", shap_vals_to_use.shape)
assert shap_vals_to_use.shape == X.shape, "SHAP values shape mismatch!"

os.makedirs('reports/figures/shap', exist_ok=True)

plt.figure()
shap.summary_plot(shap_vals_to_use, X, show=False)
plt.savefig('reports/figures/shap/summary_plot.png', bbox_inches='tight')
plt.close()

plt.figure()
shap.summary_plot(shap_vals_to_use, X, plot_type='bar', show=False)
plt.savefig('reports/figures/shap/beeswarm_plot.png', bbox_inches='tight')
plt.close()

idx = 0
force_plot = shap.force_plot(
    explainer.expected_value[1],
    shap_vals_to_use[idx],
    X.iloc[idx],
    matplotlib=True,
    show=False
)
plt.savefig('reports/figures/shap/force_plot.png', bbox_inches='tight')
plt.close()

plt.figure()
shap.plots.waterfall(shap.Explanation(values=shap_vals_to_use[idx],
                                      base_values=explainer.expected_value[1],
                                      data=X.iloc[idx]))
plt.savefig('reports/figures/shap/waterfall_plot.png', bbox_inches='tight')
plt.close()

plt.figure()
shap.dependence_plot('MonthlyCharges', shap_vals_to_use, X, show=False)
plt.savefig('reports/figures/shap/dependence_plot_monthlycharges.png', bbox_inches='tight')
plt.close()

mean_shap = abs(shap_vals_to_use).mean(axis=0)
plt.figure()
pd.Series(mean_shap, index=X.columns).sort_values(ascending=False).plot.bar()
plt.title('Mean Absolute SHAP Value per Feature')
plt.tight_layout()
plt.savefig('reports/figures/shap/mean_shap_plot.png', bbox_inches='tight')
plt.close()

print("SHAP plots saved successfully!")
