"""Projeto Final N2 - Big Data e Data Science (FATESG 2026/02) - Hugo Vilhena
Modelagem preditiva: PIB per capita 2023 dos municipios de Goias (regressao)."""
import numpy as np, pandas as pd, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

df = pd.read_csv("dados_municipios_goias.csv", index_col=0)
# --- Tratamento: outlier estrutural (VA industrial negativo, usina hidreletrica) ---
outl = df.va_ind < 0
print("Removidos do treino:", df.loc[outl,"nome"].tolist())
d = df[~outl].copy()
d["share_total_check"] = d[["share_agro","share_ind","share_serv","share_adm"]].sum(axis=1)
assert np.allclose(d.share_total_check, 1)
d["y"] = np.log(d.pib_pc_2023)                      # alvo em log (distribuicao muito assimetrica)
d = d.join(pd.get_dummies(d.regiao_intermediaria, prefix="reg", dtype=float))
feats = ["log_pop","share_agro","share_ind","share_serv","share_adm","moradores_por_dom","cresc_pop_24_25"] + [c for c in d if c.startswith("reg_")]
X, y = d[feats], d.y
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
models = {
 "Ridge (linear)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
 "Random Forest": RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=42),
 "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42),
}
cv = KFold(5, shuffle=True, random_state=42)
rows, preds = [], {}
for n, m in models.items():
    s = cross_validate(m, Xtr, ytr, cv=cv, scoring=("r2","neg_mean_absolute_error"))
    m.fit(Xtr, ytr); p = m.predict(Xte); preds[n] = p
    rows.append(dict(modelo=n, r2_cv=s["test_r2"].mean(), r2_cv_std=s["test_r2"].std(),
        r2_teste=r2_score(yte,p), mae_log_teste=mean_absolute_error(yte,p),
        rmse_log_teste=mean_squared_error(yte,p)**.5,
        mae_reais_teste=mean_absolute_error(np.exp(yte),np.exp(p))))
res = pd.DataFrame(rows).round(3); print(res.to_string(index=False))
res.to_csv("resultados_modelos.csv", index=False, encoding="utf-8-sig")
# baseline ingenuo
base = np.full(len(yte), ytr.mean()); print("baseline R2:", round(r2_score(yte,base),3), "MAE R$:", round(mean_absolute_error(np.exp(yte),np.exp(base))))
best = res.sort_values("r2_teste").iloc[-1].modelo; print("melhor:", best)
# importancia por permutacao no melhor modelo
pi = permutation_importance(models[best], Xte, yte, n_repeats=30, random_state=42, scoring="r2")
imp = pd.Series(pi.importances_mean, index=feats).sort_values()
imp.to_csv("importancia_variaveis.csv", header=["importancia"]); print(imp.tail(6).round(3))
# --- figuras do modelo ---
fig, ax = plt.subplots(1,3, figsize=(15,4.4), sharex=True, sharey=True)
for a,(n,p) in zip(ax,preds.items()):
    a.scatter(yte,p,alpha=.7,color="#1d5fd6"); lim=[yte.min()-.1,yte.max()+.1]; a.plot(lim,lim,"k--",lw=1)
    a.set_title(f"{n}\nR² teste = {r2_score(yte,p):.2f}"); a.set_xlabel("ln(PIB per capita) real")
ax[0].set_ylabel("ln(PIB per capita) previsto"); plt.tight_layout(); plt.savefig("figuras/05_previsto_vs_real.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(7,4.2)); imp.tail(10).plot.barh(ax=ax,color="#0e7a8a")
ax.set_title(f"Importância das variáveis (permutação) - {best}"); ax.set_xlabel("queda no R² ao embaralhar a variável")
plt.tight_layout(); plt.savefig("figuras/06_importancia_variaveis.png", dpi=150); plt.close()
json.dump({"n_treino":len(Xtr),"n_teste":len(Xte),"melhor":best,"feats":feats}, open("meta.json","w"))
