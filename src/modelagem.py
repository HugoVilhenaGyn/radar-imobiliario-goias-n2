"""Projeto Final N2 - Big Data e Data Science (FATESG 2026/02)
Modelagem preditiva completa. Executar a partir da raiz do repositorio:  python src/modelagem.py

Parte A - Regressao: prever o PIB per capita 2023 (em log) dos municipios de Goias.
Parte B - Teste de robustez (ablacao): o quanto o resultado depende das participacoes setoriais.
Parte C - Classificacao complementar: municipio de ALTO x BAIXO PIB per capita.
"""
import json, warnings
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold, GridSearchCV, cross_val_score, learning_curve, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,
                             accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.inspection import permutation_importance, partial_dependence
warnings.filterwarnings("ignore")
SEED = 42
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
AZUL, VERDE, ROSA, OURO = "#1d5fd6", "#0e7a8a", "#c0355a", "#b3801e"

# ------------------------------------------------------------------ dados
df = pd.read_csv("dados/dados_municipios_goias.csv", index_col=0)
d = df[df.va_ind >= 0].copy()                      # exclui Cachoeira Dourada (VA industrial negativo)
d["y"] = np.log(d.pib_pc_2023)                     # alvo em log
priv = d[["va_agro", "va_ind", "va_serv"]].sum(axis=1)
for s in ["agro", "ind"]:                          # composicao so da economia PRIVADA (sem adm. publica)
    d[f"priv_{s}"] = d[f"va_{s}"] / priv
d = d.join(pd.get_dummies(d.regiao_intermediaria, prefix="reg", dtype=float))
REG = [c for c in d if c.startswith("reg_")]
BASE = ["log_pop", "moradores_por_dom", "cresc_pop_24_25"] + REG
FEATS = ["log_pop", "share_agro", "share_ind", "share_serv", "share_adm", "moradores_por_dom", "cresc_pop_24_25"] + REG
NOMES = {"share_adm": "% adm. pública", "share_agro": "% agro", "share_ind": "% indústria", "share_serv": "% serviços",
         "log_pop": "ln população", "moradores_por_dom": "moradores/domicílio", "cresc_pop_24_25": "cresc. população",
         "priv_agro": "% agro (só privado)", "priv_ind": "% indústria (só privado)"}
nome = lambda c: NOMES.get(c, c.replace("reg_", "região ").split(" - ")[0])

idx_tr, idx_te = train_test_split(d.index, test_size=0.2, random_state=SEED)
Xtr, Xte, ytr, yte = d.loc[idx_tr, FEATS], d.loc[idx_te, FEATS], d.loc[idx_tr, "y"], d.loc[idx_te, "y"]
print(f"treino={len(Xtr)}  teste={len(Xte)}  variaveis={len(FEATS)}")

# ------------------------------------------------------------------ PARTE A: regressao
cv5 = KFold(5, shuffle=True, random_state=SEED)
candidatos = {
    "Ridge": (make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [0.01, 0.1, 1, 10, 100]}),
    "KNN": (make_pipeline(StandardScaler(), KNeighborsRegressor()), {"kneighborsregressor__n_neighbors": [3, 5, 7, 11, 15], "kneighborsregressor__weights": ["uniform", "distance"]}),
    "Árvore de Decisão": (DecisionTreeRegressor(random_state=SEED), {"max_depth": [2, 3, 4, 6, None], "min_samples_leaf": [1, 3, 5, 10]}),
    "Random Forest": (RandomForestRegressor(random_state=SEED), {"n_estimators": [300], "max_depth": [None, 6, 10], "min_samples_leaf": [1, 3, 5], "max_features": [0.5, 1.0]}),
    "Gradient Boosting": (GradientBoostingRegressor(random_state=SEED), {"n_estimators": [150, 300], "learning_rate": [0.03, 0.1], "max_depth": [2, 3], "subsample": [0.8, 1.0]}),
}
modelos, params = {"Média (referência)": DummyRegressor().fit(Xtr, ytr)}, {}
for n, (est, grid) in candidatos.items():           # ajuste de hiperparametros so com o TREINO
    gs = GridSearchCV(est, grid, cv=cv5, scoring="r2", n_jobs=-1).fit(Xtr, ytr)
    modelos[n] = gs.best_estimator_; params[n] = {k.split("__")[-1]: v for k, v in gs.best_params_.items()}
    print(f"{n:18s} melhores parametros: {params[n]}")

rcv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED)   # 25 avaliacoes por modelo
linhas, cv_scores, prev = [], {}, pd.DataFrame({"nome": d.loc[idx_te, "nome"], "regiao": d.loc[idx_te, "regiao_intermediaria"], "real": np.exp(yte)})
for n, m in modelos.items():
    sc = cross_val_score(clone(m), Xtr, ytr, cv=rcv, scoring="r2", n_jobs=-1); cv_scores[n] = sc
    p = m.predict(Xte); prev[n] = np.exp(p)
    linhas.append(dict(modelo=n, r2_cv=sc.mean(), r2_cv_std=sc.std(), r2_teste=r2_score(yte, p),
                       mae_reais_teste=mean_absolute_error(np.exp(yte), np.exp(p)),
                       rmse_reais_teste=mean_squared_error(np.exp(yte), np.exp(p)) ** .5,
                       mape_teste=mean_absolute_percentage_error(np.exp(yte), np.exp(p)) * 100,
                       rmse_log_teste=mean_squared_error(yte, p) ** .5))
res = pd.DataFrame(linhas); res.to_csv("dados/resultados_modelos.csv", index=False, encoding="utf-8-sig")
print(res.round(3).to_string(index=False))
MELHOR = res[res.modelo != "Média (referência)"].sort_values("r2_cv").iloc[-1].modelo   # escolhido pela validacao cruzada, nao pelo teste
best = modelos[MELHOR]; pbest = best.predict(Xte); print("melhor (pela validacao cruzada):", MELHOR)
prev["erro_pct"] = (prev[MELHOR] / prev["real"] - 1) * 100
prev.round(1).to_csv("dados/previsoes_teste.csv", encoding="utf-8-sig")

# figuras da regressao
ordem = res.sort_values("r2_cv").modelo.tolist()
fig, ax = plt.subplots(figsize=(8, 4.2)); ax.boxplot([cv_scores[n] for n in ordem], vert=False, patch_artist=True, boxprops=dict(facecolor="#cfdcf7")); ax.set_yticks(range(1, len(ordem) + 1)); ax.set_yticklabels(ordem)
ax.scatter([res.set_index("modelo").r2_teste[n] for n in ordem], range(1, len(ordem) + 1), color=ROSA, zorder=3, label="R² no teste")
ax.set_xlabel("R² (validação cruzada 5×5 no treino)"); ax.set_title("Comparação dos modelos de regressão"); ax.legend(frameon=False); ax.set_xlim(-.3, 1)
plt.tight_layout(); plt.savefig("figuras/07_comparacao_modelos.png", dpi=150); plt.close()

tres = [n for n in ["Ridge", "Random Forest", "Gradient Boosting"]]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True, sharey=True)
for a, n in zip(ax, tres):
    p = modelos[n].predict(Xte); a.scatter(yte, p, alpha=.7, color=AZUL); lim = [yte.min() - .1, yte.max() + .1]; a.plot(lim, lim, "k--", lw=1)
    a.set_title(f"{n}\nR² teste = {r2_score(yte, p):.2f}"); a.set_xlabel("ln(PIB per capita) real")
ax[0].set_ylabel("ln(PIB per capita) previsto"); plt.tight_layout(); plt.savefig("figuras/05_previsto_vs_real.png", dpi=150); plt.close()

pi = permutation_importance(best, Xte, yte, n_repeats=30, random_state=SEED, scoring="r2")
imp = pd.Series(pi.importances_mean, index=[nome(c) for c in FEATS]).sort_values()
imp.to_csv("dados/importancia_variaveis.csv", header=["importancia"])
fig, ax = plt.subplots(figsize=(7, 4.2)); imp.tail(10).plot.barh(ax=ax, color=VERDE)
ax.set_title(f"Importância das variáveis (permutação) - {MELHOR}"); ax.set_xlabel("queda no R² ao embaralhar a variável"); plt.tight_layout(); plt.savefig("figuras/06_importancia_variaveis.png", dpi=150); plt.close()

resid = yte - pbest
fig, ax = plt.subplots(1, 2, figsize=(11, 4)); ax[0].scatter(pbest, resid, color=AZUL, alpha=.7); ax[0].axhline(0, color="k", lw=1, ls="--")
ax[0].set_xlabel("ln(PIB per capita) previsto"); ax[0].set_ylabel("resíduo (real − previsto)"); ax[0].set_title("Resíduos × previsto")
ax[1].hist(resid, bins=12, color=AZUL, alpha=.8); ax[1].set_xlabel("resíduo"); ax[1].set_title("Distribuição dos resíduos")
plt.suptitle(f"Análise de resíduos - {MELHOR} (teste)"); plt.tight_layout(); plt.savefig("figuras/08_residuos.png", dpi=150); plt.close()

tam, tr_s, va_s = learning_curve(clone(best), Xtr, ytr, cv=cv5, scoring="r2", train_sizes=np.linspace(.2, 1, 6), n_jobs=-1)
pd.DataFrame({"n_treino": tam, "r2_treino": tr_s.mean(1), "r2_validacao": va_s.mean(1)}).to_csv("dados/curva_aprendizado.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(tam, tr_s.mean(1), "o-", color=OURO, label="treino"); ax.plot(tam, va_s.mean(1), "o-", color=AZUL, label="validação")
ax.fill_between(tam, va_s.mean(1) - va_s.std(1), va_s.mean(1) + va_s.std(1), color=AZUL, alpha=.15)
ax.set_xlabel("municípios usados no treino"); ax.set_ylabel("R²"); ax.set_title(f"Curva de aprendizado - {MELHOR}"); ax.legend(frameon=False)
plt.tight_layout(); plt.savefig("figuras/09_curva_aprendizado.png", dpi=150); plt.close()

pdp = partial_dependence(best, Xtr, ["share_adm"], grid_resolution=40, method="brute")
gx, gy = pdp["grid_values"][0] * 100, np.exp(pdp["average"][0]) / 1000
pd.DataFrame({"share_adm_pct": gx, "pib_pc_previsto_mil": gy}).to_csv("dados/dependencia_parcial.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(gx, gy, color=ROSA, lw=2.5); ax.set_xlabel("% da administração pública no valor adicionado"); ax.set_ylabel("PIB per capita previsto (R$ mil)")
ax.set_title("Dependência parcial: efeito da administração pública"); plt.tight_layout(); plt.savefig("figuras/10_dependencia_parcial.png", dpi=150); plt.close()

err = prev.assign(abs_pct=prev.erro_pct.abs()); err["regiao"] = err.regiao.str.split(" - ").str[0]
por_reg = err.groupby("regiao").agg(municipios=("abs_pct", "size"), erro_medio_pct=("abs_pct", "mean")).round(1); print(por_reg)
por_reg.to_csv("dados/erro_por_regiao.csv", encoding="utf-8-sig")

# ------------------------------------------------------------------ PARTE B: robustez (ablacao)
conjuntos = {
    "Completo (com % dos 4 setores)": FEATS,
    "Só economia privada (sem adm. pública)": BASE + ["priv_agro", "priv_ind"],
    "Só porte, domicílios e região": BASE,
}
abl = []
for n, cols in conjuntos.items():
    m = clone(best); sc = cross_val_score(m, d.loc[idx_tr, cols], ytr, cv=rcv, scoring="r2", n_jobs=-1)
    m.fit(d.loc[idx_tr, cols], ytr); p = m.predict(d.loc[idx_te, cols])
    abl.append(dict(variaveis=n, r2_cv=sc.mean(), r2_cv_std=sc.std(), r2_teste=r2_score(yte, p), mae_reais_teste=mean_absolute_error(np.exp(yte), np.exp(p))))
abl = pd.DataFrame(abl); abl.to_csv("dados/ablacao.csv", index=False, encoding="utf-8-sig"); print(abl.round(3).to_string(index=False))
print("correlacao share_adm x ln(PIB pc 2021):", round(np.corrcoef(d.share_adm, np.log(d.pib_pc_2021))[0, 1], 3))
fig, ax = plt.subplots(figsize=(8, 3.6)); ax.barh(abl.variaveis[::-1], abl.r2_cv[::-1], xerr=abl.r2_cv_std[::-1], color=[OURO, VERDE, AZUL])
ax.set_xlabel("R² (validação cruzada 5×5)"); ax.set_title(f"Teste de robustez - {MELHOR} com menos variáveis"); ax.axvline(0, color="k", lw=.8)
plt.tight_layout(); plt.savefig("figuras/11_robustez_ablacao.png", dpi=150); plt.close()

# ------------------------------------------------------------------ PARTE C: classificacao
corte = np.exp(ytr).median()                        # mediana calculada SO no treino
ctr, cte = (np.exp(ytr) > corte).astype(int), (np.exp(yte) > corte).astype(int)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
clfs = {
    "Regressão Logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=SEED),
}
cl, rocs, cms = [], {}, {}
for n, m in clfs.items():
    sc = cross_validate(clone(m), Xtr, ctr, cv=skf, scoring=("accuracy", "f1", "roc_auc"))
    m.fit(Xtr, ctr); p = m.predict(Xte); pr = m.predict_proba(Xte)[:, 1]
    cl.append(dict(modelo=n, acuracia_cv=sc["test_accuracy"].mean(), f1_cv=sc["test_f1"].mean(), auc_cv=sc["test_roc_auc"].mean(),
                   acuracia=accuracy_score(cte, p), precisao=precision_score(cte, p), recall=recall_score(cte, p), f1=f1_score(cte, p), auc=roc_auc_score(cte, pr)))
    rocs[n] = roc_curve(cte, pr); cms[n] = confusion_matrix(cte, p)
cl = pd.DataFrame(cl); cl.to_csv("dados/resultados_classificacao.csv", index=False, encoding="utf-8-sig"); print(cl.round(3).to_string(index=False))
print(f"corte (mediana do treino): R$ {corte:,.0f} | classe majoritaria no teste: {max(cte.mean(), 1 - cte.mean()):.3f}")
MELHOR_C = cl.sort_values("auc_cv").iloc[-1].modelo
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
for a, (n, cm) in zip(ax, cms.items()):
    a.imshow(cm, cmap="Blues"); a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xticklabels(["baixo", "alto"]); a.set_yticklabels(["baixo", "alto"])
    for i in range(2):
        for j in range(2): a.text(j, i, cm[i, j], ha="center", va="center", fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "black")
    a.set_title(n); a.set_xlabel("previsto"); a.spines[:].set_visible(False)
ax[0].set_ylabel("real"); plt.suptitle("Matrizes de confusão (49 municípios do teste)"); plt.tight_layout(); plt.savefig("figuras/12_matriz_confusao.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(5.6, 4.6))
for (n, (fpr, tpr, _)), c in zip(rocs.items(), [OURO, VERDE, AZUL]): ax.plot(fpr, tpr, color=c, lw=2, label=f"{n} (AUC {cl.set_index('modelo').auc[n]:.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("taxa de falsos positivos"); ax.set_ylabel("taxa de verdadeiros positivos"); ax.set_title("Curva ROC (teste)"); ax.legend(frameon=False, fontsize=9)
plt.tight_layout(); plt.savefig("figuras/13_curva_roc.png", dpi=150); plt.close()

json.dump({"n_treino": len(Xtr), "n_teste": len(Xte), "melhor_regressao": MELHOR, "melhor_classificacao": MELHOR_C, "parametros": params,
           "corte_classificacao": float(corte), "cv_scores": {k: v.tolist() for k, v in cv_scores.items()},
           "confusao": {k: v.tolist() for k, v in cms.items()}, "roc": {k: {"fpr": v[0].tolist(), "tpr": v[1].tolist()} for k, v in rocs.items()},
           "corr_shareadm_pibpc2021": float(np.corrcoef(d.share_adm, np.log(d.pib_pc_2021))[0, 1]),
           "baseline_classe_majoritaria": float(max(cte.mean(), 1 - cte.mean()))},
          open("dados/meta_modelos.json", "w"), ensure_ascii=False, indent=1, default=str)
print("ok")
