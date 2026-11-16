"""Painel de apresentacao (arquivo unico, funciona sem internet). Executar na pasta dados/ ou ajustar CSV."""
import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.offline as po
CSV = "dados/dados_municipios_goias.csv"
df = pd.read_csv(CSV, index_col=0); d = df[df.va_ind >= 0].copy()
d["regiao"] = d.regiao_intermediaria.str.replace(" - .*","",regex=True).str.replace(" de Montes Belos","")
res = pd.read_csv("dados/resultados_modelos.csv"); imp = pd.read_csv("dados/importancia_variaveis.csv", index_col=0).importancia
COR = dict(zip(sorted(d.regiao.unique()), ["#1d5fd6","#0e7a8a","#7c3aed","#b3801e","#c0355a","#0e7a3f"]))
SET = ["#0e7a3f","#b3801e","#1d5fd6","#c0355a"]
L = dict(template="simple_white", margin=dict(l=70,r=20,t=55,b=70),title_font=dict(size=15), height=380, font=dict(size=13), legend=dict(font=dict(size=11),orientation="h",y=-0.22,x=0))
def fig_html(f): return po.plot(f, include_plotlyjs=False, output_type="div", config={"displayModeBar":False,"responsive":True})
figs = []
# 1
f = go.Figure()
for r,g in d.groupby("regiao"):
    f.add_trace(go.Scatter(x=g.pop_2024,y=g.pib_pc_2023,mode="markers",name=r,text=g.nome,marker=dict(color=COR[r],size=8,opacity=.8),hovertemplate="%{text}<br>pop %{x:,}<br>R$ %{y:,.0f}<extra></extra>"))
f.update_xaxes(type="log",title="População 2024 (escala log)",tickvals=[1e3,1e4,1e5,1e6],ticktext=["1 mil","10 mil","100 mil","1 milhão"]); f.update_yaxes(type="log",title="PIB per capita 2023 (R$, escala log)",tickvals=[1e4,2e4,5e4,1e5,2e5],ticktext=["10 mil","20 mil","50 mil","100 mil","200 mil"])
f.update_layout(title="1. Tamanho não garante renda por habitante",**L); figs.append(f)
# 2
s = d.groupby("regiao")[["va_agro","va_ind","va_serv","va_adm"]].sum(); s = s.div(s.sum(axis=1),axis=0)*100; s = s.loc[s.va_adm.sort_values().index]
f = go.Figure([go.Bar(y=s.index,x=s[c],name=n,orientation="h",marker_color=k) for c,n,k in zip(s.columns,["Agropecuária","Indústria","Serviços","Adm. pública"],SET)])
f.update_layout(barmode="stack",title="2. Estrutura econômica por região (% do valor adicionado)",**L); f.update_xaxes(title="%"); figs.append(f)
# 3
t = d.nlargest(15,"pib_pc_2023").sort_values("pib_pc_2023")
f = go.Figure(go.Bar(y=t.nome,x=t.pib_pc_2023/1000,orientation="h",marker_color="#1d5fd6")); f.update_xaxes(title="R$ mil por habitante")
f.update_layout(title="3. Os 15 maiores PIB per capita",**{**L,"height":460}); figs.append(f)
# 4
cols = ["log_pop","share_agro","share_ind","share_serv","share_adm","moradores_por_dom","cresc_pop_24_25"]
nom = ["ln população","% agro","% indústria","% serviços","% adm. pública","moradores/domicílio","cresc. população"]
cc = d[cols].corrwith(np.log(d.pib_pc_2023)); cc.index = nom; cc = cc.sort_values()
f = go.Figure(go.Bar(y=cc.index,x=cc.values,orientation="h",marker_color=["#c0355a" if v<0 else "#0e7a8a" for v in cc],text=[f"{v:.2f}" for v in cc],textposition="outside"))
f.update_xaxes(range=[-1,.6],title="correlação com ln(PIB per capita)"); f.update_layout(title="4. O que se relaciona com o PIB per capita",**{**L,"height":460}); figs.append(f)
# ---------- R3: regressao (le os resultados gerados por src/modelagem.py)
import json
meta = json.load(open("dados/meta_modelos.json", encoding="utf-8")); MELHOR = meta["melhor_regressao"]
r = res.sort_values("r2_cv")
f = go.Figure([go.Bar(y=r.modelo,x=r.r2_cv,orientation="h",name="R² validação cruzada (5×5)",marker_color="#9bb7ee",error_x=dict(type="data",array=r.r2_cv_std,color="#4a5468",thickness=1)),
               go.Bar(y=r.modelo,x=r.r2_teste,orientation="h",name="R² teste",marker_color="#1d5fd6",text=[f"{v:.2f}".replace(".",",") for v in r.r2_teste],textposition="outside")])
f.update_layout(barmode="group",title="5. Regressão: 5 algoritmos e a referência (média)",**{**L,"height":460}); f.update_xaxes(range=[-.15,1.05],title="R²"); figs.append(f)
pv = pd.read_csv("dados/previsoes_teste.csv", index_col=0); tv=[1e4,2e4,5e4,1e5,2e5]; tx=["10 mil","20 mil","50 mil","100 mil","200 mil"]
lim = [pv[["real",MELHOR]].min().min()*.9, pv[["real",MELHOR]].max().max()*1.1]
f = go.Figure([go.Scatter(x=pv.real,y=pv[MELHOR],mode="markers",text=pv.nome,marker=dict(color="#1d5fd6",size=9,opacity=.8),name="municípios do teste",hovertemplate="%{text}<br>real R$ %{x:,.0f}<br>previsto R$ %{y:,.0f}<extra></extra>"),
               go.Scatter(x=lim,y=lim,mode="lines",line=dict(color="black",dash="dash"),name="previsão perfeita")])
f.update_xaxes(type="log",title="PIB per capita real (R$)",tickvals=tv,ticktext=tx); f.update_yaxes(type="log",title="PIB per capita previsto (R$)",tickvals=tv,ticktext=tx)
r2b = res.set_index("modelo").r2_teste[MELHOR]
f.update_layout(title=f"6. {MELHOR}: previsto × real nos 49 municípios do teste (R² = {r2b:.2f})".replace("0.","0,"),**{**L,"height":460}); figs.append(f)
im = imp.sort_values().tail(8)
f = go.Figure(go.Bar(y=im.index,x=im.values,orientation="h",marker_color="#0e7a8a")); f.update_xaxes(title="queda do R² ao embaralhar a variável")
f.update_layout(title=f"7. Variáveis mais importantes ({MELHOR})",**L); figs.append(f)
dp = pd.read_csv("dados/dependencia_parcial.csv")
f = go.Figure(go.Scatter(x=dp.share_adm_pct,y=dp.pib_pc_previsto_mil,mode="lines",line=dict(color="#c0355a",width=3))); f.update_xaxes(title="% da administração pública no valor adicionado"); f.update_yaxes(title="PIB per capita previsto (R$ mil)")
f.update_layout(title="8. Quanto mais administração pública, menor o PIB per capita previsto",**L); figs.append(f)
ca = pd.read_csv("dados/curva_aprendizado.csv")
f = go.Figure([go.Scatter(x=ca.n_treino,y=ca.r2_treino,mode="lines+markers",name="treino",line=dict(color="#b3801e")),go.Scatter(x=ca.n_treino,y=ca.r2_validacao,mode="lines+markers",name="validação",line=dict(color="#1d5fd6"))])
f.update_xaxes(title="municípios usados no treino"); f.update_yaxes(title="R²",range=[0,1.05]); f.update_layout(title="9. Curva de aprendizado: mais dados ainda ajudariam",**L); figs.append(f)
ab = pd.read_csv("dados/ablacao.csv").iloc[::-1]; ab["variaveis"] = [f"{a}: R² {v:.2f}".replace("0.","0,") for a,v in zip(ab.variaveis,ab.r2_cv)]
f = go.Figure(go.Bar(y=ab.variaveis,x=ab.r2_cv,orientation="h",marker_color=["#b3801e","#0e7a8a","#1d5fd6"],error_x=dict(type="data",array=ab.r2_cv_std,color="#4a5468",thickness=1)))
f.update_xaxes(title="R² (validação cruzada)",range=[-.45,1]); f.update_layout(title="10. Teste de robustez: o resultado depende da % de adm. pública",**{**L,"margin":dict(l=310,r=20,t=55,b=70)}); figs.append(f)
# ---------- R3: classificacao
cl = pd.read_csv("dados/resultados_classificacao.csv"); mets = [("acuracia","Acurácia"),("precisao","Precisão"),("recall","Recall"),("f1","F1"),("auc","AUC")]
f = go.Figure([go.Bar(x=[m[1] for m in mets],y=[row[m[0]] for m in mets],name=row.modelo,marker_color=c,text=[f"{row[m[0]]:.2f}".replace(".",",") for m in mets],textposition="outside") for (_,row),c in zip(cl.iterrows(),["#b3801e","#0e7a8a","#1d5fd6"])])
f.update_yaxes(range=[0,1.12]); f.update_layout(barmode="group",title="11. Classificação (alto × baixo PIB per capita): métricas no teste",**L); figs.append(f)
MC = meta["melhor_classificacao"]; cm = np.array(meta["confusao"][MC])
f = go.Figure(go.Heatmap(z=cm,x=["previsto: baixo","previsto: alto"],y=["real: baixo","real: alto"],colorscale="Blues",showscale=False,text=cm,texttemplate="%{text}",textfont=dict(size=26)))
f.update_yaxes(autorange="reversed"); f.update_layout(title=f"12. Matriz de confusão ({MC}, 49 municípios do teste)",**L); figs.append(f)
f = go.Figure([go.Scatter(x=v["fpr"],y=v["tpr"],mode="lines",name=k,line=dict(color=c,width=2.5)) for (k,v),c in zip(meta["roc"].items(),["#b3801e","#0e7a8a","#1d5fd6"])]+[go.Scatter(x=[0,1],y=[0,1],mode="lines",name="acaso",line=dict(color="black",dash="dash"))])
f.update_xaxes(title="taxa de falsos positivos"); f.update_yaxes(title="taxa de verdadeiros positivos"); f.update_layout(title="13. Curva ROC dos três classificadores (teste)",**L); figs.append(f)
best = res.set_index("modelo").loc[MELHOR]; clb = cl.set_index("modelo").loc[MC]; fmt = lambda v: f"{v:.2f}".replace(".",",")
kpis = [("246","municípios de Goiás (IBGE)"),("8","modelos treinados: 5 de regressão e 3 de classificação"),(fmt(best.r2_cv),f"R² do {MELHOR} na validação cruzada ({fmt(best.r2_teste)} no teste)"),(f"R$ {best.mae_reais_teste/1000:.1f} mil".replace(".",","),"erro médio por habitante (referência pela média: R$ 23,5 mil)"),(f"{clb.acuracia*100:.0f}%",f"acurácia da classificação alto × baixo ({MC})")]
tiles = "".join(f'<div class="k"><b>{a}</b><span>{b}</span></div>' for a,b in kpis)
cards = "".join(f'<section class="c">{fig_html(f)}</section>' for f in figs)
js = po.get_plotlyjs()
html = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Radar Imobiliário de Goiás — Painel N2</title>
<style>body{{margin:0;background:#f4f5f8;font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#1c2230}}
header{{background:#0d2f6e;color:#fff;padding:22px 32px}} h1{{margin:0;font-size:1.7rem}} header p{{margin:.3rem 0 0;opacity:.85}}
.kp{{display:flex;gap:14px;flex-wrap:wrap;padding:18px 32px 0}} .k{{background:#fff;border-radius:10px;padding:14px 18px;flex:1;min-width:170px;box-shadow:0 1px 3px #0001}}
.k b{{display:block;font-size:1.9rem;color:#1d5fd6}} .k span{{font-size:.82rem;color:#4a5468}}
h2{{margin:26px 32px 0;font-size:1.05rem;color:#4a5468;text-transform:uppercase;letter-spacing:.06em}}
.g{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,560px),1fr));gap:16px;padding:12px 32px 8px}} .c{{min-width:0;overflow:hidden;background:#fff;border-radius:10px;padding:8px;box-shadow:0 1px 3px #0001}}
footer{{padding:16px 32px 30px;font-size:.85rem;color:#4a5468}}</style></head><body>
<header><h1>Radar Imobiliário de Goiás</h1><p>Projeto Final N2 · Big Data e Data Science · FATESG 2026/02 · Hugo Flávio Vilhena, Ricardo César Santana e Ana Maria Lima Quartiero</p></header>
<div class="kp">{tiles}</div>
<h2>R2 — Dashboard de visualização</h2><div class="g">{"".join(f'<section class="c">{fig_html(f)}</section>' for f in figs[:4])}</div>
<h2>R3 — Machine Learning: regressão</h2><div class="g">{"".join(f'<section class="c">{fig_html(f)}</section>' for f in figs[4:10])}</div>
<h2>R3 — Machine Learning: classificação complementar</h2><div class="g">{"".join(f'<section class="c">{fig_html(f)}</section>' for f in figs[10:])}</div>
<footer>Fonte: IBGE (população, PIB municipal, valor adicionado e Censo 2022). Cachoeira Dourada fica fora dos gráficos e do modelo (valor adicionado industrial negativo, usina hidrelétrica). A participação da administração pública espelha em parte o nível de renda de 2021; sem ela o R² cai (gráfico 10). A base mede renda, não preço de imóvel.</footer>
<script>function rz(){{document.querySelectorAll(".js-plotly-plot").forEach(function(g){{Plotly.Plots.resize(g)}})}}window.addEventListener("load",function(){{rz();setTimeout(rz,300);setTimeout(rz,1200)}});window.addEventListener("resize",rz)</script></body></html>"""
# o plotly precisa estar carregado antes dos graficos: reordena colocando o script no inicio do body
html = html.replace(f"<script>{js}</script></body>","</body>").replace("<header>",f"<script>{js}</script><header>",1)
open("painel_apresentacao.html","w",encoding="utf-8").write(html); print(len(html)//1024,"KB")
