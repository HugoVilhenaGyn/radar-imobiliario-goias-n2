"""Painel de apresentacao (arquivo unico, funciona sem internet). Executar na pasta dados/ ou ajustar CSV."""
import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.offline as po
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
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
# 5 modelos
f = go.Figure([go.Bar(x=res.modelo,y=res.r2_cv,name="R² validação cruzada",marker_color="#9bb7ee"),go.Bar(x=res.modelo,y=res.r2_teste,name="R² teste",marker_color="#1d5fd6",text=res.r2_teste.round(3),textposition="outside")])
f.update_layout(barmode="group",title="5. Comparação dos três modelos",**L); f.update_yaxes(range=[0,1],title="R²"); figs.append(f)
# 6 previsto x real (Gradient Boosting, mesma semente e mesmas variaveis do modelagem.py)
d["y"] = np.log(d.pib_pc_2023); d = d.join(pd.get_dummies(d.regiao_intermediaria,prefix="reg",dtype=float))
feats = cols + [c for c in d if c.startswith("reg_")]
Xtr,Xte,ytr,yte = train_test_split(d[feats],d.y,test_size=.2,random_state=42)
gb = GradientBoostingRegressor(n_estimators=300,learning_rate=.05,max_depth=3,subsample=.8,random_state=42).fit(Xtr,ytr); p = gb.predict(Xte)
lim = [min(yte.min(),p.min())-.1,max(yte.max(),p.max())+.1]
f = go.Figure([go.Scatter(x=np.exp(yte),y=np.exp(p),mode="markers",text=d.loc[yte.index,"nome"],marker=dict(color="#1d5fd6",size=9,opacity=.8),name="municípios do teste",hovertemplate="%{text}<br>real R$ %{x:,.0f}<br>previsto R$ %{y:,.0f}<extra></extra>"),
               go.Scatter(x=np.exp(lim),y=np.exp(lim),mode="lines",line=dict(color="black",dash="dash"),name="previsão perfeita")])
tv=[1e4,2e4,5e4,1e5,2e5]; tx=["10 mil","20 mil","50 mil","100 mil","200 mil"]; f.update_xaxes(type="log",title="PIB per capita real (R$)",tickvals=tv,ticktext=tx); f.update_yaxes(type="log",title="PIB per capita previsto (R$)",tickvals=tv,ticktext=tx)
f.update_layout(title=f"6. Gradient Boosting nos 49 municípios do teste (R² = {r2_score(yte,p):.2f})",**L); figs.append(f)
# 7 importancia
im = imp.sort_values().tail(8); im.index = [i.replace("share_adm","% adm. pública").replace("share_agro","% agro").replace("share_ind","% indústria").replace("share_serv","% serviços").replace("cresc_pop_24_25","cresc. população").replace("log_pop","ln população").replace("moradores_por_dom","moradores/dom.").replace("reg_","região ").replace(" - Águas Lindas de Goiás","").replace(" - Uruaçu","").replace(" - Iporá","").replace(" de Montes Belos","") for i in im.index]
f = go.Figure(go.Bar(y=im.index,x=im.values,orientation="h",marker_color="#0e7a8a")); f.update_xaxes(title="queda do R² ao embaralhar a variável")
f.update_layout(title="7. Variáveis mais importantes (Gradient Boosting)",**L); figs.append(f)
best = res.sort_values("r2_teste").iloc[-1]
kpis = [("246","municípios de Goiás (IBGE)"),("3","modelos comparados"),(f"{best.r2_teste:.2f}".replace(".",","),"R² do melhor (Gradient Boosting)"),(f"R$ {best.mae_reais_teste/1000:.1f} mil".replace(".",","),"erro médio por habitante (média ingênua: R$ 23,5 mil)"),("−0,84","correlação: % adm. pública × PIB per capita")]
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
<h2>R3 — Modelos de Machine Learning</h2><div class="g">{"".join(f'<section class="c">{fig_html(f)}</section>' for f in figs[4:])}</div>
<footer>Fonte: IBGE (população, PIB municipal, valor adicionado e Censo 2022). Cachoeira Dourada fica fora dos gráficos e do modelo (valor adicionado industrial negativo, usina hidrelétrica). A base mede renda, não preço de imóvel.</footer>
<script>function rz(){{document.querySelectorAll(".js-plotly-plot").forEach(function(g){{Plotly.Plots.resize(g)}})}}window.addEventListener("load",function(){{rz();setTimeout(rz,300);setTimeout(rz,1200)}});window.addEventListener("resize",rz)</script></body></html>"""
# o plotly precisa estar carregado antes dos graficos: reordena colocando o script no inicio do body
html = html.replace(f"<script>{js}</script></body>","</body>").replace("<header>",f"<script>{js}</script><header>",1)
open("painel_apresentacao.html","w",encoding="utf-8").write(html); print(len(html)//1024,"KB")
