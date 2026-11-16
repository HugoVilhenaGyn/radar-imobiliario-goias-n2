"""Dashboard N2 - Radar Imobiliario de Goias (IBGE). Gera os PNGs 01-04 do README. Executar da raiz do repositorio. O painel interativo e gerado por dashboard_apresentacao.py."""
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, plotly.graph_objects as go
from plotly.subplots import make_subplots
df = pd.read_csv("dados/dados_municipios_goias.csv", index_col=0); d = df[df.va_ind>=0].copy()
d["regiao"] = d.regiao_intermediaria.str.replace(" - .*","",regex=True).str.replace(" de Montes Belos","")
reg_cor = dict(zip(sorted(d.regiao.unique()), ["#1d5fd6","#0e7a8a","#7c3aed","#b3801e","#c0355a","#0e7a3f"]))
plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":10})
# 1) populacao x PIB per capita
fig,ax=plt.subplots(figsize=(8,5))
for r,g in d.groupby("regiao"): ax.scatter(g.pop_2024,g.pib_pc_2023,s=28,alpha=.75,color=reg_cor[r],label=r)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("População 2024 (escala log)"); ax.set_ylabel("PIB per capita 2023, R$ (escala log)")
ax.set_title("Municípios maiores não são os mais ricos por habitante"); ax.legend(fontsize=8,frameon=False)
for n in ["Goiânia","Catalão","Anápolis"]:
    r=d[d.nome==n].iloc[0]; ax.annotate(n,(r.pop_2024,r.pib_pc_2023),fontsize=8,xytext=(5,5),textcoords="offset points")
plt.tight_layout(); plt.savefig("figuras/01_populacao_x_pib_per_capita.png",dpi=150); plt.close()
# 2) estrutura economica por regiao
s=d.groupby("regiao")[["va_agro","va_ind","va_serv","va_adm"]].sum(); s=s.div(s.sum(axis=1),axis=0)*100
s.columns=["Agropecuária","Indústria","Serviços","Adm. pública"]; s=s.loc[s["Adm. pública"].sort_values().index]
ax=s.plot.barh(stacked=True,figsize=(8,4.5),color=["#0e7a3f","#b3801e","#1d5fd6","#c0355a"],width=.7)
ax.set_xlabel("% do valor adicionado (2021)"); ax.set_ylabel(""); ax.set_title("Estrutura econômica por região intermediária")
ax.legend(ncol=4,fontsize=8,frameon=False,loc="upper center",bbox_to_anchor=(.5,-.15)); plt.tight_layout(); plt.savefig("figuras/02_estrutura_economica_regiao.png",dpi=150); plt.close()
# 3) top 15 PIB per capita
t=d.nlargest(15,"pib_pc_2023").sort_values("pib_pc_2023")
fig,ax=plt.subplots(figsize=(8,5)); ax.barh(t.nome,t.pib_pc_2023/1000,color="#1d5fd6"); ax.set_xlabel("PIB per capita 2023 (R$ mil)")
ax.set_title("15 municípios com maior PIB per capita"); plt.tight_layout(); plt.savefig("figuras/03_top15_pib_per_capita.png",dpi=150); plt.close()
# 4) correlacoes
cols=["log_pop","share_agro","share_ind","share_serv","share_adm","moradores_por_dom","cresc_pop_24_25"]
c=d.assign(ln_pib_pc=np.log(d.pib_pc_2023))[cols+["ln_pib_pc"]].corr()
c.index=c.columns=["ln pop","% agro","% indústria","% serviços","% adm. púb.","moradores/dom","cresc. pop","ln PIB pc"]
fig,ax=plt.subplots(figsize=(7,5.8)); im=ax.imshow(c,cmap="RdBu_r",vmin=-1,vmax=1)
ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns,rotation=40,ha="right"); ax.set_yticks(range(len(c))); ax.set_yticklabels(c.index)
for i in range(len(c)):
    for j in range(len(c)): ax.text(j,i,f"{c.iloc[i,j]:.2f}",ha="center",va="center",fontsize=8)
ax.set_title("Correlação entre variáveis"); plt.colorbar(im,fraction=.04); plt.tight_layout(); plt.savefig("figuras/04_correlacao.png",dpi=150); plt.close()
# dashboard interativo
f=make_subplots(rows=2,cols=2,subplot_titles=("População x PIB per capita (log)","Estrutura econômica por região (% VA)","Top 15 PIB per capita (R$ mil)","Correlação com ln(PIB per capita)"),vertical_spacing=.16)
for r,g in d.groupby("regiao"):
    f.add_trace(go.Scatter(x=g.pop_2024,y=g.pib_pc_2023,mode="markers",name=r,text=g.nome,marker=dict(color=reg_cor[r],size=7,opacity=.8),hovertemplate="%{text}<br>pop %{x:,}<br>R$ %{y:,.0f}"),1,1)
for col,cor in zip(s.columns,["#0e7a3f","#b3801e","#1d5fd6","#c0355a"]):
    f.add_trace(go.Bar(y=s.index,x=s[col],name=col,orientation="h",marker_color=cor),1,2)
f.add_trace(go.Bar(y=t.nome,x=t.pib_pc_2023/1000,orientation="h",marker_color="#1d5fd6",showlegend=False),2,1)
cc=c["ln PIB pc"].drop("ln PIB pc").sort_values()
f.add_trace(go.Bar(y=cc.index,x=cc.values,orientation="h",marker_color=["#c0355a" if v<0 else "#0e7a8a" for v in cc],showlegend=False),2,2)
f.update_xaxes(type="log",row=1,col=1); f.update_yaxes(type="log",row=1,col=1); f.update_layout(barmode="stack",height=900,title="Radar Imobiliário de Goiás - Dashboard N2 (IBGE)",template="simple_white",legend=dict(orientation="h",y=-.08))
print("ok")
