import pandas as pd, numpy as np
from parse import mun,imed,inter,pop,pib,va,dom
for d,cs in [(pop,['cod','ano','pop']),(pib,['cod','ano','pib']),(va,['cod','ano','va_agro','va_ind','va_serv','va_adm']),(dom,['cod','ano','dom','moradores'])]:
    for c in cs: d[c]=pd.to_numeric(d[c],errors='coerce')
print("nulos:",{n:int(d.isna().sum().sum()) for n,d in dict(pop=pop,pib=pib,va=va,dom=dom).items()})
p=pop.pivot(index='cod',columns='ano',values='pop').add_prefix('pop_')
b=pib.pivot(index='cod',columns='ano',values='pib').add_prefix('pib_')
df=(mun.assign(cod=lambda x:x.cod.astype(int)).merge(imed.assign(cod_imediata=lambda x:x.cod_imediata),on='cod_imediata')
    .merge(inter,on='cod_inter').set_index('cod')
    .join(p).join(b).join(va.set_index('cod').drop(columns='ano')).join(dom.set_index('cod').drop(columns='ano')))
df=df.drop(columns=['cod_imediata','cod_inter'])
df['va_total']=df[['va_agro','va_ind','va_serv','va_adm']].sum(axis=1)
for s in ['agro','ind','serv','adm']: df[f'share_{s}']=df[f'va_{s}']/df.va_total
df['pib_pc_2023']=df.pib_2023*1000/df.pop_2024   # R$ por habitante (pop. 2024 como proxy de 2023)
df['pib_pc_2021']=df.pib_2021*1000/df.pop_2024
df['cresc_pop_24_25']=df.pop_2025/df.pop_2024-1
df['cresc_pib_21_23']=df.pib_2023/df.pib_2021-1
df['moradores_por_dom']=df.moradores/df.dom
df['hab_por_dom_2025']=df.pop_2025/df.dom
df['log_pop']=np.log(df.pop_2024)
print(df.isna().sum().sum(), df.shape); print(df.describe().T[['min','50%','max']].round(3))
df.to_csv('dados_municipios_goias.csv',encoding='utf-8-sig')
