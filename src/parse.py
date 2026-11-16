import re, sys, pandas as pd
p="/mnt/user-data/uploads/02 - Faculdade FATESG/02.2 Big Data e Banco Nao Relacional/N1_BigData/N1_BigData_Hugo_Vilhena.sql"
t=open(p,encoding='utf-8').read()
out={}
for m in re.finditer(r"INSERT INTO (\w+)\s*(?:\([^)]*\))?\s*VALUES(.*?);\s*\n", t, re.S|re.I):
    name,body=m.groups()
    rows=re.findall(r"\(((?:[^()']|'(?:[^']|'')*')*)\)", body)
    data=[]
    for r in rows:
        vals=re.findall(r"'(?:[^']|'')*'|[^,]+", r)
        data.append([v.strip().strip("'").replace("''","'") for v in vals])
    out.setdefault(name,[]).extend(data)
for k,v in out.items():
    print(k,len(v),v[0])

def df(name,cols):
    return pd.DataFrame(out[name],columns=cols)
mun=df('municipio',['cod','nome','cod_imediata'])
imed=df('regiao_imediata',['cod_imediata','regiao_imediata','cod_inter'])
inter=df('regiao_intermediaria',['cod_inter','regiao_intermediaria'])
pop=df('populacao_estimada',['cod','ano','pop'])
pib=df('pib_municipal',['cod','ano','pib'])
va=df('valor_adicionado',['cod','ano','va_agro','va_ind','va_serv','va_adm'])
dom=df('domicilios_censo',['cod','ano','dom','moradores'])
if __name__=="__main__":
    print(pop.ano.value_counts().to_dict(), pib.ano.value_counts().to_dict(), va.ano.value_counts().to_dict())
    print(pop.head(4)); print(pib.head(4))
