#==============================================================================
# Autor(es): Juan Felipe Agudelo Rios                                         |
#                                                                             |
# Fecha creación: 29/08/2022                                                  |
# Fecha última modificación: 29/08/2022                                       |
#==============================================================================




"""
#==============================================================================
#                               Paquetes y Librerias                          |
#                                                                             |                                                                        |
#==============================================================================
"""
import regex as re
import pandas as pd
import numpy as np
import regex as re
import glob
import os
import pandas as pd
import datetime as dt
import unidecode
path = os.getcwd()
print(path)
#os.chdir("C:/Users/j.agudelo/Dropbox/Manejo de Datos")
import datetime as dt
import stata_setup
#stata_setup.config("C:\Program Files\Stata17","se")
stata_setup.config("/Applications/Stata/", "se")
from pystata import stata
import time
from sfi import Data
import math
from sklearn import preprocessing
import pandas as pd
import datetime as dt
import numpy as np 
import pandas as pd 
from sklearn import preprocessing
import numpy as np 
import math
pd.set_option('use_inf_as_na', True)
"""
#==============================================================================
#                               ICFES                                         |
#                                                                             |
#                                                                             |
#                                                                             |
#==============================================================================
"""


ICFES1=glob.glob("/Users/j.agudelo/Dropbox/ManejodeDatos/DATA/ICFES/*.txt")
ICFES2=glob.glob("/Users/j.agudelo/Dropbox/ManejodeDatos/DATA/ICFES/*.TXT")


ICFES=ICFES1+ICFES2


Main_ICFES20142=pd.DataFrame()
Main_ICFES20051=pd.DataFrame()

for ar in ICFES:
    periodo=ar.split("_")
    periodo=periodo[-1]
    periodo=periodo.split(".")
    periodo=periodo[0]
    periodo=int(periodo[:5])
    print(periodo)
    
    
    if periodo >=20142:
        stata.run(f"""
                  clear all
                  import delimited using {ar}, delimiters("¬") bindquote(nobind)
                  """)
        dataraw = Data.getAsDict()
        df = pd.DataFrame.from_dict(dataraw)
        if  2 > len(list(df.columns.values)):
                    stata.run(f"""
                              clear all
                              import delimited using {ar}, delimiters("|") bindquote(nobind)
                              """)
                    dataraw = Data.getAsDict()
                    df = pd.DataFrame.from_dict(dataraw)
                    df.columns= df.columns.str.strip().str.lower()
                    try:
                        df["punt_sociales_ciudadanas"]=df["punt_comp_ciudadana"]
                    except KeyError:
                        pass 
                    
                    df["SEM_TT"]=periodo    
                    df=df[["SEM_TT","estu_genero","fami_estratovivienda","punt_c_naturales",
                           "punt_lectura_critica","punt_matematicas","punt_sociales_ciudadanas",
                           "punt_ingles","estu_cod_reside_mcpio","cole_cod_mcpio_ubicacion",'estu_cod_mcpio_presentacion']]
            
                    
                    Main_ICFES20142=pd.concat([Main_ICFES20142,df])
                    Main_ICFES20142.reset_index(drop=True,inplace=True)
        else:                     
            df.columns= df.columns.str.strip().str.lower()
            try:
                df["punt_sociales_ciudadanas"]=df["punt_comp_ciudadana"]
            except KeyError:
                pass 
            
            df["SEM_TT"]=periodo           
            df=df[["SEM_TT","estu_genero","fami_estratovivienda","punt_c_naturales",
                   "punt_lectura_critica","punt_matematicas","punt_sociales_ciudadanas",
                   "punt_ingles","estu_cod_reside_mcpio","cole_cod_mcpio_ubicacion",'estu_cod_mcpio_presentacion']]
            
            Main_ICFES20142=pd.concat([Main_ICFES20142,df])
            Main_ICFES20142.reset_index(drop=True,inplace=True)
    
    
    elif 20081 <= periodo <=20141:
        
        stata.run(f"""
                  clear all
                  import delimited using {ar}, delimiters("¬") bindquote(nobind)
                  """)
        dataraw = Data.getAsDict()
        df = pd.DataFrame.from_dict(dataraw)
        if  2 > len(list(df.columns.values)):
                    stata.run(f"""
                              clear all
                              import delimited using {ar}, delimiters("|") bindquote(nobind)
                              """)
                    dataraw = Data.getAsDict()
                    df = pd.DataFrame.from_dict(dataraw)
                    df.columns= df.columns.str.strip().str.lower()
                    df["punt_comp_ciudadana"]=df["punt_sociales_ciudadanas"]
                    df["SEM_TT"]=periodo    
                    df.rename(columns = {'fami_estratovivienda':'fami_estrato_vivienda',
                                         'punt_ciencias_sociales':'punt_c_sociales',}, inplace = True)
                    
                    df=df[["SEM_TT","estu_genero","fami_estrato_vivienda","punt_lenguaje",
                           "punt_matematicas","punt_c_sociales","punt_filosofia","punt_biologia",
                           "punt_quimica","punt_fisica",'punt_ingles',
                           "cole_cod_mcpio_ubicacion"]]
                    
                    Main_ICFES20051=pd.concat([Main_ICFES20051,df])
                    Main_ICFES20051.reset_index(drop=True,inplace=True)
                    
        else:                     
            df.columns= df.columns.str.strip().str.lower()
            df["SEM_TT"]=periodo
            df.rename(columns = {'fami_estratovivienda':'fami_estrato_vivienda',
                                 'punt_ciencias_sociales':'punt_c_sociales'}, inplace = True)
            
            df=df[["SEM_TT","estu_genero","fami_estrato_vivienda","punt_lenguaje",
                       "punt_matematicas","punt_c_sociales","punt_filosofia","punt_biologia",
                       "punt_quimica","punt_fisica",'punt_ingles',"cole_cod_mcpio_ubicacion"]]
            Main_ICFES20051=pd.concat([Main_ICFES20051,df])
            Main_ICFES20051.reset_index(drop=True,inplace=True)
        
Main_ICFES20142["PUNT_TOTAL"]=(Main_ICFES20142["punt_c_naturales"]+Main_ICFES20142["punt_lectura_critica"]+
                           Main_ICFES20142["punt_matematicas"]+Main_ICFES20142["punt_sociales_ciudadanas"]+
                           Main_ICFES20142["punt_ingles"])

Main_ICFES20142=Main_ICFES20142.groupby(["SEM_TT","fami_estratovivienda","estu_genero"]).mean()
Main_ICFES20142.reset_index(drop=False, inplace= True)
Main_ICFES20051=Main_ICFES20142.groupby(["SEM_TT","fami_estratovivienda","estu_genero"]).mean()
Main_ICFES20051.reset_index(drop=False, inplace= True)
Main_ICFES20142.to_csv("/Users/j.agudelo/Dropbox/ManejodeDatos/CLEANDATA/ICEFES20142.csv")
Main_ICFES20051.to_csv("/Users/j.agudelo/Dropbox/ManejodeDatos/CLEANDATA/ICEFES20081.csv")






"""
#==============================================================================
#                               TRATAMIENTO                                   |
#                                                                             |
#                                                                             |
#                                                                             |
#==============================================================================
"""



PNFO=pd.read_csv("C:/Users/j.agudelo/Dropbox/ManejodeDatos/DATA/Proyecto_Nacional_de_Fibra__ptica.csv")
PCAV=pd.read_csv("C:/Users/j.agudelo/Dropbox/ManejodeDatos/DATA/Conectividad_de_Alta_Velocidad.csv")


PNFO["FECHA_OPERACION"] = pd.to_datetime(PNFO["FECHA OPERACION"])
PCAV["FECHA_OPERACION"] = pd.to_datetime(PCAV["FECHA INICIO OPERACION"])

PNFO=PNFO.dropna(subset=['FECHA_OPERACION'])
PCA=PCAV.dropna(subset=['FECHA_OPERACION'])

PNFO["ANO_TT"]=(PNFO["FECHA_OPERACION"].dt.year)
PCAV["ANO_TT"]=(PCAV["FECHA_OPERACION"].dt.year)


PNFO["SEM_TT"]= PNFO["FECHA_OPERACION"].dt.year.astype(str) +"" +np.where(PNFO["FECHA_OPERACION"].dt.quarter.gt(2),2,1).astype(str)
PCAV["SEM_TT"]= PCAV["FECHA_OPERACION"].dt.year.astype(str) +"" +np.where(PCAV["FECHA_OPERACION"].dt.quarter.gt(2),2,1).astype(str)


PNFO["codmpio"]=PNFO["MUNICIPIO_COD"]
PCAV["codmpio"]=PCAV["MUNICIPIO_COD"]


PNFO["ESTADO"]=PNFO["ESTADO ACTUAL"]

PNFO["NMUN"]=PNFO["MUNICIPIO_NOMBRE"]
PCAV["NMUN"]=PCAV["MUNICIPIO BENEFICIARIOS"]

PNFO=PNFO[["codmpio","ANO_TT","ESTADO","SEM_TT","NMUN"]]
PCAV=PCAV[["codmpio","ANO_TT","ESTADO","SEM_TT","NMUN"]]

Tratamiento=pd.concat([PNFO,PCAV])

def tratamiento(i):
    if i == "En Operaci�n":
        return 1
    elif i == "En Operaci�n - Fibra":
        return 1
    else:
        return 0

Tratamiento["TRATADO"]=Tratamiento["ESTADO"].apply(tratamiento)

def Periodo(df):
    if df["TRATADO"] ==1:
        return df["SEM_TT"]
    else: 
        return 0

Tratamiento["PERIODO_TRATADO"]=Tratamiento.apply(Periodo, axis=1)

Tratamiento = Tratamiento.groupby(["codmpio","PERIODO_TRATADO"]).min()

Tratamiento=Tratamiento.reset_index()

Tratamiento["DROP"]=0

Tratamiento.to_csv("C:/Users/j.agudelo/Dropbox/ManejodeDatos/CLEANDATA/Tratamiento.csv")


"""
#==============================================================================
#                            PEGUE Y DESAGREGACION                            |
#                                                                             |
#                                                                             |
#                                                                             |
#==============================================================================
"""



TRATAMIENTO=pd.read_csv("/Users/j.agudelo/Dropbox/ManejodeDatos/CLEANDATA/Tratamiento.csv")

ICFES_20081=pd.read_csv("/Users/j.agudelo/Dropbox/ManejodeDatos/CLEANDATA/ICEFES20081.csv")

ICFES_20142=pd.read_csv("/Users/j.agudelo/Dropbox/ManejodeDatos/CLEANDATA/ICEFES20142.csv")


ICFES_20081["codmpio"]=ICFES_20081["cole_cod_mcpio_ubicacion"]

ICFES_20142["codmpio"]=ICFES_20142["cole_cod_mcpio_ubicacion"]


#ICFES_20081=ICFES_20081[ICFES_20081["SEM_TT"]>20091]
#ICFES_20142=ICFES_20142[ICFES_20142["SEM_TT"]>20142]


ICFES_20081.rename(columns = {'SEM_TT':'Semestre',}, inplace = True)
ICFES_20142.rename(columns = {'SEM_TT':'Semestre',}, inplace = True)

ICFES_20081=ICFES_20081.merge(TRATAMIENTO,how='left', on=["codmpio"])
ICFES_20142=ICFES_20142.merge(TRATAMIENTO,how='left', on=["codmpio"])


#ICFES_20081=ICFES_20081[ICFES_20081["DROP"]==0] 
#ICFES_20142=ICFES_20142[ICFES_20142["DROP"]==0] 
#ICFES_20142=ICFES_20142[(ICFES_20142["PERIODO_TRATADO"]==0)|(ICFES_20142["PERIODO_TRATADO"]>20142)]

ICFES_20081['Semestre']=ICFES_20081['Semestre'].astype(str)
ICFES_20142['Semestre']=ICFES_20142['Semestre'].astype(str)

ICFES_20081['CALENDARIO'] = ICFES_20081['Semestre'].str[-1:]
ICFES_20142['CALENDARIO'] = ICFES_20142['Semestre'].str[-1:]



def nunca_tratados(i):
    if math.isnan(i) ==True:
        return 0
    else: 
        return i
    
ICFES_20142["ANO_TT"]=ICFES_20142["ANO_TT"].apply(nunca_tratados)
ICFES_20142["TRATADO"]=ICFES_20142["TRATADO"].apply(nunca_tratados)
ICFES_20142["SEM_TT"]=ICFES_20142["SEM_TT"].apply(nunca_tratados)

ICFES_20081["ANO_TT"]=ICFES_20081["ANO_TT"].apply(nunca_tratados)
ICFES_20081["TRATADO"]=ICFES_20081["TRATADO"].apply(nunca_tratados)
ICFES_20081["SEM_TT"]=ICFES_20081["SEM_TT"].apply(nunca_tratados)



def tonumber(i):
    i=str(i)
    i=i.replace(",",".")
    i=float(i)
    return i

for col in list(ICFES_20081.columns.values):
    if "punt_" in col: 
        ICFES_20081[col]=ICFES_20081[col].apply(tonumber)

for col in list(ICFES_20142.columns.values):
    if "punt_" in col: 
        ICFES_20142[col]=ICFES_20142[col].apply(tonumber)

def NOSPACE(i):
    i=str(i)
    i=i.replace(" ","")
    return i

ICFES_20081["fami_estrato_vivienda"]=ICFES_20081["fami_estrato_vivienda"].apply(NOSPACE)
ICFES_20142["fami_estratovivienda"]=ICFES_20142["fami_estratovivienda"].apply(NOSPACE)
"""
20081
"""




ICFES_20081["PUNT_TOTAL"]=(ICFES_20081["punt_lenguaje"]+ICFES_20081["punt_matematicas"]+
                           ICFES_20081["punt_c_sociales"]+ICFES_20081["punt_filosofia"]+
                           ICFES_20081["punt_biologia"]+ICFES_20081["punt_quimica"]+
                           ICFES_20081["punt_fisica"]+ICFES_20081["punt_ingles"])


LISTA_X8=["punt_lenguaje","punt_matematicas","punt_c_sociales","punt_filosofia","punt_biologia","punt_quimica",
          "punt_fisica","punt_ingles","PUNT_TOTAL"]

ICFES_20081_GENERO=ICFES_20081.copy()
ICFES_20081_GENERO=ICFES_20081_GENERO.groupby(by=["Semestre","estu_genero","codmpio","CALENDARIO"]).mean()
ICFES_20081_GENERO=ICFES_20081_GENERO.reset_index()

for x in LISTA_X8:
    ICFES_20081_GENERO["ln"+x]=np.log(ICFES_20081_GENERO[x])

for genero in list(ICFES_20081_GENERO["estu_genero"].unique()):
    globals()[f"ICFES_20081_{genero}"]=ICFES_20081_GENERO[ICFES_20081_GENERO["estu_genero"]==genero]
    

ICFES_20081_ESTRATO=ICFES_20081.copy()
ICFES_20081_ESTRATO=ICFES_20081_ESTRATO.groupby(by=["Semestre","fami_estrato_vivienda","codmpio","CALENDARIO"]).mean()
ICFES_20081_ESTRATO=ICFES_20081_ESTRATO.reset_index()

for x in LISTA_X8:
    ICFES_20081_ESTRATO["ln"+x]=np.log(ICFES_20081_ESTRATO[x])

for estrato in list(ICFES_20081_ESTRATO["fami_estrato_vivienda"].unique()):
    globals()[f"ICFES_20081_{estrato}"]=ICFES_20081_ESTRATO[ICFES_20081_ESTRATO["fami_estrato_vivienda"]==estrato]




ICFES_20081_general=ICFES_20081.copy()
ICFES_20081_general=ICFES_20081_general.groupby(by=["Semestre","codmpio","CALENDARIO"]).mean()
ICFES_20081_general=ICFES_20081_general.reset_index()


for x in LISTA_X8:
    ICFES_20081_general["ln"+x]=np.log(ICFES_20081_general[x])

"""
200142
"""



ICFES_20142["PUNT_TOTAL"]=(ICFES_20142["punt_c_naturales"]+ICFES_20142["punt_lectura_critica"]+
                           ICFES_20142["punt_matematicas"]+ICFES_20142["punt_sociales_ciudadanas"]+
                           ICFES_20142["punt_ingles"])

LISTA_X=["punt_c_naturales","punt_lectura_critica","punt_matematicas","punt_sociales_ciudadanas","punt_ingles","PUNT_TOTAL"]


ICFES_20142_GENERO=ICFES_20142.copy()
ICFES_20142_GENERO=ICFES_20142_GENERO.groupby(by=["Semestre","estu_genero","codmpio","CALENDARIO"]).mean()
ICFES_20142_GENERO=ICFES_20142_GENERO.reset_index()

for x in LISTA_X:
    ICFES_20142_GENERO["ln"+x]=np.log(ICFES_20142_GENERO[x])


for genero in list(ICFES_20142_GENERO["estu_genero"].unique()):
    globals()[f"ICFES_20142_{genero}"]=ICFES_20142_GENERO[ICFES_20142_GENERO["estu_genero"]==genero]
    
ICFES_20142_ESTRATO=ICFES_20142.copy()
ICFES_20142_ESTRATO=ICFES_20142_ESTRATO.groupby(by=["Semestre","fami_estratovivienda","codmpio","CALENDARIO"]).mean()
ICFES_20142_ESTRATO=ICFES_20142_ESTRATO.reset_index()

for x in LISTA_X:
    ICFES_20142_ESTRATO["ln"+x]=np.log(ICFES_20142_ESTRATO[x])


for estrato in list(ICFES_20142_ESTRATO["fami_estratovivienda"].unique()):
    globals()[f"ICFES_20142_{estrato}"]=ICFES_20142_ESTRATO[ICFES_20142_ESTRATO["fami_estratovivienda"]==estrato]


ICFES_20142_general=ICFES_20142.copy()
ICFES_20142_general=ICFES_20142_general.groupby(by=["Semestre","codmpio","CALENDARIO"]).mean()
ICFES_20142_general=ICFES_20142_general.reset_index()

for x in LISTA_X:
    ICFES_20142_general["ln"+x]=np.log(ICFES_20142_general[x])


"""
#==============================================================================
#                            ESTIMACION                                       |
#                                                                             |
#                                                                             |
#                                                                             |
#==============================================================================
"""





ICFES_20081.replace([np.inf, -np.inf], np.nan, inplace=True)
ICFES_20142.replace([np.inf, -np.inf], np.nan, inplace=True)

#==============================================================================
#                                                                             |
#    la mayoria de los tratados presentan las pruebas en el segundo periodo   |
#                                                                             |
#                                                                             |
#==============================================================================



LISTA_DF=list(globals())
import os
os.chdir("/Users/j.agudelo/Dropbox/ManejodeDatos/ESTIMACIONES/")

for x in LISTA_DF:
    try:
        name=str(x)
        x.to_stata(f"{name}.dta")
    except: 
        pass 



LISTA_LNX=["lnpunt_c_naturales","lnpunt_lectura_critica","lnpunt_matematicas","lnpunt_sociales_ciudadanas","lnpunt_ingles","lnPUNT_TOTAL"]

LISTA2014=[(ICFES_20142_Estrato1,'ICFES_20142_Estrato1'),(ICFES_20142_Estrato2,'ICFES_20142_Estrato2'),(ICFES_20142_Estrato3,"ICFES_20142_Estrato3"),
           (ICFES_20142_Estrato4,"ICFES_20142_Estrato4"),(ICFES_20142_Estrato5,'ICFES_20142_Estrato5'),(ICFES_20142_Estrato6,"ICFES_20142_Estrato6"),
           (ICFES_20142_F,"ICFES_20142_F"),(ICFES_20142_M,"ICFES_20142_M"),(ICFES_20142_general,"ICFES_20142_general")]


for BASE,NOMBREBASE in LISTA2014:
    for VAR in LISTA_LNX:
        try:
            stata.run("""
                      clear all 
                      """)
            CLEANVAR = re.sub('\W+','', VAR )
            CLEANBASE =re.sub('\W+','', NOMBREBASE )
            stata.pdataframe_to_data(BASE, force=True)
            stata.run("""
    
            cap log close
            set more off
            
    
            
            tab Semestre
            
            drop if CALENDARIO == "1"
            
            sort PERIODO_TRATADO Semestre
            
            egen SEMESTRE_T= group(PERIODO_TRATADO)
            replace SEMESTRE_T =. if PERIODO_TRATADO==0
            egen SEMESTRE= group(Semestre)
            
            gen rel_time=SEMESTRE-SEMESTRE_T
            replace rel_time=. if PERIODO_TRATADO==0 
            tab rel_time, gen(evt)
            
            tab SEMESTRE
            
            egen wanted = total(inrange(SEMESTRE, 1, 8)), by(codmpio)
            drop if wanted != 7
            
            order codmpio SEMESTRE TRATADO
            xtset codmpio SEMESTRE
            xtdes
            
             
            
            *LAGS                            
            forvalues x = 1/7{ 
            local j= 8-`x'
            ren evt`x' evt_l`j'
            cap label var evt_l`j' '-`j'' 
            }
                            
                            
            *FORWARG
                            
            forvalues x = 0/6{
            local j= 8+`x'
            ren evt`j' evt_f`x'
            cap label var evt_f`x' '`x''  
            }
            
            replace evt_l1=0
            rename evt_l1 ref 
            
            """)
            
            
            stata.run(f"""
            gen never=(TRATADO==0)
            
            
            gen gvar2 = cond(SEMESTRE_T==0, ., SEMESTRE_T)
            
            destring PERIODO_TRATADO Semestre, replace
            
            gen Dit = (Semestre >= PERIODO_TRATADO & PERIODO_TRATADO!=0)
            
            gen PRETRATADO = cond(Semestre<PERIODO_TRATADO, 0,TRATADO)
            
            sort codmpio PRETRATADO TRATADO PERIODO_TRATADO Semestre
    
            
            
            //  CODIGO DE LAS COMPLEMENTARIA PARA ESTUDIOS DE EVENTO SE UTILIZAN 
            // F STRINGS DE PYTHON PARA PODER HACER TODOS LOS POSIBLES CICLOS Y ESIMTAR 
            //  TODOS AQUELLOS EFECTOS HETEROGENES QUE SE PUEDAN. 
                            
                   
                            
                            * Globals importantes
                            
            global post 6 /* Número de periodos post sin contar el 0*/
            global pre 7 /* Número de periodos pre*/
            global ep event_plot
            global g0 "default_look"
            global g1 xla(-$pre (1)$post) /*global g1 xla(-5(1)5)*/
            global g2 xt("Periodos relativos al tratamiento")
            global g3 yt("Efecto causal")
            global g  $g1 $g2 $g3
            global t "together"
                            
            // Estimación con did_imputation de Borusyak et al. (2021)
                            
            did_imputation {VAR} codmpio SEMESTRE SEMESTRE_T,    horizons(0/$post) autosample pretrend($pre) minn(0)
            estimates store bjs // storing the estimates for later
            $ep bjs, $t $g0 graph_opt($g ti("BJS 21") name(gBJS, replace))
                                  
                            
                            
                            // Estimación con csdid of Callaway and Sant'Anna (2020)
            csdid {VAR}, ivar(codmpio) time(SEMESTRE) gvar(SEMESTRE_T) notyet
            estat event, estore(cs) // this produces and stores the estimates at the same time
            $ep cs, stub_lag(Tp#) stub_lead(Tm#) $t $g0 graph_opt($g ti("CS 20") name(gCS, replace))
                            
                            // Estimación con eventstudyinteract of Sun and Abraham (2020)
                            
            eventstudyinteract {VAR} evt_l* evt_f*,absorb(codmpio) cohort(SEMESTRE_T) control_cohort(never) vce(cluster i.codmpio)
            $ep e(b_iw)#e(V_iw), stub_lag(evt_f#) stub_lead(evt_l#) $t $g0 graph_opt($g ti("SA 20")  name(gSA, replace)) 
            matrix sa_b = e(b_iw) 
            matrix sa_v = e(V_iw)
                            
                            // Estimación por TWFE
               
            
            reghdfe {VAR} evt_l* ref evt_f*, nocon absorb(codmpio SEMESTRE) cluster(codmpio)  
            estimates store ols // saving the estimates for later
            $ep ols,  stub_lag(evt_f#) stub_lead(evt_l#) $t $g0 graph_opt($g ti("OLS") name(gOLS, replace))  
                            
                            
                            /* Descomposición de Goodman-Bacon */
            bacondecomp {VAR} PRETRATADO, ddetail gropt(legend(off) name(gGB, replace)) 
            
            
            
                            // Estimación con did_multiplegt of de Chaisemartin and D'Haultfoeuille (2020)
                            
            
            sort codmpio SEMESTRE 
    
            				
            did_multiplegt {VAR} codmpio SEMESTRE Dit,robust_dynamic breps(100) cluster(codmpio) dynamic(3) placebo(2)
            event_plot e(estimates)#e(variances), stub_lag(Effect_#) stub_lead(Placebo_#) $t $g0 graph_opt($g ti("CD 20") name(gCD, replace))
            matrix dcdh_b = e(estimates) // storing the estimates for later
            matrix dcdh_v = e(variances) 
            
            
            
                            
                            /* gY gBJS gCD gCS gSA gOLS gGB gG gCDLZ */
            graph combine gOLS gGB gBJS gCD gCS gSA, ycommon  name(combined, replace)
            graph export /Users/j.agudelo/Dropbox/ManejodeDatos/{CLEANBASE}{CLEANVAR}PLOTS.jpg, as(jpg) quality(100) name("combined")  replace 
            
                            
                            
                            // Combine all plots using the stored estimates
            event_plot /// 
            bjs  dcdh_b#dcdh_v cs sa_b#sa_v  ols, ///
                            	stub_lag( tau# Effect_# Tp# evt_f#  evt_f#) ///
                            	stub_lead( pre# Placebo_#  Tm# evt_l# evt_l# ) ///
                            	plottype(scatter) ciplottype(rcap) ///
                            	together perturb(-0.325(0.1)0.325) trimlead(5) noautolegend ///
                            	graph_opt(  ///
                            	title("Todos los estimadores de estudios de evento", size(med)) ///
                            	xtitle("Periodos relativos al evento", size(small)) ///
                            	ytitle("Efecto causal promedio estimado", size(small)) xlabel(-$pre(1)$post)  ///
                            	legend(order(1 "BJS" 3 "dCdH" ///
                            				5 "CS" 7 "SA" 9 "TWFE") rows(2) position(6) region(style(none))) ///
                            	/// the following lines replace default_look with something more elaborate
                            		xline(0, lcolor(gs8) lpattern(dash)) yline(0, lcolor(gs8)) graphregion(color(white)) bgcolor(white) ylabel(, angle(horizontal)) ///
                            	) 	///
                            	lag_opt1(msymbol(+) color(black)) lag_ci_opt1(color(black)) ///
                            	lag_opt2(msymbol(O) color(cranberry)) lag_ci_opt2(color(cranberry)) ///
                            	lag_opt3(msymbol(Dh) color(navy)) lag_ci_opt3(color(navy)) ///
                            	lag_opt4(msymbol(Th) color(forest_green)) lag_ci_opt4(color(forest_green)) ///
                            	lag_opt5(msymbol(Sh) color(dkorange)) lag_ci_opt5(color(dkorange))
            graph export /Users/j.agudelo/Dropbox/ManejodeDatos/{CLEANBASE}{CLEANVAR}COEFICIENTES.jpg,  as(jpg) quality(100) name("Graph") replace
            
            misstable sum  SEMESTRE_T
            
            clear all
            """)
            
            
        except:
            pass


"""
#==============================================================================
#              ESTADSITICAS   DESCRIPTIVAS                                    |
#                                                                             |
#                                                                             |
#                                                                             |
#==============================================================================
"""
stata.run("misstable sum  SEMESTRE_T")




stata.pdataframe_to_data(TRATAMIENTO, force=True)

stata.run("""
          
          sum
          
          label var TRATADO "# Municipios Tratados"
          label var SEM_TT "Semestre"
          
          bysort SEM_TT: outreg2 using TRATAMIENTO.tex, replace sum(log) eqkeep(N) keep(TRATADO) label
          
          """)

stata.pdataframe_to_data(ICFES_20142_general, force=True)

stata.run("""

          label var punt_c_naturales "Ciencias Naturales"
          label var punt_lectura_critica "Lectura Critica"
          label var punt_matematicas "Matematicas"
          label var punt_ingles "Ingles"
          label var punt_sociales_ciudadanas "Ciencias sociales"
          label var PUNT_TOTAL "Puntaje Total"

            estpost summarize punt_c_naturales punt_lectura_critica punt_matematicas punt_sociales_ciudadanas punt_ingles PUNT_TOTAL, listwise
            esttab, cells("mean sd min max") nomtitle nonumber
            esttab using DESCRIPTIVASICFES.tex , style(tex) cells("mean sd min max") nomtitle nonumber stats(N) label replace
            

""")


stata.pdataframe_to_data(ICFES_20142_GENERO, force=True)

stata.run("""

          label var punt_c_naturales "Ciencias Naturales"
          label var punt_lectura_critica "Lectura Critica"
          label var punt_matematicas "Matematicas"
          label var punt_ingles "Ingles"
          label var punt_sociales_ciudadanas "Ciencias sociales"
          label var PUNT_TOTAL "Puntaje Total"
          
            bysort estu_genero: eststo: summarize punt_c_naturales punt_lectura_critica punt_matematicas punt_sociales_ciudadanas punt_ingles PUNT_TOTAL
            esttab, cells("mean sd min max") nomtitle nonumber
            esttab using DESCRIPTIVASICFESSEXO.tex , style(tex) cells("mean sd min max") nomtitle nonumber stats(N) label replace
            

""")



stata.pdataframe_to_data(ICFES_20142_GENERO, force=True)

stata.run("""

          label var punt_c_naturales "Ciencias Naturales"
          label var punt_lectura_critica "Lectura Critica"
          label var punt_matematicas "Matematicas"
          label var punt_ingles "Ingles"
          label var punt_sociales_ciudadanas "Ciencias sociales"
          label var PUNT_TOTAL "Puntaje Total"
          

          

          bysort estu_genero: outreg2 using  DESCRIPTIVASICFESSEXO.tex, replace sum(log) eqkeep(N mean sd min max) keep(punt_c_naturales punt_lectura_critica punt_matematicas punt_sociales_ciudadanas punt_ingles PUNT_TOTAL) stats(N) label 
            
""")


ICFES_20142_GENERO["estu_genero"].value_counts()





stata.pdataframe_to_data(ICFES_20142_GENERO, force=True)

stata.run("""

          label var punt_c_naturales "Ciencias Naturales"
          label var punt_lectura_critica "Lectura Critica"
          label var punt_matematicas "Matematicas"
          label var punt_ingles "Ingles"
          label var punt_sociales_ciudadanas "Ciencias sociales"
          label var PUNT_TOTAL "Puntaje Total"

          bysort estu_genero: outreg2 using  DESCRIPTIVASICFESSEXO.tex, replace sum(log) eqkeep(N mean sd min max) keep(punt_c_naturales punt_lectura_critica punt_matematicas punt_sociales_ciudadanas punt_ingles PUNT_TOTAL) stats(N) label 
            
""")


ICFES_20142_GENERO["estu_genero"].value_counts()


stata.pdataframe_to_data(TRATAMIENTO, force=True)
stata.run("""
drop if SEM_TT<=20142
drop if TRATADO==0
sum
estpost tabstat codmpio, by(SEM_TT) statistics(count) columns(statistics) nototal listwise
esttab using Primer_Tratamiento.tex, main(count)  nostar unstack noobs nonote nomtitle nonumber replace
clear all 
""",quietly=False)



"""
#==============================================================================
#              TENDENCIAS   PARALELAS                                         |
#                                                                             |
#                                                                             |
#                                                                             |
#==============================================================================
"""



ICFES_20081_general=ICFES_20081_general[ICFES_20081_general["SEM_TT"]>=20142]
ICFES_20081_general=ICFES_20081_general[ICFES_20081_general["CALENDARIO"]=="2"]

ICFES_20081_general["SEM_TT"].value_counts()


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_palette('gist_rainbow')
sns.set_style("whitegrid")


for col in list(ICFES_20081_general.columns.values):
        sns.lineplot(data=ICFES_20081_general, x="Semestre", y=col, hue="SEM_TT")
        ytitulo=""
        for x in col:
            if "ln" in col:
                print(x)
            elif "y" ==x:
                print(x)
            elif "x" ==x:
                print(x)
            else: 
                ytitulo=ytitulo+" "+x

        plt.xlabel('Semestre')
        plt.ylabel(col) 
        plt.legend(title="Cohorte")
        plt.show()
        





















