import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
import base64
import rdkit
import useful_rdkit_utils
from PIL import Image
from padelpy import padeldescriptor


# HEADER
image = Image.open('logo.png')
st.image(image, use_column_width=True)
st.write("""
Beta version, currently under development ...\n
	~Naeem Abdul Ghafoor¹ & Ömür Baysal¹
###### ¹Department of Molecular Biology and Genetics, Mugla Sitki Kocman University, 48000 Turkey.
\n 
User Input:
""")

# BACKEND FUNCTIONS

def smiles2canonical(df, smiles_col):
	# Canonicalizes smiles
  canon_smiles = []
  smiles = df[smiles_col]
  for i in range(len(smiles)):
    try:
      m = rdkit.Chem.MolFromSmiles(smiles[i])
      csmi = rdkit.Chem.rdmolfiles.MolToSmiles(m)
    except:
      csmi = 'uncanonicalizable'
    canon_smiles.append(csmi)
  df['canonical_smiles'] = canon_smiles
  #print('Number of Uncanonicalizable Smiles dropped', df.loc[df['canonical_smiles'] == 'uncanonicalizable'].shape[0])
  df = df.loc[df['canonical_smiles'] != 'uncanonicalizable']
  df.reset_index(inplace=True, drop=True)
  df = df.drop([smiles_col], axis=1)
  return df 

	
def get_rdkp(df):
  # Calculates RDKit Properties
  RDKP = []
  smiles = df['canonical_smiles']
  rdkit_props = useful_rdkit_utils.RDKitProperties()
  for i in range(len(smiles)):
    m = rdkit.Chem.MolFromSmiles(smiles[i])
    feature = rdkit_props.calc_mol(m)
    RDKP.append(feature)
  df['RDKP'] = RDKP
  return df
		
def get_features(df, smiles_col, id_col):
  smiles2canonical(df, smiles_col)
  temp1 = get_rdkp(df)
  temp2 = temp1[[id_col, 'canonical_smiles', 'RDKP']]
  return temp2

def get_predictions(df, id_col):
  GY_RDKP = np.stack(df['RDKP'])
  GYM = pickle.load(open('GY_RDKP_SMOGN_ETsR_R2LOO845.pkl', 'rb'))
  GYMpred = GYM.predict(GY_RDKP)
  results = pd.DataFrame()
  results['IDs'] = df[id_col]
  results['canonical_smiles'] = df['canonical_smiles']
  results['pIC50 against DNA Gyrase'] = GYMpred
  return results

def convert_df(df):
	return df.to_csv().encode('utf-8')

# FRONTEND INPUTS

example = pd.read_csv('https://raw.githubusercontent.com/naeemmrz/ANBacPP/main/sample_input.csv')
st.sidebar.download_button(
   "Example CSV File",
   convert_df(example),
   "example_csv_file.csv",
   "text/csv",
   key='download-csv')

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.write(input_df)
else:
    st.write(""" **Awaiting CSV file to be uploaded**.""")
    st.stop()

id_col = st.sidebar.text_input('ID column header in the CSV file', 'SID/CID?')
if id_col in input_df.columns:
    id_column = id_col
else:
    st.write(""" **Please enter the correct header name for the ID column (case senstive)**.""")
    st.stop()

smiles_col = st.sidebar.text_input('SMILES column header in the CSV', 'smiles/smi?')
if smiles_col in input_df.columns:
    smiles_column = smiles_col
else:
    st.write(""" **Please enter the correct header name for the SMILES column (case senstive)**.""")
    st.stop()

st.sidebar.markdown("""
Please cite this work as "XYZ" 
""")

st.write(""" **Making Predictions on The Uploaded Data, Be Patient :)** .""")

# BACKEND PREDICTIONS
dF = get_features(input_df, smiles_column, id_column)
DF = get_predictions(dF, id_column)

st.write("""
# Predictions
""")
st.write(DF)

st.sidebar.download_button(
   "Press To Download Results",
   convert_df(DF),
   "results.csv",
   "text/csv",
   key='download-csv')
