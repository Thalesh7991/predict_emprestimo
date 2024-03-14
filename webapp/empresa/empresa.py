import pandas as pd
import numpy as np
import pickle
from sklearn.impute import KNNImputer

class PredictEmprestimo(object):
    def __init__(self):
        self.finalidade_emprestimo_encoding                     = pickle.load(open('../parameter/finalidade_emprestimo_encoding.pkl', 'rb'))
        self.grau_risco_emprestimo_encoding                     = pickle.load(open('../parameter/grau_risco_emprestimo_encoding.pkl', 'rb'))
        self.posse_casa_encoding                                = pickle.load(open('../parameter/posse_casa_encoding.pkl', 'rb'))
        self.historico_credito_scaler                           = pickle.load(open('../parameter/historico_credito_scaler.pkl', 'rb'))
        self.idade_scaler                                       = pickle.load(open('../parameter/idade_scaler.pkl', 'rb'))
        self.proporcao_emprestimo_historico_credito_scaler      = pickle.load(open('../parameter/proporcao_emprestimo_historico_credito_scaler.pkl', 'rb'))
        self.proporcao_emprestimo_idade_scaler                  = pickle.load(open('../parameter/proporcao_emprestimo_idade_scaler.pkl', 'rb'))
        self.proporcao_emprestimo_tempo_emprego_scaler          = pickle.load(open('../parameter/proporcao_emprestimo_tempo_emprego_scaler.pkl', 'rb'))
        self.proporcao_renda_emprestimo_scaler                  = pickle.load(open('../parameter/proporcao_renda_emprestimo_scaler.pkl', 'rb'))
        self.relacao_emprestimo_renda_scaler                    = pickle.load(open('../parameter/relacao_emprestimo_renda_scaler.pkl', 'rb'))
        self.renda_scaler                                       = pickle.load(open('../parameter/renda_scaler.pkl', 'rb'))
        self.taxa_juros_ajustada_renda_scaler                   = pickle.load(open('../parameter/taxa_juros_ajustada_renda_scaler.pkl', 'rb'))
        self.taxa_juros_emprestimo_scaler                       = pickle.load(open('../parameter/taxa_juros_emprestimo_scaler.pkl', 'rb'))
        self.tempo_emprego_scaler                               = pickle.load(open('../parameter/tempo_emprego_scaler.pkl', 'rb'))
        self.valor_emprestimo_scaler                            = pickle.load(open('../parameter/valor_emprestimo_scaler.pkl', 'rb'))
        self.imputer                                            = pickle.load(open('../parameter/imputer_knn.pkl', 'rb'))
    def data_cleaning(self, df1):
        num_attributes = df1.select_dtypes(include=['int64','float64'])
        cat_attributes = df1.select_dtypes(include=['object','category'])
        if num_attributes.isnull().any().any():
            df_imputed = pd.DataFrame(self.imputer.transform(num_attributes), columns=num_attributes.columns)
            df1 = pd.concat([df_imputed, cat_attributes], axis=1) 
            return df1
        else:
            return df1

    def feature_engineering(self, df2):
        # Taxa de Juros Ajustada pela Renda
        df2['taxa_juros_ajustada_renda'] = df2['taxa_juros_emprestimo'] / df2['renda']

        # Proporção do Valor do Empréstimo em Relação à Idade
        df2['proporcao_emprestimo_idade'] = df2['valor_emprestimo'] / df2['idade']

        # Proporção do Valor do Empréstimo em Relação ao Tempo de Emprego
        df2['tempo_emprego'] = np.where(df2['tempo_emprego'] == 0, 1, df2['tempo_emprego'])
        df2['proporcao_emprestimo_tempo_emprego'] = df2['valor_emprestimo'] / df2['tempo_emprego']


        # Proporção do Valor do Empréstimo em Relação à História de Crédito
        df2['proporcao_emprestimo_historico_credito'] = df2['valor_emprestimo'] / df2['historico_credito']

        # Proporção da Renda em Relação ao Valor do Empréstimo
        df2['proporcao_renda_emprestimo'] = df2['renda'] / df2['valor_emprestimo']

        return df2 
    
    def data_preparation(self, df5):
        df5['finalidade_emprestimo']                     = self.finalidade_emprestimo_encoding.transform(df5['finalidade_emprestimo'])              
        df5['grau_risco_emprestimo']                     =  self.grau_risco_emprestimo_encoding.transform(df5['grau_risco_emprestimo'])               
        df5['posse_casa']                                =  self.posse_casa_encoding.transform(df5['posse_casa'])                            
        df5['historico_credito']                         =  self.historico_credito_scaler.transform(df5[['historico_credito']].values)                       
        df5['idade']                                     = self.idade_scaler.transform(df5[['idade']].values)                                   
        df5['proporcao_emprestimo_historico_credito']    =  self.proporcao_emprestimo_historico_credito_scaler.transform(df5[['proporcao_emprestimo_historico_credito']].values)  
        df5['proporcao_emprestimo_idade']                =  self.proporcao_emprestimo_idade_scaler.transform(df5[['proporcao_emprestimo_idade']].values)              
        df5['proporcao_emprestimo_tempo_emprego']        =  self.proporcao_emprestimo_tempo_emprego_scaler.transform(df5[['proporcao_emprestimo_tempo_emprego']].values)      
        df5['proporcao_renda_emprestimo']                =  self.proporcao_renda_emprestimo_scaler.transform(df5[['proporcao_renda_emprestimo']].values)              
        df5['relacao_emprestimo_renda']                  =  self.relacao_emprestimo_renda_scaler.transform(df5[['relacao_emprestimo_renda']].values)                
        df5['renda']                                     = self.renda_scaler.transform(df5[['renda']].values)                                   
        df5['taxa_juros_ajustada_renda']                 = self.taxa_juros_ajustada_renda_scaler.transform(df5[['taxa_juros_ajustada_renda']].values)               
        df5['taxa_juros_emprestimo']                     = self.taxa_juros_emprestimo_scaler.transform(df5[['taxa_juros_emprestimo']].values)                   
        df5['tempo_emprego']                             = self.tempo_emprego_scaler.transform(df5[['tempo_emprego']].values)                           
        df5['valor_emprestimo']                          = self.valor_emprestimo_scaler.transform(df5[['valor_emprestimo']].values)

        chosen_features = ['proporcao_renda_emprestimo', 'relacao_emprestimo_renda',
                            'grau_risco_emprestimo', 'posse_casa', 'taxa_juros_emprestimo',
                            'taxa_juros_ajustada_renda', 'renda', 'finalidade_emprestimo',
                            'proporcao_emprestimo_tempo_emprego', 'proporcao_emprestimo_idade',
                            'tempo_emprego', 'valor_emprestimo',
                            'proporcao_emprestimo_historico_credito', 'idade', 'historico_credito']

        return df5[chosen_features]
    
    def get_predictions(self, model, test_data, original_data):
        pred = model.predict( test_data )
        original_data['prediction'] = pred
        
        return original_data.to_json( orient='records', date_format='iso' )
