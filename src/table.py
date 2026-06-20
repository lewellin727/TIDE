import pandas as pd

class Table:
    def __init__(self, table_name, df, score):
        self.table_name = table_name
        self.df = df
        self.score = score
        self.schema = None
        self.matched_column = None   
        self.predicate = None     


    def get_column(self, col_name):
        if col_name not in self.df.columns:
            return None
        dtype = self.df[col_name].dtype
        return {
            "col_type": "Numerical" if str(dtype).startswith(('int', 'float')) else "Categorical",
            "values": self.df[col_name]
        }
    

    def to_schema(self):
        return self.df.columns.tolist()

    def state_dict(self):
        return {
            'table_name': self.table_name,
            'score': self.score,
            'df': self.df.to_dict()  
        }
    
    @staticmethod
    def load(state_dict):
        table_name = state_dict['table_name']
        score = state_dict['score']
        df = pd.DataFrame.from_dict(state_dict['df'])
        return Table(table_name, df, score)