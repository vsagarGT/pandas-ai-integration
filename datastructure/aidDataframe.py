import pandas as pd
import openai
from config import Config
import re
import os


class AIDataFrame(pd.DataFrame):
    def __init__(self, df, config=None, description=None, name=None) -> None:
        super().__init__(df)

        #initialize pandas dataframe
        self.pd_df = df
        self.config = Config()
        
        if len(df)>0:
            self.is_df_loaded = True
        else:
            self.is_df_loaded = False

        #set the description
        self.description = description
        
        #set the config
        if config:
            self.config = config
        
        #set name
        self.name = name

        #setup cache
        self.cache = {}

    @property
    def col_count(self):
        if self.is_df_loaded:
            return len(list(self.pd_df.columns))
        
    @property
    def row_count(self):
        if self.is_df_loaded:
            return len(self.pd_df)
        
    @property
    def sample_head_csv(self):
        if self.is_df_loaded:
            return self.pd_df.head(5).to_csv()
        
    
    @property
    def metadata(self):
        return self.pd_df.info()
    
    def to_csv(self, file_path):
        self.pd_df.to_csv(file_path)
    

    def clear_cache(self):
        self.cache = {}
    
        
    def initialize_middleware(self):
        open_ai_key = self.config.get_open_ai_key()
        openai.api_key = open_ai_key
        self.openai_model = "gpt-3.5-turbo"
        return
    

    def create_query_prompt(self, query: str):
        prompt = f"I need you to write a python3.8 program for the following dataframe. \
            You are given the following pandas dataframe. \
            The dataframe has {self.col_count} columns. The columns are {list(self.columns)}. \
            The first 2 rows of data in the csv format are {self.iloc[0:2].to_csv()} .\
            Give me the python code for the following query: {query}.\
            Write this code in a function named 'pandas_query_function' and it should take the pandas dataframe as input. \
            Do not create a new dataframe. assume that it is given as input to the function.\
            Output of the function should be a string. Add the required imports for the function. \
            Print the output in the format query: result in the function.\
            Do not add any code for example usage to execute the function. Write only the function code.\
            The response should have only the python code and no additional text. \
            I repeat.. give the python code only for the function. NO ADDITIONAL CODE."
        prompt = re.sub(' +', ' ', prompt)
        return prompt
    
    def create_plot_prompt(self, plot_req: str):
        prompt = f"I need you to write a python3.8 program for the following dataframe. \
            You are given the following pandas dataframe. \
            The dataframe has {self.col_count} columns. The columns are {list(self.columns)}. \
            The first 2 rows of data in the csv format are {self.iloc[0:2].to_csv()} .\
            Give me the python code to create the following plot: {plot_req}.\
            Write this code in a function named 'pandas_plot_function' and it should take the pandas dataframe as input. \
            Do not create a new dataframe. assume that it is given as input to the function.\
            Save the output plot to a file named plot.png. do not return anything.\
            Add the required imports for the function. \
            Do not add any code for example usage to execute the function. Write only the function code.\
            The response should have only the python code and no additional text. \
            I repeat.. give the python code only for the function. NO ADDITIONAL CODE."
        prompt = re.sub(' +', ' ', prompt)
        return prompt
    
    def create_manipulation_prompt(self, manipulation: str): 
        prompt = f"I need you to write a python3.8 program for the following dataframe. \
            You are given the following pandas dataframe. \
            The dataframe has {self.col_count} columns. The columns are {list(self.columns)}. \
            The first 2 rows of data in the csv format are {self.iloc[0:2].to_csv()} .\
            Give me the python code to perform the following manipulation: {manipulation}.\
            Write this code in a function named 'pandas_manipulation_function' and it should take the pandas dataframe as input. \
            Do not create a new dataframe. assume that it is given as input to the function.\
            The output should be the dataframe after the manipulations are done.\
            Add the required imports for the function. \
            Do not add any code for example usage to execute the function. Write only the function code.\
            The response should have only the python code and no additional text. \
            I repeat.. give the python code only for the function. NO ADDITIONAL CODE."
        prompt = re.sub(' +', ' ', prompt)
        return prompt



    def execute_python(self, python_code, type: str):
        if type=="query":
            with open("tmp.py", "w+") as file:
                file.write(python_code)
            
            from tmp import pandas_query_function
            answer = pandas_query_function(self.pd_df)
            
            #delete file
            os.remove("tmp.py")

            return answer
        
        elif type == "plot":
            with open("tmp.py", "w+") as file:
                file.write(python_code)
            from tmp import pandas_plot_function
            pandas_plot_function(self.pd_df)

            #delete file
            os.remove("tmp.py")

            return "Your plot is stored in plot.png"
        
        elif type == "manipulation":
            with open("tmp.py", "w+") as file:
                file.write(python_code)
            from tmp import pandas_manipulation_function
            output  = pandas_manipulation_function(self.pd_df)
            
            os.remove("tmp.py")
            
            return output
            

    def query_dataframe(self, query):
        prompt = self.create_query_prompt(query)
        
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", \
                                                  temperature=0.2, \
                                                  messages=[{"role": "user", "content": prompt}])
        
        python_code = completion.choices[0].message.content
        answer = self.execute_python(python_code, "query")

        return f"Question is {query} and Answer is {answer}"
    
    def plot_dataframe(self, query):
        prompt = self.create_plot_prompt(query)
        
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", \
                                                  temperature=0.2, \
                                                  messages=[{"role": "user", "content": prompt}])
        
        python_code = completion.choices[0].message.content
        self.execute_python(python_code, "plot")

        return f"please find the plot in the file plot.png"
    

    def manipulate_dataframe(self, query):
        prompt = self.create_manipulation_prompt(query)
        
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", \
                                                  temperature=0.2, \
                                                  messages=[{"role": "user", "content": prompt}])
        
        python_code = completion.choices[0].message.content
        print(python_code)
        answer = self.execute_python(python_code, "manipulation")

        return answer

    
    
    


