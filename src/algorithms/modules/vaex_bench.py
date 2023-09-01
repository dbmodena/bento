from ast import arguments
import datetime
from operator import le
from socket import if_indextoname
import pandas as pd
import numpy as np
import psutil
import vaex as vx
import vaex.ml
from src.algorithms.utils import timing
from src.datasets.dataset import Dataset
from haversine import haversine
from src.algorithms.algorithm import AbstractAlgorithm


class VaexBench(AbstractAlgorithm):
    df_: vx.dataframe.DataFrame = None
    backup_: vx.dataframe.DataFrame = None
    ds_ : Dataset = None
    name = "vaex"
    pipeline = False
    
    def __init__(self, mem: str=None, cpu:int=None, pipeline: bool = False):
        self.mem_ = mem
        self.cpu_ = cpu
        self.pipeline = pipeline

    def set_df(self, df):
        """
        Sets the internal dataframe
        """
        self.df_ = df
        return self.df_

    def get_df(self):
        """
        Returns the internal dataframe
        """
        return self.df_

    def load_from_pandas(self, df):
        """
        Loads data from a pandas dataframe
        """
        return self.set_df(vx.from_pandas(df)) 

    @timing
    def get_pandas_df(self):
        """
        Returns the internal dataframe as a pandas dataframe
        """
        return self.df_.to_pandas_df()

    @timing
    def done(self):
        """
        Called when the execution of the algorithm is done
        """
        self.df_.execute()

    def get_memory_usage(self):
        """
        Return the current memory usage of this algorithm instance
        (in kilobytes), or None if this information is not available.
        """
        # return in kB for backwards compatibility
        return psutil.Process().memory_info().rss / 1024

    @timing
    def load_dataset(self, ds: Dataset, conn=None, **kwargs):
        """
        Load the provided dataframe
        """
        self.ds_ = ds
        path = ds.dataset_attribute.path
        format = ds.dataset_attribute.type

        if format == "csv":
            self.df_ = self.read_csv(path, **kwargs)
        elif format == "excel":
            self.df_ = self.read_excel(path, **kwargs)
        elif format == "hdf5":
            self.df_ = self.read_hdf5( path, **kwargs)
        elif format == "json":
            self.df_ = self.read_json( path, **kwargs)
        elif format == "parquet":
            self.df_ = self.read_parquet( path, **kwargs)
        elif format == "sql":
            self.df_ = self.read_sql( path, conn, **kwargs)
        elif format == "xml":
            self.df_ = self.read_xml( path, **kwargs)
        return self.df_

    def read_json(self, path, **kwargs):
        """
        :param path: path of the file to load
        :param kwargs: extra arguments
        Read a json file
        """
        return vx.from_json(path, **kwargs)

    def read_csv(self, path, **kwargs):
        """
        Read a csv file
        :param path: path of the file to load
        :param kwargs: extra arguments
        """
        return vx.from_csv_arrow(path, **kwargs)  
        
    def read_xml(self, path, **kwargs):
        """
        Read a xml file
        :param path: path of the file to load
        :param kwargs: extra arguments
        """
        return self.load_from_pandas(pd.read_xml(path, **kwargs))

    def read_excel(self, path, **kwargs):
        """
        Read an excel file
        :param path: path of the file to load
        :param kwargs: extra arguments
        """
        return self.load_from_pandas(pd.read_excel(path, **kwargs))

    def read_parquet(self, path, **kwargs):
        """
        Read a parquet file
        :param path: path of the file to load
        :param kwargs: extra arguments
        """
        return vx.open(path, **kwargs)
    
    def read_hdf5(self, path, **kwargs):
        """
        Read a hdf5 file
        :param path: path of the file to load
        :param kwargs: extra arguments
        """
        try:
            return vx.open(path, **kwargs)
        except:
            print('Dataset is not column based, it will be opened using pandas api')
            return vx.from_pandas(pd.read_hdf(path, **kwargs))
            

    def read_sql(self, query, conn, **kwargs):
        """
        Given a connection and a query
        creates a dataframe from the query output
        :param query query to run to get the data
        :param conn connection to a database
        :param kwargs: extra arguments
        """
        return self.load_from_pandas(pd.read_sql(query, conn, **kwargs))

    @timing
    def sort(self, columns, ascending=True):
        """
        Sort the dataframe by the provided columns
        Columns is a list of column names
        :param columns columns to use for sorting
        :param ascending if sets to False sorts in descending order (default True)
        """
        
        self.df_ = self.df_.sort(columns, ascending = ascending)
        
        return self.df_

    @timing
    def get_columns(self):
        """
        Return a list containing the names of the columns in the dataframe
        """
        return self.df_.get_column_names()

    @timing
    def is_unique(self, column):
        """
        Check the uniqueness of all values contained in the provided column_name
        :param column column to check
        """
        return len(self.df_.unique(column)) == self.df_.length_original()

    @timing
    def delete_columns(self, columns):
        """
        Delete the provided columns
        Columns is a list of column names
        :param columns columns to delete
        """

        column_mantain = [c for c in self.get_columns() if c not in columns]
        self.df_ = self.df_[column_mantain]
        return self.df_

    @timing
    def rename_columns(self, columns):
        """
        Rename the provided columns using the provided names
        Columns is a dictionary: {"column_name": "new_name"}
        :param columns a dictionary that contains for each column to rename the new name
        """

        assert type(columns) == dict, "Columns parameter must be a dict, formatted as {\"column_name\": \"new_name\"}"

        for el in columns.items():
            self.df_.rename(el[0],el[1])
        return self.df_

    @timing
    def merge_columns(self, columns, separator, name):
        """
        Create a new column with the provided name combining the two provided columns using the provided separator
        Columns is a list of two column names; separator and name are strings
        :param columns columns to merge
        :param separator separator to use
        :param name new column name
        """
        
        assert len(columns) == 2, "Merge is possible only with two columns"

        if type(columns) == str:
            columns = [columns]

        self.df_[name] = self.df_[columns[0]].astype(str) + separator + self.df_[columns[1]].astype(str)
        return self.df_

    @timing
    def fill_nan(self, value, columns=None, func=False):
        """
        Fill nan values in the dataframe with the provided value
        :param value: value to use for replacing null values
        :param columns: columns to fill, if empty all the dataframe is filled
        """
        if columns is None:
            columns = self.get_columns()
        
        if func:
            value = eval(value)
        
        self.df_ =   self.df_.fillna(value=value, column_names=columns)
        return self.df_

    @timing
    def one_hot_encoding(self, columns):
        """
        Performs one-hot encoding of the provided columns
        Columns is a list of column names
        :param columns columns to encode
        """
        if type(columns) == str:
            columns = [columns]
        
        one_hot_encoder = vaex.ml.OneHotEncoder(features=columns)
        self.df_ = one_hot_encoder.fit_transform(self.df_)
        return self.df_

    @timing
    def locate_null_values(self, column):
        """
        Returns the rows of the dataframe which contains
        null value in the provided column.
        :param column column to explore
        """
        if column == 'all':
            column = self.get_columns()
            
        return self.df_[self.df_[column] != self.df_[column]]

    @timing
    def search_by_pattern(self, column, pattern):
        """
        Returns the rows of the dataframe which
        match with the provided pattern
        on the provided column.
        Pattern could be a regular expression.
        :param column column to search on
        :param pattern pattern to search, string or regex
        """
        return self.df_[self.df_[column].str.contains(pattern)]

    @timing
    def locate_outliers(self, column, lower_quantile=0.1, upper_quantile=0.99, **kwargs):
        """
        Returns the rows of the dataframe that have values
        in the provided column lower or higher than the values
        of the lower/upper quantile.
        :param column column to search on
        :param lower_quantile lower quantile (default 0.1)
        :param upper_quantile upper quantile (default 0.99)
        """
        if column == 'all':
            column = self.get_columns()
        cols = [c for c in column if self.df_[c].dtype in ['int64', 'float64']]
        q_low = self.df_.percentile_approx(cols, (lower_quantile*100))
        q_hi  = self.df_.percentile_approx(cols, (upper_quantile*100))
        #print(q_low, q_hi)
        return self.df_[(self.df_[column] < q_low.max()) | (self.df_[column] > q_hi.min())]
    
    @timing
    def get_columns_types(self):
        """
        Returns a dictionary with column types
        """
        return self.df_.dtypes

    @timing
    def cast_columns_types(self, dtypes):
        """
        Cast the data types of the provided columns
        to the provided new data types.
        dtypes is a dictionary that provides for each
        column to cast the new data type.
        :param dtypes a dictionary that provides for ech column to cast the new datatype
        For example  {'col_name': 'int8'}
        """

        assert type(dtypes) == dict, "dtypes parameter must be a dict, formatted like {'col_name': 'type'} "
        for col, dtype in dtypes.items():
            if str(self.df_[col].dtype) == 'time32[s]':
                    self.df_[col] = self.df_[col].astype(str)
                    self.df_[col] = "1970-01-01 " + self.df_[col]
                    self.df_[col] = self.df_[col].astype('datetime64')
            
            elif str(self.df_[col].dtype) == dtype:
                continue
            else:
                self.df_[col] = self.df_[col].astype(dtype)
        return self.df_

    @timing
    #MANCANO I PERCENTILI
    def get_stats(self):
        """
        Returns dataframe statistics.
        Only for numeric columns.
        Min value, max value, average value, standard deviation, and standard quantiles.
        """
        df_copy = self.df_.copy()
        for c in self.get_columns():
            if str(df_copy[c].dtype) in {'date32[day]', 'time32[s]'} :
                df_copy[c] = df_copy[c].astype(str)
        
        
        return df_copy.describe(strings=False)

    #SOLUTION NOT FOUND
    @timing
    def find_mismatched_dtypes(self):
        """
        Returns, if exists, a list of columns with mismatched data types.
        For example, a column with string dtypes that contains only integer values.
        For every columns the list contain an object with three keys:
         - Col: name of the column
         - current_dtype: current data type
         - suggested_dtype: suggested data type
        """
        pass

    @timing
    def check_allowed_char(self, column, pattern):
        """
        Return true if all the values of the provided column
        follow the provided pattern.
        For example, if the pattern [a-z] is provided the string
        'ciao' will return true, the string 'ciao123' will return false.
        :param column column to check
        :param pattern pattern to use
        """
        return self.df_[column].str.contains(pattern)

    #Vaex non contiene questa funzione.
    #Soluzione proposta dagli sviluppatori di vaex non performante
    # L'esecuzione è in-memory, quindi tutti i benefici di vaex diventano inutili
    # https://github.com/vaexio/vaex/pull/1623
    # Nel caso di specie, non potrà essere applicato perchè il count supera il limite di storage int64 python
    # def drop_duplicates(self, columns=None):
    #     """
    #     Drop duplicate rows.
    #     """
        
    #     self.df_ = vx.from_pandas(self.df_.to_pandas_df().drop_duplicates())
    #     return self.df_
    @timing
    def drop_duplicates(self, columns=None):
        if columns is None:
            columns = self.get_columns()
        # This is a non trivial problem actually and we do not have a official implementation of this yet.
        try:
            columns = [self.df_[c] for c in self.df_.get_column_names()]
            self.df_['hashed'] = self.df_.apply(lambda *row: hash(str(row)), arguments=columns)
            unique_hashes = self.df_['hashed'].unique()
            self.df_ = self.df_.filter(self.df_['hashed'].isin(unique_hashes))
            self.df_ = self.df_.drop('hashed')
        except Exception:
            print("Warning: drop_duplicates is not implemented for this backend, falling back to pandas")
            self.df_ = vx.from_pandas(self.df_.to_pandas_df().drop_duplicates(subset=columns))
            
        return self.df_

    @timing
    def change_date_time_format(self, column, format):
        """
        Change the date/time format of the provided column
        according to the provided formatting string.
        column datatype must be datetime
        An example of str_date_time_format is '%m/%d/%Y'
        :param column column to format
        :param str_date_time_format datetime formatting string
        """
        #from datetime import datetime
        # converting col to pandas
        #pandas_df = self.df_.to_pandas_df()
        # converting col to datetime
        #print("Warning: change_date_time_format is not implemented for this backend, falling back to pandas")
        
        # column_values = self.df_[column].astype(str).values
        # formatted_values = []
        # for v in column_values:
        #     if v in ['NaT', 'nan', '', None]:
        #         formatted_values.append(None)
        #     else:
        #         formatted_values.append(datetime.datetime.strptime(str(v), format))
        # print(formatted_values)
        from dateutil import parser
        
        
        self.df_[column] = self.df_[column].apply(lambda x: parser.parse(str(x)) if x not in ['NaT', 'nan', '', None] else '')
        # port to string with the new format
        self.df_[column] = self.df_[column].apply(lambda x: x.strftime(format) if x not in ['NaT', 'nan', '', None] else '')
        #pandas_df[column] = pd.to_datetime(pandas_df[column], errors='ignore', format=format)
        #self.df_ = vx.from_pandas(pandas_df)
        return self.df_
        
        
    @timing
    def set_header_case(self, case):
        """
        Put dataframe headers in the provided case
        Supported cases: "lower", "upper", "title", "capitalize", "swapcase"
        :param case case format (lower, upper, title, capitalize, swapcase)
        """
        col = self.get_columns()
        if case == "lower":
            d = {el : el.lower() for el in col}
            self.rename_columns(d)
        elif case == "upper":
            d = {el : el.upper() for el in col}
            self.rename_columns(d)
        elif case == "title":
            d = {el : el.title() for el in col}
            self.rename_columns(d)
        elif case == "capitalize":
            d = {el : el.capitalize() for el in col}
            self.rename_columns(d)
        elif case == "swapcase":
            d = {el : el.swapcase() for el in col}
            self.rename_columns(d)
        return self.df_
                
    @timing
    def set_content_case(self, columns, case):
        """
        Put dataframe content in the provided case
        Supported cases: "lower", "upper", "title", "capitalize", "swapcase"
        (see definitions in pandas documentation)
        Columns is a list of two column names; empty list for the whole dataframe
        :param columns columns to modify
        :param case case format (lower, upper, title, capitalize, swapcase)
        """

        if type(columns) == str:
            columns = [columns]

        if len(columns) == 0:
            columns = self.get_columns()
        for column in columns:
            if case == "lower":
                self.df_[column] = self.df_[column].str.lower()
            elif case == "upper":
                self.df_[column] = self.df_[column].str.upper()
            elif case == "title":
                self.df_[column] = self.df_[column].str.title()
            elif case == "capitalize":
                self.df_[column] = self.df_[column].str.capitalize()
            elif case == "swapcase":
                self.df_[column] = self.df_[column].str.swapcase()
        return self.df_

    #Realizzata con virtual column.
    #Espressione valutata on-fly - non occupa memoria
    @timing
    def duplicate_columns(self, columns):
        """
        Duplicate the provided columns (add to the dataframe with "_duplicate" suffix)
        Columns is a list of column names
        :param columns columns to duplicate
        """

        if type(columns) == str:
            columns = [columns]

        for column in columns:
            self.df_.add_virtual_column(column + "_duplicate",self.df_.column)
        return self.df_

    #Operazioni di PIVOT - UNPIVOT non supportate in VAEX
    @timing
    def pivot(self, index, columns, values, aggfunc):
        """
        Define the lists of columns to be used as index, columns and values respectively,
        and the dictionary to aggregate ("sum", "mean", "count") the values for each column: {"col1": "sum"}
        :param index Column to use to make new frame’s index. If None, uses existing index.
        :param columns Column to use to make new frame’s columns.
        :param values  Column(s) to use for populating new frame’s values.
        :param aggfunc dictionary to aggregate ("sum", "mean", "count") the values for each column
               {"col1": "sum"}
        """
        import itertools
        try:
            df_copy = self.df_.copy()
            agg = {c: aggfunc(v) for c, v in itertools.product(columns, values)}
            if len(index) > 1:
                print('multi-index')
                df_copy['index'] = df_copy.apply(lambda *row: str(row), arguments=[df_copy[c] for c in index])
                index = 'index'

            return df_copy.groupby(by=index).agg(agg)

        except Exception:
            import pandas as pd
            print("Warning: pivot is not implemented for this backend, falling back to pandas")
            pivot = pd.pivot_table(self.df_.to_pandas_df(), index=index, columns=columns, values=values, aggfunc=aggfunc)
            return vx.from_pandas(pivot.reset_index())
    
    #Operazioni di PIVOT - UNPIVOT non supportate in VAEX
    @timing
    def unpivot(self, columns, var_name, val_name):
        """
        Define the list of columns to be used as values for the variable column,
        the name for variable columns and the one for value column_name
        """
        pass

    @timing
    def delete_empty_rows(self, columns):
        """
        Delete the rows with null values for all provided Columns
        Columns is a list of column names
        :param columns columns to check
        """
        if columns == 'all':
            columns = self.get_columns()
            
        self.df_ = self.df_.dropna(column_names=columns)
        return self.df_
  
    @timing
    def string_to_date(self, columns):  # sourcery skip: extract-method
        """
        Convert string column to datetime64 columns
        :param columns column to convert
        """
        if type(columns) == str:
            columns = [columns]

        map_month  = {"Gennaio" : "01", "Febbraio" : "02", "Marzo" : "03","Aprile" : "04", 
                        "Maggio" : "05", "Giugno" : "06", "Luglio" : "07", "Agosto" : "08", 
                        "Settembre" : "09", "Ottobre" : "10", "Novembre" : "11", "Dicembre" : "12"}

        map_day = {"1":"01","2":"02","3":"03","4":"04","5":"05","6":"06","7":"07","8":"08","9":"09"}

        for col in columns:
            
            self.split(col, " ", 2,["day", "month" , "year"])

            self.set_content_case(['month'], 'title')

            self.df_ = self.replace(["month"], map_month,0,False)
            self.df_ = self.replace(["day"], map_day,0,False)
            self.df_[col] = (self.df_["year"] + "-" + self.df_["month"] + "-" + self.df_["day"]).astype("datetime64").dt.date

            self.delete_columns(["day", "month", "year"])
        
        return self.df_

    @timing
    def split(self, column, sep, splits, col_names):
        """
        Split the provided column into splits + 1 columns named after col_names
        using the provided sep string as separator
        Col_names is a list of column names
        :param column column to split
        :param sep separator
        :param splits number of splits, limit the number of splits
        :param col_names name of the new columns
        """
        
        assert type(col_names) == list, "Columns parameter must be a list" 

        self.df_['split'] = self.df_[column].str.split(sep, splits)
        for el in col_names:
            self.df_[el] = (self.df_.func.split_list(self.df_['split'], col_names.index(el)))
        
        self.delete_columns('split')

        return self.df_

    @timing
    def strip(self, columns, chars):
        """
        Remove the characters appearing in chars at the beginning/end of the provided columns
        Columns is a list of column names
        :param columns columns to edit
        :param chars characters to remove
        """

        if type(columns) == str:
            columns = [columns]

        for column in columns:
            self.df_[column] = self.df_[column].str.strip(chars)
        return self.df_

    @timing
    def remove_diacritics(self, columns):
        """
        Remove diacritics from the provided columns
        Columns is a list of column names
        :param columns columns to edit
        """

        if type(columns) == str:
            columns = [columns]

        for column in columns:
            self.df_[column] = self.df_[column].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        return self.df_

    #Vaex has an index not modifiable
    #This index is useful in join or in group by operation   
    @timing
    def set_index(self, column):
        """
        Set the provided column as index
        :param column to use as index
        """
        pass
    
    @timing
    def change_num_format(self, formats:dict):
        """
        Round one ore more columns to a variable number of decimal places.
        formats is a dictionary with the column names as key and the number of decimal places as value.
        :param formats new column(s) format(s).
               E.g. {'col_name' : 2}
        """
        
        for el in formats.items():
            self.df_[el[0]] = self.df_[el[0]].round(el[1])
        return self.df_

    @timing
    def calc_column(self, col_name, columns:list, f):
        """
        Calculate the new column col_name by applying
        the function f to the whole dataframe.
        :param col_name column on which apply the function
        :param f function to apply
        """
        if type(columns) == str:
            columns = [columns]

        if type(f) == str:
            f = eval(f)
            
        self.df_[col_name] = self.df_.apply(f, arguments=columns)
        return self.df_

    @timing
    def join(self, other, left_on=None, right_on=None, how='inner', **kwargs):
        # sourcery skip: extract-method
        """
        Joins current dataframe (left) with a new one (right).
        left_on/right_on are the keys on which perform the equijoin
        how is the type of join
        **kwargs: additional parameters
        The result is stored in the current dataframe.
        :param other dataframe to join
        :param left_on key of the current dataframe to use for join
        :param right_on key of the other dataframe to use for join
        :param how type of join (inner, left, right, outer)
        :param kwargs extra parameters
        """
        assert how != 'outer', "Outer join is not supported"
        if type(left_on) == str:
            self.df_.join(other, left_on=left_on, right_on=right_on, how=how, inplace=True)
        if (type(left_on) == list) and (type(right_on) == list):
            assert len(left_on) == len(right_on), "Left and right keys must have the same length"
            self.df_['mergingcol'] = self.df_.apply(lambda *x: f'{x}', arguments=[self.df_[c] for c in left_on])
            other['mergingcol'] = other.apply(lambda *x: f'{x}', arguments=[other[c] for c in right_on])
            self.delete_columns(left_on)
            self.df_.join(other, left_on='mergingcol', right_on='mergingcol', how=how, inplace=True)
            self.df_= self.delete_columns(['mergingcol'])
            
        return self.df_

    @timing
    def groupby(self, columns, f):
        """
        Aggregate the dataframe by the provided columns
        then applies the function f on every group
        :param columns columns to use for group by
        :param f aggregation function
        """
        
        return  self.df_.groupby(columns, agg=f)

    @timing
    def categorical_encoding(self, columns):
        """
        See label encoding / ordinal encoding by sklearn
        Convert the categorical values in these columns into numerical values
        Columns is a list of column names
        :param columns columns to encode
        """
        #label_encoder = vaex.ml.LabelEncoder(features=columns)
        #self.df_ = label_encoder.fit_transform(self.df_)
        for c in columns:
            self.df_ = self.df_.ordinal_encode(c)
        return self.df_

    @timing
    def sample_rows(self, frac, num):
        """
        Return a sample of the rows of the dataframe
        Frac is a boolean:
        - if true, num is the percentage of rows to be returned
        - if false, num is the exact number of rows to be returned
        :param frac percentage or exact number of samples to take
        :param num if set to True uses frac as a percentage, otherwise frac is used as a number
        """
        return self.df_.sample(frac=num/100) if frac else self.df_.sample(n=num)

    @timing
    def append(self, other, ignore_index=False):
        """
        Append the rows of another dataframe (other) at the end of the provided dataframe
        All columns are kept, eventually filled by nan
        Ignore index is a boolean: if true, reset row indices
        :param other other dataframe to append
        :param ignore_index if set to True reset row indices
        ignore_index is set to False by default: Vaex does not mantain index on rows
        """
        
        return self.df_.concat(other)

    @timing
    def replace(self, columns, to_replace, value, regex):
        """
        Replace all occurrences of to_replace (numeric, string, regex, list, dict) in the provided columns using the
        provided value
        Regex is a boolean: if true, to_replace is interpreted as a regex
        Columns is a list of column names
        :param columns columns on which apply the method
        :param to_replace value to search (could be a regex)
        :param value value to replace with
        :param regex if True means that to_replace is a regex
        """

        if type(columns) == str:
            columns = [columns]

        for col in columns:
            if type(to_replace) in [int, float]:
                self.df_[col] = self.df_.func.where(self.df_[col] == to_replace, value, self.df_[col])
            elif type(to_replace) == str:
                self.df_[col] = self.df_[col].str.replace(to_replace, value, regex=regex)
            elif type (to_replace) == dict:
                for k,v in to_replace.items():
                    self.df_[col] = self.df_.func.where(self.df_[col] == k, v, self.df_[col])
            elif type(to_replace) == list:
                for el in to_replace:
                    self.df_[col] = self.df_.func.where(self.df_[col] == el, value, self.df_[col])

        return self.df_

    @timing
    def edit(self, columns, func):
        """
        Edit the values of the cells in the provided columns using the provided expression
        Columns is a list of column names
        :param columns columns on which apply this method
        :param func function to apply
        """
        if type(columns) == str:
            columns = [columns]
        
        for col in columns:
            if type(func) == str:
                func = eval(func)
            
            self.df_[col] = self.df_[col].apply(func)
        return self.df_

    #Data are immutable in vaex
    @timing
    def set_value(self, index, column, value):
        """
        Set the cell identified by index and column to the provided value
        :param index row indices
        :param column column name
        :param value value to set
        """
        pass

    @timing
    def min_max_scaling(self, columns, min, max):
        """
        Independently scale the values in each provided column in the range (min, max)
        Columns is a list of column names
        :param columns columns on which apply this method
        :param min min value
        :param max max value
        """
        scaler = vaex.ml.MinMaxScaler(features=columns, feature_range=(min,max))
        return scaler.fit_transform(self.df_)

    @timing
    def round(self, columns, n):
        """
        Round the values in columns using n decimal places
        Columns is a list of column names
        :param columns columns on which apply this method
        :param n decimal places
        """

        if type(columns) == str:
            columns = [columns]

        for col in columns:
            self.df_[col] = self.df_[col].round(n)
        
        return self.df_

    @timing
    def get_duplicate_columns(self):
        """
        Return a list of duplicate columns, if exists.
        Duplicate columns are those which have same values for each row.
        """
        cols = self.get_columns()
        
        return [(cols[i], cols[j])
                for i in range(len(cols))
                for j in range(i + 1, len(cols))
                if self.df_[cols[i]] == self.df_[cols[j]]]
        
    @timing
    def to_csv(self, path=f"./pipeline_output/{name}_loan_output.csv", **kwargs):
        """
        Export the dataframe in a csv file.
        :param path path on which store the csv
        :param kwargs extra parameters
        """
        import os
        if not os.path.exists("./pipeline_output"):
            os.makedirs("./pipeline_output")
        
        self.df_.export(f"./pipeline_output/{self.name}_output.csv", progress=True)
        
    @timing
    def to_parquet(self, path="./pipeline_output/vaex_loan_output.parquet", **kwargs):
        self.df_.export(path, progress=True)

    @timing
    def query(self, query, inplace=False):
        #Supponendo che la query sia ben definita con () per delimitare le priorità, questo metodo ritorna il risultato corretto
        # (self.df_.col1 > 1) & (self.df_.col2 < 10)
        #Il metodo filter valuta una espressione alla volta..
        """
        Queries the dataframe and returns the corresponding
        result set.
        :param query: a string with the query conditions, e.g. "(col1 > 1) & (col2 < 10)"
        :return: subset of the dataframe that correspond to the selection conditions
        Query must be well formatted.
        """
        if inplace:
            self.df_ = self.df_.filter(query)
            return self.df_

        return self.df_.filter(query)
    

    #METODO UTILE per la funzione append
    @timing
    def extract(self):
        '''Return a DataFrame containing only the filtered rows.

        {note_copy}

        The resulting DataFrame may be more efficient to work with when the original DataFrame is
        heavily filtered (contains just a small number of rows).

        If no filtering is applied, it returns a trimmed view.
        For the returned df, len(df) == df.length_original() == df.length_unfiltered()

        :rtype: DataFrame
        '''
        self.df_ = self.df_.trim()
        if self.df_.filtered:
            self.df_._push_down_filter()
            self.df_._invalidate_caches()
        return self.df_

    @vx.register_function(on_expression=True)
    def split_list(x, i):
        return np.array([el[i] for el in x], dtype=str)
    
    def backup(self):
        pass
    
    def drop_by_pattern(self, column, pattern):
        pass
    
    def force_execution(self):
        return self.df_.execute()
    
    def restore(self):
        pass
    
    def set_construtor_args(self, **kwargs):
        pass