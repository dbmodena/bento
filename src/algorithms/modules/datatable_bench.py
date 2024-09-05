from calendar import c
import datetime
import decimal
from unicodedata import name, numeric
import warnings

from polars import col
from pyspark.ml.util import W

warnings.filterwarnings('ignore')
import re
from typing import Union
from haversine import haversine
import pandas as pd
import datatable as dt
from datatable import f as F
import numpy as np
import h5py
from src.algorithms.utils import timing
from src.datasets.dataset import Dataset
from src.algorithms.algorithm import AbstractAlgorithm


class DataTableBench(AbstractAlgorithm):
    df_: Union[pd.DataFrame, pd.Series] = None
    backup_: Union[pd.DataFrame, pd.Series] = None
    ds_ : Dataset = None

    def __init__(self, name:str, mem: str = None, cpu: int = None, pipeline: bool = False):
        self.mem_ = mem
        self.cpu_ = cpu
        self.pipeline = pipeline
        self.name = name
        self.dataframes = {}

    def backup(self):
        """
        Creates a backup copy of the current dataframe
        """
        self.backup_ = self.df_.copy()

    def restore(self):
        """
        Replace the internal dataframe with the backup
        """
        self.df_ = self.backup_.copy()

    @timing
    def get_pandas_df(self):
        """
        Returns the internal dataframe as a pandas dataframe
        """
        return self.df_.to_pandas()

    @timing
    def load_from_pandas(self, df):
        """
        Loads data from a pandas dataframe
        """
        self.df_ = df

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
        elif format == "json":
            self.df_ = self.read_json(path, **kwargs)
        elif format == "parquet":
            self.df_ = self.read_parquet(path, **kwargs)
        elif format == "sql":
            self.df_ = self.read_sql(path, conn, **kwargs)
        elif format == "hdf5":
            self.df_ = self.read_hdf5(path, **kwargs)
        elif format == "xml":
            self.df_ = self.read_xml(path, **kwargs)
            
        return self.df_

    def read_sql(self, query, conn, **kwargs):
        """
        Given a connection and a query
        creates a dataframe from the query output
        """
        pass

    def read_json(self, path, **kwargs):
        """
        Read a json file
        """
        pass

    def read_csv(self, path, **kwargs):
        """
        Read a csv file
        """
        self.df_ = dt.fread(path, **kwargs)
        return self.df_
    
    def read_hdf5(self, path, **kwargs):
        """
        Given a connection and a query
        creates a dataframe from the query output
        """
        pass

    def read_xml(self, path, **kwargs):
        """
        Read a xml file
        """
        pass

    def read_excel(self, path, **kwargs):
        """
        Read an excel file
        """
        pass

    def read_parquet(self, path, **kwargs):
        """
        Read a parquet file
        """
        self.df_ = dt.fread(path, **kwargs)
        return self.df_

    @timing
    def sort(self, columns, ascending=True):
        """
        Sort the dataframe by the provided columns
        Columns is a list of column names
        """
        self.df_ = self.df_.sort(columns)
        return self.df_

    @timing
    def get_columns(self):
        """
        Return the name of the columns in the dataframe
        """
        return self.df_.names

    @timing
    def is_unique(self, column):
        """
        Check the uniqueness of all values contained in the provided column_name
        """
        pass

    @timing
    def delete_columns(self, columns):
        """
        Delete the specified columns
        Columns is a list of column names
        """
        columns_not_drop = [c for c in self.df_.names if c not in columns]
        self.df_ = self.df_[:, columns_not_drop]
        
        return self.df_

    @timing
    def rename_columns(self, columns):
        """
        Rename the provided columns using the provided names
        Columns is a dictionary: {"column_name": "new_name"}
        """
        for c, n in columns.items():
            self.df_[n] = self.df_[:, c]
            self.df_ = self.df_[:, [x for x in self.df_.names if x != c]]
        return self.df_

    @timing
    def merge_columns(self, columns, separator, name):
        """
        Create a new column with the provided name combining the two provided columns using the provided separator
        Columns is a list of two column names; separator and name are strings
        """
        pass

    @timing
    def fill_nan(self, value, columns=None, func=False):
        """
        Fill nan values in the dataframe with the provided value
        :param value value to use for replacing null values
        :columns columns to fill, if empty all the dataframe is filled
        """
        
        
        if func:
            value = eval(value)

        if columns is None:
            columns = []

        if type(columns) == str:
            columns = [columns]

        for c in columns:
            self.df_[c] = dt.ifelse(dt.isna(dt.f[c]), value, dt.f[c])
        return self.df_


    @timing
    def one_hot_encoding(self, columns):
        """
        Performs one-hot encoding of the provided columns
        Columns is a list of column names
        """
        for column in columns:
            unique_values = dt.unique(self.df_[column]).to_list()[0]
            for value in unique_values:
                self.df_[f"{column}_{value}"] = dt.ifelse(dt.f[column] == value, 1, 0)
        return self.df_

    @timing
    def locate_null_values(self, column):
        """
        Returns the rows of the dataframe which contains
        null value in the provided column.
        """
        if column == "all":
            column = self.get_columns()

        return self.df_[:, dt.isna(dt.f[column])]
    @timing
    def search_by_pattern(self, column, pattern):
        """
        Returns the rows of the datatable which
        match with the provided pattern
        on the provided column.
        Pattern could be a regular expression.
        """
        search = self.df_.copy()
        try:
            filtered = list(filter(lambda x: re.match(pattern, x), search[column].to_list()[0]))
        except Exception:
            print("not regex")
            filtered = list(filter(lambda x: pattern in x if x else "", search[column].to_list()[0]))
        if filtered:
            return search[:, dt.f[column] == filtered]
    

    @timing
    def locate_outliers(self, column, lower_quantile=0.1, upper_quantile=0.99, **kwargs):
        """
        Returns the rows of the dataframe that have values
        in the provided column lower or higher than the values
        of the lower/upper quantile.
        """       
        if column == "all":
            column = self.df_.names

        cols = [
            c
            for c in column
            if str(self.df_[c].types[0])
            in {"Type.int32", "Type.int64", "Type.float32", "Type.float64"}
        ]
        filtered = self.df_[:, cols]
        lower, upper = np.percentile(np.array(filtered), [(lower_quantile*100), (upper_quantile*100)], axis=0)
        for c in cols:
            filtered = filtered[dt.f[c] <= lower[0], :]
            filtered = filtered[dt.f[c] >= upper[0], :]
        return filtered

    @timing
    def get_columns_types(self):
        """
        Returns a dictionary with column types
        """
        return self.df_.types


    @timing
    def cast_columns_types(self, dtypes):
        """
        Cast the data types of the provided columns
        to the provided new data types.
        dtypes is a dictionary that provide for each
        column to cast the new data type.
        """
        for column, dtype in dtypes.items():
            if (str(dtype) != "str") and (column in self.df_.names):
                if str(dtype) == "Type.time64":                
                    try:
                        self.df_ =  self.df_[:, dt.as_type(F.column, dtype)]
                    except Exception:
                        print("the column is not a time, we try to convert it")
                        arr = [f'1970-01-01 {i}' for i in self.df_[column].to_list()[0]]
                        arr = [datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S') for i in arr]
                        self.df_[column] = np.array(arr)
                else:
                    if self.df_[column].type != dtype:
                        self.df_ = self.df_[:, dt.as_type(F.column, dtype)]
                
        return self.df_


    @timing
    def get_stats(self):
        """
        Returns dataframe statistics.
        Only for numeric columns.
        Min value, max value, average value, standard deviation, and standard quantiles.
        """
        rows = []
        for c in self.df_.names:
            if str(self.df_[c].types[0]) in {"Type.int32", "Type.int64", "Type.float32", "Type.float64"}:
                rows.append((c, 
                             self.df_[c].min()[0, 0],
                                self.df_[c].max()[0, 0],
                                self.df_[c].mean()[0, 0],
                                self.df_[c].sd()[0, 0],
                                self.df_[:, dt.count(dt.f[c])][0, 0],
                                np.percentile(np.array(self.df_[:, c]), [25], axis=0)[0][0],
                                np.percentile(np.array(self.df_[:, c]), [50], axis=0)[0][0],
                                np.percentile(np.array(self.df_[:, c]), [75], axis=0)[0][0]
                            ))
        stats = dt.Frame(rows, names=["Column", "Min", "Max", "Mean", "Std", "Count", "25%", "50%", "75%"])      
        return stats

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
        """
        pass

    @timing
    def drop_duplicates(self):
        """
        Drop duplicate rows.
        """
        import numpy as np
        arr_dt = np.array(self.df_.to_list()).T
        s = {tuple(i) for i in arr_dt}
        indices = [i for i, value in enumerate(arr_dt) if tuple(value) in s]
        self.df_ = self.df_[indices, :]
        return self.df_

    @timing
    def drop_by_pattern(self, column, pattern):
        """
        Delete the rows where the provided pattern
        occurs in the provided column.
        """
        pass

    @timing
    def change_date_time_format(self, column, format):
        """
        Change the date/time format of the provided column
        according to the provided formatting string.
        column datatype must be datetime
        An example of format is '%m/%d/%Y'
        """
        if self.df_[column].types[0] not in {"Type.date32", "Type.time", "Type.datetime"}:
            from datetime import datetime
            date_time = []
            for date in self.df_[column].to_list()[0]:
                try:
                    date_time.append(datetime.strptime(date, format))
                except Exception:
                    date_time.append('')
            self.df_[column] = np.array(date_time)
        
        date_time = []
        for i in self.df_[column].to_list()[0]:
            if i != '':
                date_time.append(i.strftime(format))
            else:   
                date_time.append('')
        self.df_[column] = np.array(date_time)
        return self.df_
    
    @timing
    def set_header_case(self, case):
        """
        Put dataframe headers in the provided case
        Supported cases: "lower", "upper", "title", "capitalize", "swapcase"
        (see definitions in pandas documentation)
        """
        pass

    @timing
    def set_content_case(self, columns, case):
        """
        Put dataframe content in the provided case
        Supported cases: "lower", "upper", "title", "capitalize", "swapcase"
        (see definitions in pandas documentation)
        Columns is a list of two column names; empty list for the whole dataframe
        """

        if len(columns) == 0:
            columns = self.df_.names
        for column in columns:
            arr = self.df_[:, columns].to_list()[0]
            if case == "lower":
                arr = list(map(str.lower,arr))
            elif case == "upper":
                arr =  list(map(str.upper,arr))
            elif case == "title":
                arr =  list(map(str.title,arr))
            elif case == "capitalize":
                arr =  list(map(str.capitalize,arr))
            elif case == "swapcase":
                 list(map(str.swapcase,arr))
            self.df_[column] = np.array(arr)

        return self.df_

    @timing
    def duplicate_columns(self, columns):
        """
        Duplicate the provided columns (add to the dataframe with "_duplicate" suffix)
        Columns is a list of column names
        """
        pass
    
    @timing
    def pivot(self, index, columns, values, aggfunc, other_df = None):
        """
        Define the lists of columns to be used as index, columns and values respectively,
        and the dictionary to aggregate ("sum", "mean", "count") the values for each column: {"col1": "sum"}
        (see pivot_table in pandas documentation)
        """
        if other_df:
            print("Pivot table is not supported yet, manually pivot the table")
            aggfunc = eval(aggfunc)
            self.dataframes[other_df] = self.dataframes[other_df][:, aggfunc(dt.f[values]), dt.by(index+columns) ]
            return self.dataframes[other_df]
        
        print("Pivot table is not supported yet, manually pivot the table")
        return  self.df_[:, aggfunc(dt.f[values]), dt.by(index+columns) ]
    



    @timing
    def unpivot(self, columns, var_name, val_name, other_df = None):
        """
        Define the list of columns to be used as values for the variable column,
        the name for variable columns and the one for value column_name
        """
        # manually unpivot
        if other_df:
            unpivoted = self.dataframes[other_df][:, columns]
        else:
            unpivoted = self.df_[:, columns]
            
        id_vars = columns
        value_vars = [c for c in unpivoted.names if c not in columns]
        rows = []
        for row in self.df_.to_tuples():
            season = row[0] 
            for i, value_var in enumerate(value_vars):
                value = row[i + 1]
                rows.append((season, value_var, value))
                
        unpivoted = dt.Frame(rows, names=[id_vars[0], var_name, val_name])
        if other_df:
            self.dataframes[other_df] = unpivoted
            return self.dataframes[other_df]

        return unpivoted
        
        

    @timing
    def delete_empty_rows(self, columns):
        """
        Delete the rows with null values for all provided Columns
        Columns is a list of column names
        """
        if columns == "all":
            columns = list(self.df_.names)

        for column in columns:
            self.df_ = self.df_[dt.f[column] != None, :]

        return self.df_

    @timing
    def split(self, column, sep, splits, col_names):
        """
        Split the provided column into splits + 1 columns named after col_names
        using the provided sep string as separator
        Col_names is a list of column names
        """
        pass

    @timing
    def strip(self, columns, chars):
        """
        Remove the characters appearing in chars at the beginning/end of the provided columns
        Columns is a list of column names
        """
        pass
    
    @timing
    def remove_diacritics(self, columns):
        """
        Remove diacritics from the provided columns
        Columns is a list of column names
        """
        pass

    @timing
    def set_index(self, column):
        """
        Set the provided column as index
        """
        pass

    @timing
    def change_num_format(self, formats):
        """
        Round one ore more columns to a variable number of decimal places.
        formats is a dictionary with the column names as key and the number of decimal places as value.
        """
        pass

    @timing
    def calc_column(self, col_name, f, columns=None, apply = True):
        """
        Calculate the new column col_name by applying
        the function f to the whole dataframe
        """
        if not columns:
            columns = self.df_.names
            
        if apply:
            if isinstance(f, str):
                self.df_[:, col_name] = eval(f)
            
        elif isinstance(f, str):
            f = eval(f)
            arr = self.df_[:, columns].to_numpy()
            self.df_[:, col_name] = np.apply_along_axis(f, 1, arr)
        return self.df_


    @timing
    def join(self, other, left_on=None, right_on=None, how="inner", **kwargs):
        """
        Joins current dataframe (left) with a new one (right).
        left_on/right_on are the keys on which perform the equijoin
        how is the type of join
        **kwargs: additional parameters

        The result is stored in the current dataframe.
        """
        if left_on != right_on:
            other[left_on] = other[:, right_on]
            other = other[:, [x for x in other.names if x != right_on]]

        other.key = left_on
        self.df_ = self.df_[:, :, dt.join(other)]
        return self.df_

    @timing
    def groupby(self, columns, f, inplace=False, new_df = None):
        """
        Aggregate the dataframe by the provided columns
        then applies the function f on every group
        """
        try:
            if new_df:
                self.dataframes[new_df] = self.df_[:, eval(f), dt.by(columns)]
                return self.dataframes[new_df]
            
            if inplace:
                self.df_ = self.df_[:, eval(f), dt.by(columns)]
                       
                for c in self.df_.names:
                    if f"{c}.0" in self.df_.names:
                        pivot_values = dt.unique(self.df_[c]).to_list()[0]
                        pivot_values = [x for x in pivot_values if x != '']
                        grouped_cols = [col for col in columns if col != c]
                        # if function is dt.sum(--) i want extract only dt.sum
                        f = f.split('(')[0]
                        grouped_df = self.df_[:, {f"{pv}": eval(f"{f}(dt.ifelse(dt.f[c] == pv, dt.f[f'{c}.0'], 0)) for pv in pivot_values", 0) for pv in pivot_values}, dt.by(grouped_cols)]
                        self.df_ = grouped_df
                        return self.df_
                return self.df_

            return self.df_[:, eval(f), dt.by(columns)]
        except Exception:
            numeric_columns = [x for x in self.df_.names if self.df_[x].types[0] in [dt.Type.int32, dt.Type.int64, dt.Type.float32, dt.Type.float64]]
            filt = self.df_.copy()
            filt = filt[:, numeric_columns]
            
            if new_df:
                self.dataframes[new_df] = filt[:, eval(f), dt.by(columns)]
                return self.dataframes[new_df]

            if inplace:
                self.df_ = filt[:, eval(f), dt.by(columns)]
                return self.df_
            
            return filt[:, eval(f), dt.by(columns)]
    

    @timing
    def categorical_encoding(self, columns):
        """
        Convert the categorical values in these columns into numerical values
        Columns is a list of column names
        """
        import itertools as it
        for c in columns:
            categories = {k: v for v, k in enumerate(set(self.df_[c].to_list()[0]))}
            mixer = it.product([c], categories)
            conditions = [(name, dt.f[name] == value, categories[value])
                        for name, value in mixer]

            for name, cond, value in conditions:
                self.df_[cond, f'{name}_cat'] = value

            self.df_[name] = self.df_[f'{name}_cat']
            self.df_ = self.df_[:, [x for x in self.df_.names if x != f'{name}_cat']]
        return self.df_

    @timing
    def sample_rows(self, frac, num):
        """
        Return a sample of the rows of the dataframe
        Frac is a boolean:
        - if true, num is the percentage of rows to be returned
        - if false, num is the exact number of rows to be returned
        """
        pass

    @timing
    def append(self, other, ignore_index):
        """
        Append the rows of another dataframe (other) at the end of the provided dataframe
        All columns are kept, eventually filled by nan
        Ignore index is a boolean: if true, reset row indices
        """
        pass

    @timing
    def replace(self, columns, to_replace, value, regex):
        """
        Replace all occurrencies of to_replace (numeric, string, regex, list, dict) in the provided columns using the provided value
        Regex is a boolean: if true, to_replace is interpreted as a regex
        Columns is a list of column names
        """
        for c in columns:
            self.df_[c] = dt.ifelse(dt.f[c] == to_replace, value, dt.f[c])
            
        return self.df_
    
    @timing
    def edit(self, columns, func):
        """
        Edit the values of the cells in the provided columns using the provided expression
        Columns is a list of column names
        """
        if isinstance(func, str):
            func = eval(func)
            
        import numpy as np
        for c in columns:
            arr = np.array(self.df_[:, c])
            arr = np.apply_along_axis(func, 1, arr)
            self.df_[:, c] = arr
        
        return self.df_

    @timing
    def set_value(self, index, column, value):
        """
        Set the cell identified by index and column to the provided value
        """
        pass

    @timing
    def min_max_scaling(self, columns, min, max):
        """
        Independently scale the values in each provided column in the range (min, max)
        Columns is a list of column names
        """
        pass

    @timing
    def round(self, columns, n):
        """
        Round the values in columns using n decimal places
        Columns is a list of column names
        """
        import numpy as np
        for c in columns:
            self.df_[c] = np.around(self.df_[:, c].to_numpy(), decimals=n)
        return self.df_

    @timing
    def get_duplicate_columns(self):
        """
        Return a list of duplicate columns, if exists.
        Duplicate columns are those which have same values for each row.
        """
        pass
    
    @timing
    def to_csv(self, path="./pipeline_output/datatable_output.csv", **kwargs):
        """
        Export the dataframe in a csv file.
        """
        import os
        if not os.path.exists("./pipeline_output"):
            os.makedirs("./pipeline_output")

        for c in self.df_.names:
            if str(self.df_[c].types[0]) == "Type.obj64":
                self.df_[c] = np.array(self.df_[c].to_list()[0], dtype=str)

        self.df_.to_csv(path, **kwargs)

    @timing
    def query(self, query, inplace=False):
        """
        Queries the dataframe and returns the corresponding
        result set.
        :param query: a string with the query conditions, e.g. "col1 > 1 & col2 < 10"
        :return: subset of the dataframe that correspond to the selection conditions
        """
        if inplace:
            self.df_ = self.df_[eval(query), :]
            return self.df_
        return self.df_[eval(query), :]
    
    def force_execution(self):
        self.df_.materialize()
    
    @timing
    def done(self):
        pass
        
    def set_construtor_args(self, args):
        pass
