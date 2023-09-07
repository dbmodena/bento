import re
from codecs import ignore_errors
from concurrent.futures import process
from pydoc import describe
from typing import Union

import dask.dataframe as dd
import h5py
import numpy as np
import pandas as pd
# config.set(scheduler='processes', num_workers=24)
from dask import config, optimize
from dask.distributed import Client, LocalCluster
from haversine import haversine
from numpy import dtype
from polars import col
from pyparsing import srange

from src.algorithms.algorithm import AbstractAlgorithm
from src.algorithms.utils import timing
from src.datasets.dataset import Dataset


class DaskBench(AbstractAlgorithm):
    df_ = None
    backup_ = None
    ds_ : Dataset = None
    name = "dask"
    constructor_args = {}
    
    def __init__(self, mem: str = None, cpu: int = None, pipeline: bool = False):
        import warnings

        import dask

        # Ignore all warnings
        warnings.filterwarnings("ignore")

        # Disable all Dask warnings
        dask.config.set({'logging.distributed': 'error'})
        self.mem_ = mem
        self.cpu_ = cpu
        self.pipeline = pipeline
        self.client = Client(LocalCluster(processes=True, memory_limit=None))
        
        
    def backup(self):
        """
        Creates a backup copy of the current dataframe
        """
        self.backup_ = self.df_.copy()  # type: ignore

    def restore(self):
        """
        Replace the internal dataframe with the backup
        """
        self.df_ = self.backup_.copy() # type: ignore

    def load_from_pandas(self, df : pd.DataFrame)-> None :
        """
        Loads data from a pandas dataframe
        """
        self.df_ = dd.from_pandas(df, npartitions=1) 

    def done(self):
        self.df_.compute()
        self.client.close()

    def get_pandas_df(self) -> pd.DataFrame:
        """
        Returns the internal dataframe as a pandas dataframe
        """
        return self.df_.compute()
    
    @timing
    def load_dataset(self, ds: Dataset, conn=None, **kwargs):
        """
        Load the provided dataframe
        """
        self.ds_ = ds
        path = ds.dataset_attribute.path
        format = ds.dataset_attribute.type

        if format == "csv":
            self.read_csv(path, **kwargs)
        elif format == "excel":
            self.read_excel(path, **kwargs)
        elif format == "json":
            self.read_json(path, **kwargs)
        elif format == "parquet":
            self.read_parquet(path, **kwargs)
        elif format == "sql":
            self.read_sql(path, conn, **kwargs)
        elif format == "hdf5":
            self.read_hdf5(path, **kwargs)
        elif format == "xml":
            self.read_xml(path, **kwargs)
        
        self.df_ = self.df_.persist()
        return self.df_

    def read_sql(self, query, conn, **kwargs) -> dd.DataFrame:
        """
        Given a connection and a query
        creates a dataframe from the query output
        """
        self.df_ = dd.read_sql(query, conn)
        return self.df_

    def read_json(self, path, **kwargs) -> dd.DataFrame:
        """
        Read a json file
        """
        self.df_ = dd.read_json(path, **kwargs)
        return self.df_

    def read_csv(self, path, **kwargs) -> dd.DataFrame:
        """
        Read a csv file
        """
        self.df_ = dd.read_csv(path, **kwargs)
        return self.df_
    
    def read_hdf5(self, path, **kwargs) -> dd.DataFrame:
        """
        Read a csv file
        """
        try:
            self.df_ = dd.read_hdf(path, **kwargs)
        except:
            keys = list(h5py.File(path, 'r').keys())
        

    def read_xml(self, path, **kwargs) -> dd.DataFrame:
        """
        Read a xml file
        """
        self.df_ = dd.from_pandas(pd.read_xml(path, **kwargs), npartitions=1)
        return self.df_

    def read_excel(self, path, **kwargs) -> dd.DataFrame:
        """
        Read an excel file
        """
        self.df_ = dd.from_pandas(pd.read_excel(path, **kwargs), npartitions=1)
        return self.df_

    def read_parquet(self, path, **kwargs):
        """
        Read a parquet file
        """
        self.df_ = dd.read_parquet(path, **kwargs)
        return self.df_

    @timing
    def sort(self, columns, ascending=True, cast=None):
        """
        Sort the dataframe by the provided column
        """
        if cast is None:
            cast = {}
        if cast:
            for column, t in cast.items():
                self.df_ = self.cast_columns_types({column: t})
        if len(columns) > 1:
            print("Not implemented yet by dask for multiple columns")

        self.df_ = self.df_.sort_values(by=columns[0], ascending=ascending)
        return self.df_

    @timing
    def get_columns(self):
        """
        Return the name of the columns in the dataframe
        """
        return list(self.df_.columns.values)

    @timing
    def is_unique(self, column):
        """
        Check the uniqueness of all values contained in all columns in the provided object
        """
        return dd.compute(self.df_[column].nunique() == self.df_.shape[0])[0]

    @timing
    def delete_columns(self, columns):
        """
        Delete the specified columns
        Columns is a list of column names
        """
        self.df_ = self.df_.drop(columns=columns, errors="ignore")
        return self.df_

    @timing
    def rename_columns(self, columns):
        """
        Rename the provided columns using the provided names
        Columns is a dictionary: {"column_name": "new_name"}
        """
        self.df_ = self.df_.rename(columns=columns)
        return self.df_

    @timing
    def merge_columns(self, columns, separator, name):
        """
        Create a new column with the provided name combining the two provided columns using the provided separator
        Columns is a list of two column names; separator and name are strings
        """
        self.df_[name] = (
            self.df_[columns[0]].astype(str)
            + separator
            + self.df_[columns[1]].astype(str)
        )
        return self.df_

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
        if len(columns) == 0:
            self.df_ = self.df_.fillna(value)  
        else:
            for c in columns:
                self.df_[c] = self.df_[c].fillna(value)
        
        return self.df_

    @timing
    def one_hot_encoding(self, columns):
        """
        Performs one-hot encoding of the provided columns
        Columns is a list of column names
        """
        for c in columns:
            self.df_[f'{c}_cat'] = self.df_[c].copy()
            self.df_ = self.df_.categorize(columns=[f'{c}_cat'])
            self.df_ = dd.get_dummies(self.df_, prefix=c, columns=[f'{c}_cat'])
        return self.df_

    @timing
    def locate_null_values(self, column):
        """
        Returns the rows of the dataframe which contains
        null value in the provided column.
        """
        if column == "all":
            column = self.get_columns()
        return self.df_[self.df_[column].isna()]

    @timing
    def search_by_pattern(self, column, pattern):
        """
        Returns the rows of the dataframe which
        match with the provided pattern
        on the provided column.
        Pattern could be a regular expression.
        """
        return self.df_[self.df_[column].str.contains(re.compile(pattern))]

    @timing
    def locate_outliers(self, column, lower_quantile=0.1, upper_quantile=0.99):
        """
        Returns the rows of the dataframe that have values
        in the provided column lower or higher than the values
        of the lower/upper quantile.
        """
        if column == "all":
            column = self.get_columns()

        numeric_only = self.df_[column].astype("float64")

        q_low = numeric_only.quantile(lower_quantile)
        q_hi = numeric_only.quantile(upper_quantile)

        return numeric_only[(numeric_only < q_low) | (numeric_only > q_hi)]

    @timing
    def get_columns_types(self):
        """
        Returns a dictionary with column types
        """
        return self.df_.dtypes.apply(lambda x: x.name).to_dict()

    @timing
    def cast_columns_types(self, dtypes: dict):
        """
        Cast the data types of the provided columns
        to the provided new data types.
        dtypes is a dictionary that provide for each
        column to cast the new data type.
        """
        for col, new_type in dtypes.items():
            self.df_[col] = self.df_[col].astype(new_type)
        return self.df_

    @timing
    def get_stats(self):
        """
        Returns dataframe statistics.
        Only for numeric columns.
        Min value, max value, average value, standard deviation, and standard quantiles.
        """
        return self.df_.describe()

    @timing
    def find_mismatched_dtypes(self):
        """
        L'implementazione originaria non si adattava alla nostra libreria.
        Dask interpreta i mismatch nella colonne come object, per cui facendo un rapido controllo
        su questo tipo di dato, otteniamo le colonne che dobbiamo andare a sistemare.
        Il risultato è un Set con due liste che riportano rispettivamente i nomi delle colonne con type=object
        e la seconda lista i rispettivi indici.
        """
        current_dtypes = self.df_.dtypes.apply(lambda x: x.name).to_dict()
        df1 = self.df_.copy()
        for c in df1.columns:
            df1[c] = dd.to_numeric(df1[c], errors="coerce")

        new_dtypes = df1.dtypes.apply(lambda x: x.name).to_dict()
        res = dd.compute({c: df1[c].isna().any() for c in df1.columns})[0]
        return [{"col": c, "current_dtype": current_dtypes[c], "suggested_dtype": new_dtypes[c],} 
                for c in self.df_.columns if (current_dtypes[c] != new_dtypes[c]) and not res[c]]

    @timing
    def check_allowed_char(self, column, pattern):
        """
        Return true if all the values of the provided column
        follow the provided pattern.
        For example, if the pattern [a-z] is provided the string
        'ciao' will return true, the string 'ciao123' will return false.
        """
        return self.df_[column].str.contains(re.compile(pattern)).compute().all()

    @timing
    def drop_duplicates(self):
        """
        Drop duplicate rows.
        """
        self.df_ = self.df_.drop_duplicates()
        return self.df_

    @timing
    def drop_by_pattern(self, column, pattern):
        """
        Delete the rows where the provided pattern
        occurs in the provided column.
        """

        self.df_ = self.df_[~self.df_[column].str.contains(re.compile(pattern))]
        return self.df_

    @timing
    def change_date_time_format(self, column, format):
        """
        Change the date/time format of the provided column
        according to the provided formatting string.
        column datatype must be datetime
        An example of format is '%m/%d/%Y'
        """
        self.df_[column] = dd.to_datetime(self.df_[column], errors='coerce')
        self.df_[column] = self.df_[column].dt.strftime(format)
        return self.df_

    @timing
    def set_header_case(self, case):
        """
        Put dataframe headers in the provided case
        Supported cases: "lower", "upper", "title", "capitalize", "swapcase"
        (see definitions in pandas documentation)
        """
        if case == "lower":
            self.df_.columns = map(str.lower, self.df_.columns)
        elif case == "upper":
            self.df_.columns = map(str.upper, self.df_.columns)
        elif case == "title":
            self.df_.columns = map(str.title, self.df_.columns)
        elif case == "capitalize":
            self.df_.columns = map(str.capitalize, self.df_.columns)
        elif case == "swapcase":
            self.df_.columns = map(str.swapcase, self.df_.columns)
        return self.df_

    @timing
    def set_content_case(self, columns, case):
        """
        Put dataframe content in the provided case
        Supported cases: "lower", "upper", "title", "capitalize", "swapcase"
        (see definitions in pandas documentation)
        Columns is a list of two column names; empty list for the whole dataframe
        """
        if len(columns) == 0:
            columns = list(self.df_.columns.values)
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

    @timing
    def duplicate_columns(self, columns):
        """
        Duplicate the provided columns (add to the dataframe with "_duplicate" suffix)
        Columns is a list of column names
        """
        for column in columns:
            self.df_[column + "_duplicate"] = self.df_[column]
        return self.df_

    @timing
    def pivot(self, index, columns, values, aggfunc):
        """
        Define the lists of columns to be used as index, columns and values respectively,
        and the dictionary to aggregate ("sum", "mean", "count") the values for each column: {"col1": "sum"}
        (see pivot_table in pandas documentation)
        """
        if len(index) > 1:
            index = index[0]
        df = self.df_.categorize(columns=columns)
        return dd.pivot_table(
            df, index=index, values=values, columns=columns, aggfunc=aggfunc
        ).reset_index()

    @timing
    def unpivot(self, columns, var_name, val_name):
        """
        Define the list of columns to be used as values for the variable column,
        the name for variable columns and the one for value column_name
        """
        self.df_ = dd.melt(
            self.df_,
            id_vars=list(set(list(self.df_.columns.values)) - set(columns)),
            value_vars=columns,
            var_name=var_name,
            value_name=val_name,
        )
        return self.df_

    @timing
    def delete_empty_rows(self, columns):
        """
        Delete the rows with null values for all provided Columns
        Columns is a list of column names
        """
        if columns == 'all':
            columns = list(self.df_.columns.values)
        self.df_ = self.df_.dropna(subset=columns)
        return self.df_

    @timing
    def split(self, column, sep, splits, col_names):
        """
        Split the provided column into splits + 1 columns named after col_names
        using the provided sep string as separator
        Col_names is a list of column names
        """
        self.df_[col_names] = self.df_[column].str.split(sep, splits, expand=True)
        return self.df_

    @timing
    def strip(self, columns, chars):
        """
        Remove the characters appearing in chars at the beginning/end of the provided columns
        Columns is a list of column names
        """
        for column in columns:
            self.df_[column] = self.df_[column].str.strip(chars)
        return self.df_

    @timing
    def remove_diacritics(self, columns):
        """
        Remove diacritics from the provided columns
        Columns is a list of column names
        """
        for column in columns:
            self.df_[column] = (
                self.df_[column]
                .str.normalize("NFKD")
                .str.encode("ascii", errors="ignore")
                .str.decode("utf-8")
            )
        return self.df_

    @timing
    def set_index(self, column):
        """
        Set the provided column as index
        """
        self.df_ = self.df_.set_index(column)
        return self.df_

    @timing
    def change_num_format(self, formats):
        """
        Round one ore more columns to a variable number of decimal places.
        formats is a dictionary with the column names as key and the number of decimal places as value.
        """
        self.df_ = self.df_.round(formats)
        return self.df_

    @timing
    def calc_column(self, col_name, f, columns=None):
        """
        Calculate the new column col_name by applying
        the function f
        """
        self.df_[col_name] = self.df_.apply(eval(f), meta=('x', 'f8'), axis=1)
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
        if type(left_on) == str:
            left_on = [left_on]
            right_on = [right_on]   
        for i in range(len(left_on)):
            if self.df_[left_on[i]].dtype != other[right_on[i]].dtype:
                other[right_on[i]] = other[right_on[i]].astype(self.df_[left_on[i]].dtype)
                
        self.df_ = self.df_.merge(
            other, left_on=left_on, right_on=right_on, how=how, **kwargs
        )
        return self.df_

    @timing
    def groupby(self, columns, f, meta=None):
        """
        Aggregate the dataframe by the provided columns
        then applied the function f on every group
        """
        if meta is None:
            return self.df_.groupby(columns).agg(f)
        
        return self.df_.groupby(columns).apply(f, meta=meta)
        
        #.apply(eval(f), meta={'x': 'f8', 'y': 'f8'})

    @timing
    def categorical_encoding(self, columns):
        """
        Convert the categorical values in these columns into numerical values
        Columns is a list of column names
        """
        cat_df = self.df_.categorize(columns=columns)
        for column in columns:
            self.df_[column] = cat_df[column].cat.codes
        return self.df_

    @timing
    def sample_rows(self, frac, num):
        """
        Return a sample of the rows of the dataframe
        Frac is a boolean:
        - if true, num is the percentage of rows to be returned
        - if false, num is the exact number of rows to be returned
        """
        if frac:
            return self.df_.sample(frac=num / 100)
        f = num / len(self.df_)
        return self.df_.sample(frac=f)

    @timing
    def append(self, other, ignore_index):
        """
        Append the rows of another dataframe (other) at the end of the provided dataframe
        All columns are kept, eventually filled by nan
        Ignore index is a boolean: if true, reset row indices
        """
        self.df_ = dd.concat([self.df_, other], axis=0)
        return self.df_

    @timing
    def replace(self, columns, to_replace, value, regex):
        """
        Replace all occurrencies of to_replace (numeric, string, regex, list, dict) in the provided columns using the provided value
        Regex is a boolean: if true, to_replace is interpreted as a regex
        Columns is a list of column names
        """
        self.df_[columns] = self.df_[columns].replace(
            to_replace=to_replace, value=value, regex=regex
        )
        return self.df_

    @timing
    def edit(self, columns, func, meta=''):
        """
        Edit the values of the cells in the provided columns using the provided expression
        Columns is a list of column names
        """
        if type(func) == str:
            func = eval(func)
        for c in columns:
            self.df_[c] = self.df_[c].map(func, meta = eval(meta))
        return self.df_

    @timing
    def set_value(self, index, column, value):
        """
        Set the cell identified by index and column to the provided value
        """

        def set_val(df):
            df.at[index, column] = value

        self.df_.map_partitions(set_val).compute()

        return self.df_

    @timing
    def min_max_scaling(self, columns, min, max):
        """
        Independently scale the values in each provided column in the range (0, 1)
        Columns is a list of column names
        """
        for column in columns:
            self.df_[column] = self.df_[column] - self.df_[column].min()
            self.df_[column] = self.df_[column] / self.df_[column].max()
            self.df_[column] = self.df_[column] * (max - min) + min
        return self.df_

    @timing
    def round(self, columns, n):
        """
        Round the values in columns using n decimal places
        Columns is a list of column names
        """
        self.df_[columns] = self.df_[columns].round(n)
        return self.df_

    @timing
    def get_duplicate_columns(self):
        """
        Return a list of duplicate columns, if exists.
        Duplicate columns are those which have same values for each row.
        """
        import dask.array as da
        
        def equals(a, b):
            return np.array_equal(a, b)
        
        cols = self.df_.columns.values
        return [(cols[i], cols[j]) 
                for i in range(len(cols)) 
                for j in range(i + 1, len(cols)) 
                if da.map_blocks(equals, self.df_[cols[i]].values, self.df_[cols[j]].values).all()]

    @timing
    def to_csv(self, path=f"/pipeline_output/{name}_loan_output_*.csv", **kwargs):
        """
        Export the dataframe in a csv file.
        """
        import os
        if not os.path.exists("./pipeline_output"):
            os.makedirs("./pipeline_output")
        self.df_.to_csv(f"/pipeline_output/{self.name}_output.csv", **kwargs)

    @timing
    def query(self, query, inplace=False):
        """
        Queries the dataframe and returns the corresponding
        result set.
        :param query: a string with the query conditions, e.g. "col1 > 1 & col2 < 10"
        :return: subset of the dataframe that correspond to the selection conditions
        """
        if inplace:
            self.df_ = self.df_.query(query)
            return self.df_
        return self.df_.query(query, meta=self.df_.dtypes)

    def force_execution(self):

        #self.df_ = optimize(self.df_)
        self.df_.compute()
        
    @timing
    def set_construtor_args(self, args):
        pass
