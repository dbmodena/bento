import contextlib
from os import fchdir
import unicodedata
from typing import Union
from haversine import haversine
import pandas as pd
import polars as pl
from src.algorithms.utils import timing
from src.datasets.dataset import Dataset

from src.algorithms.algorithm import AbstractAlgorithm


class PolarsBench(AbstractAlgorithm):
    df_: Union[pl.DataFrame, pl.Series, pl.LazyFrame] = None
    backup_: Union[pl.DataFrame, pl.Series, pl.LazyFrame] = None
    ds_ : Dataset = None
    name = "polars"
    
    def __init__(self, mem: str = None, cpu: int = None, pipeline: bool = False):
        self.mem_ = mem
        self.cpu_ = cpu
        self.pipeline = pipeline

    def backup(self):
        """
        Creates a backup copy of the current dataframepolars.stats.polar_stats(df)stats
        """
        self.backup_ = self.df_.clone()

    def restore(self):
        """
        Replace the internal dataframe with the backup
        """
        self.df_ = self.backup_.clone()

    @timing
    def load_from_pandas(self, df):
        """
        Loads data from a pandas dataframe
        """
        self.df_ = pl.from_pandas(df)
        return self.df_

    @timing
    def get_pandas_df(self):
        """
        Returns the internal dataframe as a pandas dataframe
        """
        return self.df_.to_pandas()

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
        
        self.df_ = self.df_.lazy()
        return self.df_

    def read_json(self, path, **kwargs):
        """
        :param path: path of the file to load
        :param kwargs: extra arguments
        Read a json file
        """
        self.df_ = pl.read_json(path, **kwargs)
        return self.df_

    def read_csv(self, path, **kwargs):
        """
        Read a csv file
        :param path: path of the file to load
        :param kwargs: extra arguments
        """
        try:
            self.df_ = pl.read_csv(path, **kwargs)
        except:
            self.df_ = pl.from_pandas(pd.read_csv(path, **kwargs))
        return self.df_

    def read_xml(self, path, **kwargs):
        """
        Read a xml file
        :param path: path of the file to load
        :param kwargs: extra arguments
        """
        return self.load_from_pandas(pd.read_xml(path, **kwargs)).df_

    def read_hdf5(self, path, **kwargs):
        return self.load_from_pandas(pd.read_hdf(path, **kwargs))

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
        return pl.read_parquet(path, **kwargs)

    def read_sql(self, query, conn, **kwargs):
        """
        Given a connection and a query
        creates a dataframe from the query output
        :param query query to run to get the data
        :param conn connection to a database
        :param kwargs: extra arguments
        """
        conn = (
            kwargs["db_type"]
            + "://"
            + kwargs["user_name"]
            + ":"
            + kwargs["password"]
            + "@"
            + conn
            + "/"
            + kwargs["db_name"]
        )
        # Two parameters partition_on key for patitioning the table and partition_num is the number of partionied or thread
        self.df_ = pl.read_sql(
            query, conn, kwargs["partition_on"], kwargs["partition_num"]
        )
        return self.df_

    @timing
    def sort(self, columns, ascending=True):
        """
        Sort the dataframe by the provided columns
        Columns is a list of column names
        :param columns columns to use for sorting
        :param ascending if sets to False sorts in descending order (default True)
        """
        self.df_ = self.df_.sort(columns, reverse=(not ascending))
        return self.df_

    @timing
    def get_columns(self):
        """
        Return a list containing the names of the columns in the dataframe
        """
        return self.df_.columns

    @timing
    def is_unique(self, column):
        """
        Check the uniqueness of all values contained in the provided column_name
        :param column column to check
        """
        return self.df_[column].is_unique().all()

    @timing
    def delete_columns(self, columns):
        """
        Delete the provided columns
        Columns is a list of column names
        :param columns columns to delete
        """
        
        self.df_ = self.df_.drop(columns)

        return self.df_

    @timing
    def rename_columns(self, columns):
        """
        Rename the provided columns using the provided names
        Columns is a dictionary: {"column_name": "new_name"}
        :param columns a dictionary that contains for each column to rename the new name
        """
        self.df_ = self.df_.rename(columns)
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
        self.df_ = self.df_.with_columns(
            [pl.format("{}" + separator + "{}", columns[0], columns[1]).alias(name)]
        )
        return self.df_

    @timing
    def fill_nan(self, value, columns=None, func=False):
        """
        Fill nan values in the dataframe with the provided value
        :param value value to use for replacing null values
        :columns columns to fill, if empty all the dataframe is filled
        """
        if columns is None:
            columns = self.df_.columns
        if func:
            value=eval(value)

        for c in columns:
            self.df_ = self.df_.with_column(
                pl.when(pl.col(c).is_null()).then(value).otherwise(pl.col(c)).alias(c)
            )

        return self.df_

    @timing
    def one_hot_encoding(self, columns):
        """
        Performs one-hot encoding of the provided columns
        Columns is a list of column names
        :param columns columns to encode
        """
        for c in columns:
            # get unique values
            unique_values = self.df_.select(c).unique().collect()[c]
            for v in unique_values:
                self.df_ = self.df_.with_columns(
                    pl.when(pl.col(c) == v).then(1).otherwise(0).alias(c + "_" + str(v))
                ).lazy()
        print(self.df_.columns)
        
        #self.df_ = pl.get_dummies(df = self.df_.collect(), columns=columns).lazy()
        
        return self.df_

    @timing
    def locate_null_values(self, column):
        """
        Returns the rows of the dataframe which contains
        null value in the provided column.
        :param column column to explore
        """
        return self.df_.filter(pl.col(column).is_null())

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
        return self.df_.filter(pl.col(column).str.contains(pattern))

    @timing
    def locate_outliers(self, column, lower_quantile=0.1, upper_quantile=0.99):
        """
        Returns the rows of the dataframe that have values
        in the provided column lower or higher than the values
        of the lower/upper quantile.
        :param column column to search on
        :param lower_quantile lower quantile (default 0.1)
        :param upper_quantile upper quantile (default 0.99)
        """
        numeric = self.df_.collect()
        if column=='all':
            column = [c for c in numeric.columns if numeric[c].dtype != 'str']
        

        lower_quantileDF = numeric[column].quantile(lower_quantile, "linear")
        upper_quantileDF = numeric[column].quantile(upper_quantile, "linear")
        
        for c in column:
            numeric = numeric.with_columns(pl.col(c) < lower_quantileDF[c])
            numeric = numeric.with_columns(pl.col(c) > upper_quantileDF[c])
        
        return numeric

    @timing
    def get_columns_types(self):
        """
        Returns a dictionary with column types
        """
        return self.df_.schema

    @timing
    def cast_columns_types(self, dtypes):
        """
        Cast the data types of the provided columns
        to the provided new data types.
        dtypes is a dictionary that provides for each
        column to cast the new data type.
        :param dtypes a dictionary that provides for ech column to cast the new datatype
        For example  {'col_name': pl.UInt32}
        """
        for c in dtypes:
            if dtypes[c] == "str":
                self.df_ = self.df_.with_column(pl.col(c).map(str))
            elif dtypes[c] in [pl.Date, pl.Datetime, pl.Time]:
                self.df_ = self.df_.with_column(pl.col(c).str.strptime(dtypes[c], strict=False).keep_name())
            else:
                self.df_ = self.df_.with_column(pl.col(c).cast(dtypes[c], strict=False))

        return self.df_

    @timing
    def get_stats(self):
        """
        Returns dataframe statistics.
        Only for numeric columns.
        Min value, max value, average value, standard deviation, and standard quantiles.
        """
        return self.df_.collect().describe()

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
        df_s = self.df_.schema
        dfn = self.df_.with_columns(pl.all().cast(pl.Float64, strict=False))
        dfn_s = dfn.schema

        return [{"col": c, "current_dtype": df_s[c], "suggested_dtype": dfn_s[c]} 
                for c in df_s 
                if (df_s[c] != dfn_s[c]) and dfn[c].is_not_null().all()]

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
        return len(self.df_.filter(~pl.col(column).str.contains(pattern))) == 0

    @timing
    def drop_duplicates(self):
        """
        Drop duplicate rows.
        """
        self.df_ = self.df_.unique()
        return self.df_

    @timing
    def drop_by_pattern(self, column, pattern):
        """
        Delete the rows where the provided pattern
        occurs in the provided column.
        """
        self.df_ = self.df_.filter(~pl.col(column).str.contains(pattern))

        return self.df_

    @timing
    def change_date_time_format(self, column, format):
        """
        Change the date/time format of the provided column
        according to the provided formatting string.
        column datatype must be datetime
        An example of format is '%m/%d/%Y'
        :param column column to format
        :param format datetime formatting string
        """
       # self.df_ = self.df_.with_column(pl.col(column).cast(pl.Datetime))
        if str(self.df_.select(column).dtypes[0]) in {
            'string',
            'object',
            'str',
            'Utf8',
        }:
            self.df_ = self.df_.with_column(pl.col(column).str.strptime(pl.Date, fmt=format,strict=False).keep_name())
        else:
            self.df_ = self.df_.with_column(pl.col(column).dt.strftime(format).keep_name())
        return self.df_

    @timing
    def set_header_case(self, case):
        """
        Put dataframe headers in the provided case
        Supported cases: "lower", "upper", "title", "capitalize", "swapcase"
        :param case case format (lower, upper, title, capitalize, swapcase)
        """
        if case == "upper":
            self.df_ = self.df_.select(
                [pl.col(x).alias(x.upper()) for x in self.df_.columns]
            )
        if case == "lower":
            self.df_ = self.df_.select(
                [pl.col(x).alias(x.lower()) for x in self.df_.columns]
            )
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
        for column in columns:
            if case == "lower":
                self.df_ = self.df_.with_columns(pl.col(column).str.to_lowercase())
            elif case == "upper":
                self.df_ = self.df_.with_columns(pl.col(column).str.to_uppercase())
        return self.df_

    @timing
    def duplicate_columns(self, columns):
        """
        Duplicate the provided columns (add to the dataframe with "_duplicate" suffix)
        Columns is a list of column names
        :param columns columns to duplicate
        """
        for column in columns:
            self.df_[column + "_duplicate"] = self.df_[column]
        return self.df_

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
        return self.df_.collect().pivot(
            index=index,
            columns=columns,
            values=values,
            aggregate_fn=aggfunc,
            maintain_order=True,
        )

    @timing
    def unpivot(self, columns, var_name, val_name):
        """
        Define the list of columns to be used as values for the variable column,
        the name for variable columns and the one for value column_name
        """
        self.df_ = (
            self.df_.melt(
                id_vars=list(set(self.df_.columns) - set(columns)), value_vars=columns
            )
            .with_column_renamed("variable", var_name)
            .with_column_renamed("value", val_name)
        )
        return self.df_

    @timing
    def delete_empty_rows(self, columns):
        """
        Delete the rows with null values for all provided Columns
        Columns is a list of column names

        :param columns columns to check
        """
        if columns == 'all':
            columns = self.df_.columns
        self.df_ = self.df_.drop_nulls(columns)
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
        self.seriesDF = self.df_[column].str.split(sep, False)
        self.data = {}
        self.index = 0
        for cols in col_names:
            self.data[cols] = [item[self.index] for item in self.seriesDF]
            self.index = self.index + 1
        self.df_ = pl.DataFrame(self.data).lazy()
        return self.df_

    @timing
    def strip(self, columns, chars):
        """
        Remove the characters appearing in chars at the beginning/end of the provided columns
        Columns is a list of column names
        :param columns columns to edit
        :param chars characters to remove
        """

        def func(x):
            x = str(x)
            x = x.strip(chars)
            return x

        for column in columns:
            self.df_ = self.df_.with_column(pl.col(column).apply(func))
            # self.df_ = self.df_.with_columns(pl.col(column).str.strip(chars))
        return self.df_

    @timing
    def remove_diacritics(self, columns):
        """
        Remove diacritics from the provided columns
        Columns is a list of column names
        :param columns columns to edit
        """

        def enc_str(s):
            s = str(s)
            s = unicodedata.normalize("NFKD", s)
            s = s.encode("ascii", errors="ignore").decode("utf-8")
            return s

        for c in columns:
            # for  i in range(0, len(self.df_[c])):
            # self.df_[i,c] = enc_str(self.df_[i,c])
            self.df_ = self.df_.with_column(pl.col(c).apply(enc_str))
        return self.df_

    @timing
    def set_index(self, column):
        """
        Set the provided column as index
        :param column to use as index
        """
        pass

    @timing
    def change_num_format(self, formats):
        """
        Round one ore more columns to a variable number of decimal places.
        formats is a dictionary with the column names as key and the number of decimal places as value.
        :param formats new column(s) format(s).
               E.g. {'col_name' : 2}
        """
        for x in formats:
            self.df_ = self.df_.with_columns(pl.col(x).round(formats[x]))
        return self.df_

    @timing
    def calc_column(self, col_name, columns, f):
        """
        Calculate the new column col_name by applying
        the function f to the whole dataframe.
        :param col_name column on which apply the function
        :param f function to apply. Must be in the polars format:
                 e.g. to sum two columns
                 pl.map(["col1", "col2"], lambda x: x[0] + x[1])
        """
        if type(f) == str:
            f = eval(f)
            
        self.df_ = self.df_.with_column(pl.struct(columns).apply(f).alias(col_name))
        #new_col = selected.apply(f)
        #print(new_col.collect())
        #print(self.df_.with_column(pl.struct(columns).apply(f)).collect())
        #self.df_ = self.df_.with_column(pl.struct(columns).apply(f).alias(col_name))
        return self.df_

    @timing
    def join(self, other, left_on=None, right_on=None, how="inner", **kwargs):
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
        self.df_ = self.df_.join(
            other, left_on=left_on, right_on=right_on, how=how, **kwargs
        )
        return self.df_.collect()

    @timing
    def groupby(self, columns, f):
        """
        Aggregate the dataframe by the provided columns
        then applies the function f on every group

        :param columns columns to use for group by
        :param f aggregation function
        """
        return self.df_.groupby(columns).agg(f)

    @timing
    def categorical_encoding(self, columns):
        """
        See label encoding / ordinal encoding by sklearn
        Convert the categorical values in these columns into numerical values
        Columns is a list of column names

        :param columns columns to encode
        """
        for c in columns:
            self.df_ = self.df_.with_columns([pl.col(c).cast(pl.Categorical).cat.set_ordering("physical")])
    
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
        return self.df_.sample(frac=num / 100) if frac else self.df_.sample(n=num)

    @timing
    def append(self, other, ignore_index):
        """
        Append the rows of another dataframe (other) at the end of the provided dataframe
        All columns are kept, eventually filled by nan
        Ignore index is a boolean: if true, reset row indices

        :param other other dataframe to append
        :param ignore_index if set to True reset row indices
        """
        self.df_ = pl.concat([self.df_, other])
        return self.df_

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
        if not regex:
            mapping = dict(zip(to_replace, value))
            for col in columns:
                self.df_ = self.df_.with_column(
                        pl.col(col).apply(
                            lambda x: mapping[x] if x in mapping else x
                        )
                    )
        else:
            for col in columns:
                self.df_ = self.df_.with_column(pl.col(col).str.replace(to_replace, value))
        #print(self.df_.select(pl.struct(col).apply(lambda x: mapping[x])).collect())
        #self.df_ = self.df_.with_column(pl.struct(columns).apply(lambda x: mapping[x]))
        #self.df_.select(pl.struct(columns).apply(lambda x: mapping[x]))
        return self.df_

    @timing
    def edit(self, columns, func):
        """
        Edit the values of the cells in the provided columns using the provided expression
        Columns is a list of column names

        :param columns columns on which apply this method
        :param func function to apply
        """
        func = eval(func)
        for c in columns:
            self.df_ = self.df_.with_columns([pl.col(c).apply(func, return_dtype=pl.Float64)])
        return self.df_

    @timing
    def set_value(self, index, column, value):
        """
        Set the cell identified by index and column to the provided value
        :param index row indices
        :param column column name
        :param value value to set
        """
        self.df_[index, column] = value
        return self.df_

    @timing
    def min_max_scaling(self, columns, min, max):
        """
        Independently scale the values in each provided column in the range (min, max)
        Columns is a list of column names

        :param columns columns on which apply this method
        :param min min value
        :param max max value
        """
        for column in columns:
            min_col = self.df_.select(pl.col(column).min())[column][0]
            max_col = self.df_.select(pl.col(column).max())[column][0]
            self.df_[column] = self.df_[column] - min_col
            self.df_[column] = self.df_[column] / max_col
            self.df_[column] = self.df_[column] * (max - min) + min
        return self.df_

    @timing
    def round(self, columns, n):
        """
        Round the values in columns using n decimal places
        Columns is a list of column names
        :param columns columns on which apply this method
        :param n decimal places
        """
        for x in columns:
            self.df_ = self.df_.with_columns(pl.col(x).round(n))
        return self.df_

    @timing
    def get_duplicate_columns(self):
        """
        Return a list of duplicate columns, if exists.
        Duplicate columns are those which have same values for each row.
        """
        cols = self.df_.columns

        return [(cols[i], cols[j])
                for i in range(len(cols))
                for j in range(i + 1, len(cols))
                if self.df_[cols[i]] == self.df_[cols[j]]]

    @timing
    def to_csv(self, path=f"./pipeline_output/{name}_output.csv", **kwargs):
        """
        Export the dataframe in a csv file.
        :param path path on which store the csv
        :param kwargs extra parameters
        """
        #self.df_ = self.df_.collect()
        #print(self.df_.collect().head(10))
        try:
            self.df_.collect().write_csv(path, **kwargs)
        except Exception:
            self.df_.collect().to_pandas().to_csv(path, **kwargs)
          
    @timing  
    def to_parquet(self, path="./pipeline_output/polars_output.parquet", **kwargs):
        """
        Export the dataframe in a parquet file.
        :param path path on which store the parquet
        :param kwargs extra parameters
        """
        self.df_.collect().write_parquet(path, **kwargs)

    @timing
    def query(self, query, inplace=False):
        """
        Queries the dataframe and returns the corresponding
        result set.
        :param query: a string with the query conditions, e.g. "col1 > 1 & col2 < 10"
        :return: subset of the dataframe that correspond to the selection conditions
        """
        if inplace:
            self.df_ = self.df_.filter(query)
            return self.df_
        return self.df_.filter(query)
        
    def force_execution(self):
        self.df_.collect()
    
    @timing    
    def done(self):
        self.df_.collect()
        
    def set_construtor_args(self, args):
        pass
