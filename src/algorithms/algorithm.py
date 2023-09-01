import psutil
import abc

from src.datasets.dataset import Dataset

class AbstractAlgorithm(abc.ABC):
    name: str
    constructor_args: dict
    ds_ : Dataset
    
    @property
    def df(self):
        """
        Returns the internal dataframe
        """
        return self._df

    @df.setter
    def set_df(self, df):
        """
        Sets the internal dataframe
        """
        self._df = df

    @abc.abstractmethod
    def backup(self):
        """
        Creates a backup copy of the current dataframe
        """
        pass

    @abc.abstractmethod
    def restore(self):
        """
        Replace the internal dataframe with the backup
        """
        pass

    @abc.abstractmethod
    def force_execution(self):
        """
        Forces the execution of lazy methods
        """
        pass

    @abc.abstractmethod
    def load_from_pandas(self, df):
        """
        Loads data from a pandas dataframe
        """
        pass

    @abc.abstractmethod
    def get_pandas_df(self):
        """
        Returns the internal dataframe as a pandas dataframe
        """
        pass

    @abc.abstractmethod
    def done(self):
        """
        Called when the execution of the algorithm is done
        """
        pass

    def get_memory_usage(self):
        """
        Return the current memory usage of this algorithm instance
        (in kilobytes), or None if this information is not available.
        """
        # return in kB for backwards compatibility
        return psutil.Process().memory_info().rss / 1024

    @abc.abstractmethod
    def load_dataset(self, path, format, **kwargs):
        """
        Load the provided dataframe
        :param path: path of the file to load
        :param format: format (json, csv, xml, excel, parquet, sql)
        :param kwargs: extra arguments
        :return:
        """
        pass

    @abc.abstractmethod
    def read_json(self, path, **kwargs):
        """
        :param path: path of the file to load
        :param kwargs: extra arguments
        Read a json file
        """
        pass

    @abc.abstractmethod
    def read_csv(self, path, **kwargs):
        """
        Read a csv file
        :param path: path of the file to load
        :param kwargs: extra arguments
        """
        pass

    @abc.abstractmethod
    def read_xml(self, path, **kwargs):
        """
        Read a xml file
        :param path: path of the file to load
        :param kwargs: extra arguments
        """
        pass

    @abc.abstractmethod
    def read_excel(self, path, **kwargs):
        """
        Read an excel file
        :param path: path of the file to load
        :param kwargs: extra arguments
        """
        pass

    @abc.abstractmethod
    def read_parquet(self, path, **kwargs):
        """
        Read a parquet file
        :param path: path of the file to load
        :param kwargs: extra arguments
        """
        pass

    @abc.abstractmethod
    def read_sql(self, query, conn, **kwargs):
        """
        Given a connection and a query
        creates a dataframe from the query output
        :param query query to run to get the data
        :param conn connection to a database
        :param kwargs: extra arguments
        """

    @abc.abstractmethod
    def sort(self, columns, ascending=True):
        """
        Sort the dataframe by the provided columns
        Columns is a list of column names
        :param columns columns to use for sorting
        :param ascending if sets to False sorts in descending order (default True)
        """
        pass

    @abc.abstractmethod
    def get_columns(self):
        """
        Return a list containing the names of the columns in the dataframe
        """
        pass

    @abc.abstractmethod
    def is_unique(self, column):
        """
        Check the uniqueness of all values contained in the provided column_name
        :param column column to check
        """
        pass

    @abc.abstractmethod
    def delete_columns(self, columns):
        """
        Delete the provided columns
        Columns is a list of column names
        :param columns columns to delete
        """
        pass

    @abc.abstractmethod
    def rename_columns(self, columns):
        """
        Rename the provided columns using the provided names
        Columns is a dictionary: {"column_name": "new_name"}
        :param columns a dictionary that contains for each column to rename the new name
        """
        pass

    @abc.abstractmethod
    def merge_columns(self, columns, separator, name):
        """
        Create a new column with the provided name combining the two provided columns using the provided separator
        Columns is a list of two column names; separator and name are strings
        :param columns columns to merge
        :param separator separator to use
        :param name new column name
        """
        pass

    @abc.abstractmethod
    def fill_nan(self, value, columns=None):
        """
        Fill nan values in the dataframe with the provided value
        :param value value to use for replacing null values
        :columns columns to fill, if empty all the dataframe is filled
        """
        pass

    @abc.abstractmethod
    def one_hot_encoding(self, columns):
        """
        Performs one-hot encoding of the provided columns
        Columns is a list of column names
        :param columns columns to encode
        """
        pass

    @abc.abstractmethod
    def locate_null_values(self, column):
        """
        Returns the rows of the dataframe which contains
        null value in the provided column.
        :param column column to explore
        """
        pass

    @abc.abstractmethod
    def search_by_pattern(self, column, pattern):
        """
        Returns the rows of the dataframe which
        match with the provided pattern
        on the provided column.
        Pattern could be a regular expression.
        :param column column to search on
        :param pattern pattern to search, string or regex
        """
        pass

    @abc.abstractmethod
    def locate_outliers(self, column, lower_quantile=0.1, upper_quantile=0.99):
        """
        Returns the rows of the dataframe that have values
        in the provided column lower or higher than the values
        of the lower/upper quantile.
        :param column column to search on
        :param lower_quantile lower quantile (default 0.1)
        :param upper_quantile upper quantile (default 0.99)
        """
        pass

    @abc.abstractmethod
    def get_columns_types(self):
        """
        Returns a dictionary with column types
        """
        pass

    @abc.abstractmethod
    def cast_columns_types(self, dtypes):
        """
        Cast the data types of the provided columns
        to the provided new data types.
        dtypes is a dictionary that provides for each
        column to cast the new data type.
        :param dtypes a dictionary that provides for ech column to cast the new datatype
        For example  {'col_name': 'int8'}
        """
        pass

    @abc.abstractmethod
    def get_stats(self):
        """
        Returns dataframe statistics.
        Only for numeric columns.
        Min value, max value, average value, standard deviation, and standard quantiles.
        """
        pass

    @abc.abstractmethod
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

    @abc.abstractmethod
    def check_allowed_char(self, column, pattern):
        """
        Return true if all the values of the provided column
        follow the provided pattern.
        For example, if the pattern [a-z] is provided the string
        'ciao' will return true, the string 'ciao123' will return false.
        :param column column to check
        :param pattern pattern to use
        """
        pass

    @abc.abstractmethod
    def drop_duplicates(self):
        """
        Drop duplicate rows.
        """
        pass

    @abc.abstractmethod
    def drop_by_pattern(self, column, pattern):
        """
        Delete the rows where the provided pattern
        occurs in the provided column.
        """
        pass

    @abc.abstractmethod
    def change_date_time_format(self, column, str_date_time_format):
        """
        Change the date/time format of the provided column
        according to the provided formatting string.
        column datatype must be datetime
        An example of str_date_time_format is '%m/%d/%Y'
        :param column column to format
        :param str_date_time_format datetime formatting string
        """
        pass

    @abc.abstractmethod
    def set_header_case(self, case):
        """
        Put dataframe headers in the provided case
        Supported cases: "lower", "upper", "title", "capitalize", "swapcase"
        :param case case format (lower, upper, title, capitalize, swapcase)
        """
        pass

    @abc.abstractmethod
    def set_content_case(self, columns, case):
        """
        Put dataframe content in the provided case
        Supported cases: "lower", "upper", "title", "capitalize", "swapcase"
        (see definitions in pandas documentation)
        Columns is a list of two column names; empty list for the whole dataframe
        :param columns columns to modify
        :param case case format (lower, upper, title, capitalize, swapcase)
        """
        pass

    @abc.abstractmethod
    def duplicate_columns(self, columns):
        """
        Duplicate the provided columns (add to the dataframe with "_duplicate" suffix)
        Columns is a list of column names
        :param columns columns to duplicate
        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def unpivot(self, columns, var_name, val_name):
        """
        Define the list of columns to be used as values for the variable column,
        the name for variable columns and the one for value column_name
        """
        pass

    @abc.abstractmethod
    def delete_empty_rows(self, columns):
        """
        Delete the rows with null values for all provided Columns
        Columns is a list of column names

        :param columns columns to check
        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def strip(self, columns, chars):
        """
        Remove the characters appearing in chars at the beginning/end of the provided columns
        Columns is a list of column names
        :param columns columns to edit
        :param chars characters to remove
        """
        pass

    @abc.abstractmethod
    def remove_diacritics(self, columns):
        """
        Remove diacritics from the provided columns
        Columns is a list of column names
        :param columns columns to edit
        """
        pass

    @abc.abstractmethod
    def set_index(self, column):
        """
        Set the provided column as index
        :param column to use as index
        """
        pass

    @abc.abstractmethod
    def change_num_format(self, formats):
        """
        Round one ore more columns to a variable number of decimal places.
        formats is a dictionary with the column names as key and the number of decimal places as value.
        :param formats new column(s) format(s).
               E.g. {'col_name' : 2}
        """
        pass

    @abc.abstractmethod
    def calc_column(self, col_name, f):
        """
        Calculate the new column col_name by applying
        the function f to the whole dataframe.
        :param col_name column on which apply the function
        :param f function to apply
        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def groupby(self, columns, f):
        """
        Aggregate the dataframe by the provided columns
        then applies the function f on every group

        :param columns columns to use for group by
        :param f aggregation function
        """
        pass

    @abc.abstractmethod
    def categorical_encoding(self, columns):
        """
        See label encoding / ordinal encoding by sklearn
        Convert the categorical values in these columns into numerical values
        Columns is a list of column names

        :param columns columns to encode
        """
        pass

    @abc.abstractmethod
    def sample_rows(self, frac, num):
        """
        Return a sample of the rows of the dataframe
        Frac is a boolean:
        - if true, num is the percentage of rows to be returned
        - if false, num is the exact number of rows to be returned

        :param frac percentage or exact number of samples to take
        :param num if set to True uses frac as a percentage, otherwise frac is used as a number
        """
        pass

    @abc.abstractmethod
    def append(self, other, ignore_index):
        """
        Append the rows of another dataframe (other) at the end of the provided dataframe
        All columns are kept, eventually filled by nan
        Ignore index is a boolean: if true, reset row indices

        :param other other dataframe to append
        :param ignore_index if set to True reset row indices
        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    def edit(self, columns, func):
        """
        Edit the values of the cells in the provided columns using the provided expression
        Columns is a list of column names

        :param columns columns on which apply this method
        :param func function to apply
        """
        pass

    @abc.abstractmethod
    def set_value(self, index, column, value):
        """
        Set the cell identified by index and column to the provided value
        :param index row indices
        :param column column name
        :param value value to set
        """
        pass

    @abc.abstractmethod
    def min_max_scaling(self, columns, min, max):
        """
        Independently scale the values in each provided column in the range (min, max)
        Columns is a list of column names

        :param columns columns on which apply this method
        :param min min value
        :param max max value
        """
        pass

    @abc.abstractmethod
    def round(self, columns, n):
        """
        Round the values in columns using n decimal places
        Columns is a list of column names
        :param columns columns on which apply this method
        :param n decimal places
        """
        pass

    @abc.abstractmethod
    def get_duplicate_columns(self):
        """
        Return a list of duplicate columns, if exists.
        Duplicate columns are those which have same values for each row.
        """
        pass

    @abc.abstractmethod
    def to_csv(self, path, **kwargs):
        """
        Export the dataframe in a csv file.
        :param path path on which store the csv
        :param kwargs extra parameters
        """
        pass

    @abc.abstractmethod
    def query(self, query):
        """
        Queries the dataframe and returns the corresponding
        result set.
        :param query: a string with the query conditions, e.g. "col1 > 1 & col2 < 10"
        :return: subset of the dataframe that correspond to the selection conditions
        """
        pass
    
    @abc.abstractmethod
    def set_construtor_args(self, **kwargs):
        """
        Set the constructor arguments for the algorithm.
        :param kwargs: constructor arguments
        """
        pass

    class Config:
        arbitrary_types_allowed = True