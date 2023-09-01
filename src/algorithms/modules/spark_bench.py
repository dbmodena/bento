import os
import re
import pandas
from pydantic import NoneIsAllowedError

import pyspark
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"
os.environ['HADOOP_HOME_WARN_SUPPRESS'] = "1"
os.environ['HADOOP_ROOT_LOGGER'] = "WARN"
import unicodedata
from typing import Union
from haversine import haversine
import pyspark.sql.functions as fn
from pyspark.sql.functions import *
        
from pyspark.conf import SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
from pyspark.sql.types import DoubleType, StringType, DateType
from src.algorithms.utils import timing
from src.datasets.dataset import Dataset
from pyspark.storagelevel import StorageLevel
from src.algorithms.algorithm import AbstractAlgorithm




class SparkBench(AbstractAlgorithm):
    df_: DataFrame = None
    backup_: DataFrame = None
    ds_ : Dataset = None
    name = "spark"

    def backup(self):
        """
        Creates a backup copy of the current dataframe
        """
        self.backup_ = self.df_.alias("df")

    def restore(self):
        """
        Replace the internal dataframe with the backup
        """
        self.df_ = self.backup_.alias("df")

    def __init__(self, conf=None, master="", app_name="", jarPath=None, mem: str = None, cpu: int = None, pipeline: bool = False, **kwargs):
        self.mem_ = mem
        self.cpu_ = cpu 
        self.pipeline = pipeline
        # Initiate the spark Session inside the constructor
        # build a Spark Session,
        # It takes several parameters and return SparkSession create new one or return existing session
        # Use "JarPath" as Dict and provide the bath
        # e.g JarPath="file:////C://Users//Adeel//Downloads//spark-xml_2.11-0.6.0.jar"
        # .config for all file that uses external jars to read +6+. for example Excel, XML, SQL
        
        # self.c = SparkConf()
        # if conf:
        #     if type(conf) is str:
        #         import json
        #         conf = json.loads(conf)
        #     for k in conf:
        #         self.c.set(k, conf[k])
        #     self.c.set("spark.sql.debug.maxToStringFields", 100)
        # if jarPath:
        #     self.c.set("spark.jars", jarPath)
        #     self.c.set("spark.executor.extraClassPath", jarPath)
        #     self.c.set("spark.executor.extraLibrary", jarPath)
        
        # self.sparkSession = None
        # if len(master) > 0:
        #     self.sparkSession = (
        #         SparkSession.builder.master(master)
        #         .appName(app_name)
        #         .config(conf=self.c)
        #         .getOrCreate()
        #     )
        # else:
        #     self.sparkSession = (
        #         SparkSession.builder.appName(app_name).config(conf=self.c).getOrCreate()
        #     )
        self.c = SparkConf()
        self.c = self.c.set("spark.sql.debug.maxToStringFields", 100)
        self.c = self.c.set("spark.driver.memory", "32g")
        self.c = self.c.set("spark.driver.maxResultSize", "-1")
        self.sparkSession = SparkSession.builder.appName(app_name).config(conf=self.c).getOrCreate()
        print(self.sparkSession.sparkContext.getConf().getAll())
        #self.c = self.sparkSession.sparkContext.getConf()
        
    def force_execution(self):
        """
        Forces the execution of lazy methods
        """
        print("Forcing execution")
        self.df_.count()

    @timing
    def load_from_pandas(self, df):
        """
        Loads data from a pandas dataframe
        """
        self.df_ = self.sparkSession.createDataFrame(df)

    @timing
    def get_pandas_df(self):
        """
        Returns the internal dataframe as a pandas dataframe
        """
        return self.df_.toPandas()

    
    @timing
    def load_dataset(self, ds: Dataset, conn=None, **kwargs):
        """
        Load the provided dataframe
        """
        self.ds_ = ds
        path = ds.dataset_attribute.path
        format = ds.dataset_attribute.type
        import os
        os.system("java -XX:+PrintFlagsFinal -version | grep -Ei 'maxheapsize|maxram'")
        os.system("nproc --all")
        os.system("grep MemTotal /proc/meminfo")
        
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
        
        # remove _c0 column
        if "_c0" in self.df_.columns:
            self.df_ = self.df_.drop("_c0")
        #self.df_.persist(StorageLevel.MEMORY_AND_DISK)
        return self.df_

    @timing
    def read_json(self, path, **kwargs):
        """
        :param path: path of the file to load
        :param kwargs: extra arguments
        Read a json file
        """
        # Give the path of file where json file resides
        self.df_ = self.sparkSession.read.json(path)
        return self.df_

    def read_csv(self, path, **kwargs):
        # Use Dict for sparkSession and pass it parameters sparkSession=sc.sparkSession
        # two parameter File of CSV path and the Session address
        # Return DataFrame for the CSV File
        self.df_ = self.sparkSession.read.csv(path, header=True, inferSchema=True)  # Initiate the dataFrame
        return self.df_

    def read_xml(self, path, **kwargs):
        """
        Read a xml file
        :param path: path of the file to load
        :param kwargs: extra arguments, root tag and row tag
        """
        # XML file need rootTag and RowTag of .Xml file
        # XML executor Jar is needed: spark-xml_2.11-0.6.0.jar
        self.df_ = (
            self.sparkSession.read.format("xml")
            .options(rootTag=kwargs["rootTag"])
            .options(rowTag=kwargs["rowTag"])
            .load(path)
        )
        return self.df_

    def read_excel(self, path, **kwargs):
        """
        Read an excel file
        :param path: path of the file to load
        :param kwargs: extra arguments such as sheet name as "Sheet","sheet1"
        """
        # Executor Jar is needed in Spark Session: mysql-connector-java:5.1.44
        # .config("spark.jars.packages","com.crealytics:spark-excel_2.11:0.12.2")
        # sheet should be provided as for DICT For example Sheet="'Sheet1'!A1"
        self.df_ = (
            self.sparkSession.read.format("com.crealytics.spark.excel")
            .option("useHeader", "true")
            .option("inferSchema", "true")
            .option("dataAddress", kwargs["Sheet"])
            .load(path)
        )
        return self.df_

    def read_parquet(self, path, **kwargs):
        """
        Read a parquet file
        :param path: path of the file to load
        :param kwargs: extra arguments
        """
        # An error might come unsupported 56: It is a Compatiablity issue
        # Use JAVA_HOME to Java 1.8 for such spark operation
        self.df_ = self.sparkSession.read.parquet(path)
        return self.df_

    def read_sql(self, query, conn, **kwargs):
        """
        Given a connection and a query
        creates a dataframe from the query output
        :param query query to run to get the data
        :param conn connection to a database= 127.0.0.1:3306
        :param kwargs: extra arguments such as User, Password
        """
        # Executor Jar is needed: mysql-connector-java:5.1.44
        # .config('spark.jars','mysql:mysql-connector-java:5.1.44')\
        # it uses JDBC drivers;
        # user is the DB admin User name and password as Dict: User='UseName', Password= password
        # Query is SQL query e.g select * from db_name.tablename
        self.df_ = self.sparkSession.read.format("jdbc")\
                    .option("url", f"jdbc:mysql://{conn}")\
                    .option("driver", "com.mysql.jdbc.Driver")\
                    .option("user", kwargs["User"])\
                    .option("password", kwargs["Password"])\
                    .option("query", query).load()

        return self.df_
    
    def read_hdf5(self, path, **kwargs):
        return self.sparkSession.read.format("hdf5").load(path)

    @timing
    def sort(self, columns, ascending=True):
        """
        Sort the dataframe by the provided columns
        Columns is a list of column names
        :param columns columns to use for sorting
        :param ascending if sets to False sorts in descending order (@timing
    default True)
        """
        self.df_ = self.df_.sort(columns, ascending=ascending)
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
        num_unique = self.df_.select(column).distinct().count()
        num = self.df_.select(column).count()

        return num_unique == num

    @timing
    def delete_columns(self, columns):
        """
        Delete the provided columns
        Columns is a list of column names
        :param columns columns to delete
        """
        for c in columns:
            self.df_ = self.df_.drop(c)
        return self.df_

    @timing
    def rename_columns(self, columns):
        """
        Rename the provided columns using the provided names
        Columns is a dictionary: {"column_name": "new_name"}
        :param columns a dictionary that contains for each column to rename the new name
        """
        columnsToRename = []
        renameWith = []
        for key in columns:
            columnsToRename.append(key)
            renameWith.append(columns[key])
        for c, n in zip(columnsToRename, renameWith):
            self.df_ = self.df_.withColumnRenamed(c, n)
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
        self.df_ = self.df_.withColumn(
            name, sf.concat(sf.col(columns[0]), sf.lit(separator), sf.col(columns[1]))
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
            self.df_ = self.df_.na.fill(value)
        else:
            self.df_ = self.df_.na.fill(value, subset=columns)
        return self.df_

    @timing
    def one_hot_encoding(self, columns):
        """
        Performs one-hot encoding of the provided columns
        Columns is a list of column names
        :param columns columns to encode
        """
        from pyspark.ml.functions import vector_to_array
        collected_df = self.df_
        for c in columns:
            stringIndexer = StringIndexer(inputCol=c, outputCol=f"{c}_index")
            model = stringIndexer.fit(collected_df)
            indexed = model.transform(collected_df)
            encoder = OneHotEncoder(
                 inputCols=[f"{c}_index"], outputCols=[f"{c}_onehot"], dropLast=False
            )
            self.df_ = encoder.fit(indexed).transform(indexed)
            self.df_ = self.df_.select('*', vector_to_array(fn.col(f"{c}_onehot")).alias(f"{c}_col_onehot"))
            categories = len(self.df_.first()[f"{c}_col_onehot"])
            cols_expanded = [(fn.col(f"{c}_col_onehot")[i]).alias(f"{c}_{model.labels[i]}") for i in range(categories)]
            self.df_ = self.df_.select('*', *cols_expanded)
            self.df_ = self.delete_columns([f"{c}_index", f"{c}_onehot", f"{c}_col_onehot"])

            # for v in unique_values:
            #     print(v)
            #     self.df_ = self.df_.withColumn(f"{c}_{v}", fn.when(fn.col(c) == v, 1).otherwise(0))
            
            # # print (f"{c}_{value}")
            
        return self.df_

    @timing
    def locate_null_values(self, column):
        """
        Returns the rows of the dataframe which contains
        null value in the provided column.
        :param column column to explore
        """
        if column == "all":
            column = self.get_columns()
            
        # Construct the filter condition dynamically using a for loop
        filter_cond = fn.col(column[0]).isNull()
        for i in range(1, len(column)):
            filter_cond = filter_cond | fn.col(column[i]).isNull()

        return self.df_.filter(filter_cond)

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
        return self.df_.filter(fn.col(column).rlike(pattern))

    @timing
    def locate_outliers(self, column, lower_quantile=0.1, upper_quantile=0.99):
        """
        Returns the rows of the dataframe that have values
        in the provided column lower or higher than the values
        of the lower/upper quantile.
        :param column column to search on
        :param lower_quantile lower quantile (@timing
    default 0.1)
        :param upper_quantile upper quantile (@timing
    default 0.99)
        """

        if column == "all":
            column = [col for col in self.get_columns() if self.df_.select(col).dtypes[0][1] != "string"]

        for c in column:
            q_low = self.df_.approxQuantile(c, [lower_quantile], 0.01)
            q_hi = self.df_.approxQuantile(c, [upper_quantile], 0.01)
            filtered = self.df_.filter((fn.col(c) < q_low[0]) | (fn.col(c) > q_hi[0]))
        
        return filtered
    @timing
    def get_columns_types(self):
        """
        Returns a dictionary with column types
        """
        return {x[0]: x[1] for x in self.df_.dtypes}

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
        for c in dtypes:
            #t = eval(dtypes[c])
            self.df_ = self.df_.withColumn(c, self.df_[c].cast(dtypes[c]))
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
        Returns, if exists, a list of columns with mismatched data types.
        For example, a column with string dtypes that contains only integer values.
        For every columns the list contain an object with three keys:
         - Col: name of the column
         - current_dtype: current data type
         - suggested_dtype: suggested data type
        """
        cols = self.df_.columns

        current_dtypes = self.df_.dtypes

        ndf = self.df_.withColumn(f"{cols[0]}_cast", fn.col(cols[0]).cast("double"))
        for i in range(1, len(cols)):
            ndf = ndf.withColumn(f"{cols[i]}_cast", fn.col(cols[i]).cast("double"))

        ndftypes = {r[0]: r[1] for r in ndf.dtypes}
        return [{"col": c, "current_dtype": ndftypes[c], "suggested_dtype": ndftypes[f"{c}_cast"]} 
                for c in cols 
                if ndftypes[c] != ndftypes[f"{c}_cast"] and ndf.where(fn.col(f"{c}_cast").isNull()).count() == 0]

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
        return (
            self.df_.select(
                fn.when(fn.col(column).rlike(pattern), True)
                .otherwise(False)
                .alias("res")
            )
            .where("res is False")
            .count()
            == 0
        )

    @timing
    def drop_duplicates(self):
        """
        Drop duplicate rows.
        """
        return self.df_.groupBy(self.get_columns()).agg(*[fn.first(c).alias(c) for c in self.df_.columns])

    @timing
    def drop_by_pattern(self, column, pattern):
        """
        Delete the rows where the provided pattern
        occurs in the provided column.
        """
        self.df_ = self.df_.filter(fn.col(column).rlike(pattern) == False)
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
        # use to_timestamp to convert the string column to a timestamp column
        timestamp = fn.to_timestamp(self.df_[column], format=format)
        # use date_format to format the timestamp column as a string
        formatted = fn.date_format(timestamp, format)
        # replace the original column with the formatted column
        self.df_ = self.df_.withColumn(column, formatted).drop(column)
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
                [fn.col(x).alias(x.upper()) for x in self.df_.columns]
            )
        if case == "lower":
            self.df_ = self.df_.select(
                [fn.col(x).alias(x.lower()) for x in self.df_.columns]
            )
        return self.df_

    @timing
    def set_content_case(self, columns, case):
        """
        Put dataframe content in the provided case
        Supported cases: "lower", "upper", "title", "capitalize", "swapcase"
        (see @timing
    definitions in pandas documentation)
        Columns is a list of two column names; empty list for the whole dataframe
        :param columns columns to modify
        :param case case format (lower, upper, initcap)
        """
        if case == "upper":
            for c in columns:
                self.df_ = self.df_.withColumn(c, fn.upper(fn.col(c)))
        elif case == "lower":
            for c in columns:
                self.df_ = self.df_.withColumn(c, fn.lower(fn.col(c)))
        elif case == "initcap":
            for c in columns:
                self.df_ = self.df_.withColumn(c, fn.initcap(fn.col(c)))

        return self.df_

    @timing
    def duplicate_columns(self, columns):
        """
        Duplicate the provided columns (add to the dataframe with "_duplicate" suffix)
        Columns is a list of column names
        :param columns columns to duplicate
        """
        for cols in columns:
            self.df_ = self.df_.withColumn(f"{cols}_duplicate", self.df_[cols])
        return self.df_

    @timing
    def pivot(self, index, columns, values, aggfunc):
        """
        @timing
    define the lists of columns to be used as index, columns and values respectively,
        and the dictionary to aggregate ("sum", "mean", "count") the values for each column: {"col1": "sum"}

        :param index Columns to use to aggregate data
        :param columns columns to pivot.
        :param values  Column(s) to use for populating new frame’s values.
        :param aggfunc dictionary to aggregate ("sum", "mean", "count") the values for each column
               {"col1": "sum"}
        """
        #print(self.df_.groupBy(index).agg({"*": "count"}).show())
        #print(self.df_.groupBy(index).pivot(*columns).agg(aggfunc(*values)).show())
        #columns = [fn.col(c) for c in columns]
        #values = [fn.col(c) for c in values]
        # if type(columns) == list:
        #     print('List is not supported by pyspark, the first values will took')
        #     columns = columns[0]
        
        # return self.df_.groupBy(index).pivot(pivot_col=columns, values=values).sum(*values)
        df_copy = self.df_.select('*')
        agg_dict = {c: aggfunc for c in values}
        for c in values:
            df_copy = df_copy.join(
                df_copy.withColumn('combined', fn.concat(fn.lit(f'{c}_'), fn.col(c)))
                .groupBy(index)
                .pivot("combined")
                .agg(agg_dict),
                on=index,
                how="left",
            )
        return df_copy
                
    

    @timing
    def unpivot(self, columns, var_name, val_name):
        """
        @timing
    define the list of columns to be used as values for the variable column,
        the name for variable columns and the one for value column_name
        """
        op = []
        for c in columns:
            op.extend((f"'{str(c)}'", c))
        q = list(set(list(self.df_.columns)) - set(columns))
        op = ", ".join(op)
        q.append(fn.expr(f"stack({len(columns)}, {op}) as ({var_name}, {val_name})"))
        self.df_ = self.df_.select(list(q))
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
        self.df_ = self.df_.na.drop(subset=columns)

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
        i = 0
        for cols in col_names:
            self.df_ = self.df_.withColumn(
                cols, fn.split(fn.col(column), sep).getItem(i)
            )
            i = i + 1
        return self.df_

    @timing
    def strip(self, columns, chars):
        """
        Remove the characters appearing in chars at the beginning/end of the provided columns
        Columns is a list of column names
        :param columns columns to edit
        :param chars characters to remove
        """
        def strip_str(x, c):
                return str(x).strip(c)

        mystrip = fn.udf(strip_str, StringType())

        for c in columns:
            self.df_ = self.df_.withColumn(c, mystrip(fn.col(c), fn.lit(chars)))
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

        enc_udf = fn.udf(lambda x: enc_str(x), StringType())

        for c in columns:
            self.df_ = self.df_.withColumn(c, enc_udf(fn.col(c)))

        return self.df_

    @timing
    def set_index(self, column):
        """
        Set the provided column as index
        :param column to use as index
        """
        print("This method cannot be implemented in spark")

    @timing
    def change_num_format(self, formats):
        """
        Round one ore more columns to a variable number of decimal places.
        formats is a dictionary with the column names as key and the number of decimal places as value.
        :param formats new column(s) format(s).
               E.g. {'col_name' : 2}
        """
        for column in formats:
            self.df_ = self.df_.withColumn(
                column, fn.round(self.df_[column], formats[column])
            )
        return self.df_

    @timing
    def calc_column(self, col_name, f, columns=None):
        """
        Calculate the new column col_name by applying
        the function f to the whole dataframe.
        :param col_name column on which apply the function
        :param f function to apply, must be an expression, eg. A+1
        """
        # print schema
        udf_lambda = fn.udf(eval(f), StringType())
        new_col_expr = udf_lambda(*[fn.col(col_name) for col_name in columns])
        self.df_ = self.df_.withColumn(col_name, new_col_expr)
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
        other.cache()

        for col_name in list(set(left_on).intersection(set(right_on))):
            new_col_name = f'{col_name}_1'
            other = other.withColumnRenamed(col_name, new_col_name)
            right_on[right_on.index(col_name)] = new_col_name    
    
        join_cond = [self.df_[left_on] == other[right_on] for left_on, right_on in zip(left_on, right_on)]
        self.df_ = self.df_.join(other, join_cond, how).drop(*right_on)
        
        other.unpersist()
    
        return self.df_

    @timing
    def groupby(self, columns, f, cast=None):
        """
        Aggregate the dataframe by the provided columns
        then applies the function f on every group

        :param columns columns to use for group by
        :param f aggregation function
        """
        if cast is None:
            cast = {}
        if cast:
            for col, t in cast.items():
                self.df_ = self.df_.withColumn(col, fn.col(col).cast(t))
        if isinstance(f, str):
            agg_exprs = [getattr(fn, f)(col).alias(f"{col}_{f}") for col in self.df_.columns if col not in columns]
        else:
            agg_exprs = []
            for col, func in f.items():
                func = eval(func)
                if isinstance(func, str):
                    agg_exprs.append(getattr(fn, f)(col).alias(f"{col}_{func}"))
                else:
                    agg_exprs.append(func(col).alias(f"{col}_{func.__name__}"))

        return self.df_.groupBy(*columns).agg(*agg_exprs)


    @timing
    def categorical_encoding(self, columns):
        """
        See label encoding / ordinal encoding by sklearn
        Convert the categorical values in these columns into numerical values
        Columns is a list of column names

        :param columns columns to encode
        """
        for c in columns:
            indexer = StringIndexer(inputCol=c, outputCol=f"{c}_index")
            self.df_ = indexer.setHandleInvalid("keep").fit(self.df_).transform(self.df_)
            self.df_ = self.df_.drop(c) # drop original column
            self.df_ = self.df_.withColumnRenamed(f"{c}_index", c)
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
        if frac:
            return self.df_.sample(fraction=num / 100)
        f = num / self.df_.count()
        print(
            "warning spark do not support exact num sample, only frac is supported"
        )
        return self.df_.sample(fraction=f)

    @timing
    def append(self, other, ignore_index):
        """
        Append the rows of another dataframe (other) at the end of the provided dataframe
        All columns are kept, eventually filled by nan
        Ignore index is a boolean: if true, reset row indices

        :param other other dataframe to append
        :param ignore_index if set to True reset row indices
        """
        self.list1 = self.df_.columns
        self.list2 = other.columns
        for col1 in self.list2:
            if col1 not in self.list1:
                self.df_ = self.df_.withColumn(col1, fn.lit(None))
        for col1 in self.list1:
            if col1 not in self.list2:
                other = other.withColumn(col1, fn.lit(None))

        self.df_ = self.df_.unionByName(other)
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
        
        for column_name in columns:
            self.df_ = self.df_.withColumn(column_name, regexp_replace(col(column_name), to_replace, value))
        return self.df_

    @timing
    def edit(self, columns, func, ret_type):
        """
        Edit the values of the cells in the provided columns using the provided expression
        Columns is a list of column names

        :param columns columns on which apply this method
        :param func function to apply
        """
        my_udf = fn.udf(f=eval(func), returnType=ret_type)
        for c in columns:
           self.df_ = self.df_.withColumn(c, my_udf(fn.col(c)))
        return self.df_

    @timing
    def set_value(self, index, column, value):
        """
        Set the cell identified by index and column to the provided value
        :param index row indices
        :param column column name
        :param value value to set
        """
        print("This method cannot be implemented in spark")

    @timing
    def min_max_scaling(self, columns, min, max):
        """
        Independently scale the values in each provided column in the range (min, max)
        Columns is a list of column names

        :param columns columns on which apply this method
        :param min min value
        :param max max value
        """
        def rescale(value, old_min, old_max, new_min, new_max):
                value = value - old_min
                value = value / old_max
                value = value * (new_max - new_min) + new_min
                return value

        rescale_udf = fn.udf(rescale, DoubleType())

        for column in columns:
            dataFrame2 = self.df_.agg(fn.min(column).alias(f"min{column}"), fn.max(column).alias(f"max{column}"))

            minimum = dataFrame2.select(fn.collect_list(f"min{column}")).first()[0][0]
            maximum = dataFrame2.select(fn.collect_list(f"max{column}")).first()[0][0]
            self.df_ = self.df_.withColumn(
                column,
                rescale_udf(
                    fn.col(column),
                    fn.lit(minimum),
                    fn.lit(maximum),
                    fn.lit(min),
                    fn.lit(max),
                ),
            )
        return self.df_

    @timing
    def round(self, columns, n):
        """
        Round the values in columns using n decimal places
        Columns is a list of column names
        :param columns columns on which apply this method
        :param n decimal places
        """
        for column in columns:
            self.df_ = self.df_.withColumn(column, fn.round(self.df_[column], n))
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
                if self.df_.withColumn("ggg", fn.col(cols[i]) == fn.col(cols[j])).where("ggg == False").count() == 0]

    @timing
    def to_csv(self, path="pipeline_output/spark_output.csv", **kwargs):
        """
        Export the dataframe in a csv file.
        :param path path on which store the csv
        :param kwargs extra parameters
        """
        #path = path.replace("NAME", self.name)
        self.df_.coalesce(1).write.mode("overwrite").save(path, format="csv", header=True)

        # except Exception as e:
        #     print(e)
        #     self.df_.toPandas().to_csv(path, **kwargs)

    @timing
    def query(self, query, inplace = False):
        """
        Queries the dataframe and returns the corresponding
        result set
        :param query: a string with the query conditions, e.g. "col1 > 1 & col2 < 10"
        :return: subset of the dataframe that correspond to the selection conditions
        """
        if type(query) == str:
            query = eval(query)
        
        if inplace:
            self.df_ = self.df_.where(query)
            return self.df_
        return self.df_.where(query)
    @timing
    def done(self):
        self.df_.count()
    
    @timing
    def set_construtor_args(self, args):
        pass
    
    @timing
    def to_parquet(self, path="./pipeline_output/spark_output.parquet", **kwargs):
        """
        Export the dataframe in a csv file.
        """
        self.df_.coalesce(1).write.mode("overwrite").save(path, format="parquet", header=True)
