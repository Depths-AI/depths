import polars as pl
from deltalake import DeltaTable
from deltalake.exceptions import DeltaError, TableNotFoundError
import asyncio
from typing import Optional, List, Dict, Any, Tuple

NUM_RETRIES = 3
NO_HISTORY = {
    "delta.logRetentionDuration": "interval 0 days",
    "delta.deletedFileRetentionDuration": "interval 0 days",
}

async def create_delta(
    table_path: str,
    data: pl.DataFrame,
    mode: str = "ignore",
    num_retries: int = NUM_RETRIES,
    storage_options: Optional[Dict[str, str]] = None,
    partition_by: Optional[List[str]] = None,
    partition_filters: Optional[List[Tuple[str, str, Any]]] = None,
    delta_write_options: Optional[Dict[str, Any]] = None,
):
    """Creates a Delta table with the given data.

    Supports both local file paths (absolute and relative paths) and S3 paths (e.g., "s3://bucket/path/to/table").
    Retries the operation a specified number of times in case of failure.

    Args:
        table_path: The URI path to the Delta table.
        data: The Polars DataFrame to write to the table.
        mode: The write mode ('error', 'append', 'overwrite', 'ignore').
              Defaults to 'ignore' (if table exists, do nothing).
        num_retries: The number of times to retry the operation upon failure. Defaults to NUM_RETRIES.
        storage_options: A dictionary of options for the storage backend (e.g., S3 credentials).
                         Defaults to None.

    Raises:
        Exception: Re-raises the last exception if all retries fail.
                  Could be DeltaError or other storage-related exceptions.
    """
    write_opts = dict(delta_write_options or {})
    if partition_by:
        write_opts["partition_by"] = partition_by  #
    if partition_filters:
        write_opts["partition_filters"] = partition_filters
    cfg: Dict[str, str] = {**NO_HISTORY, **write_opts.get("configuration", {})}
    write_opts["configuration"] = cfg

    for attempt in range(num_retries):
        try:
            data.write_delta(
                table_path,
                mode=mode,
                storage_options=storage_options,
                delta_write_options=write_opts,
            )
            return
        except Exception as e:
            if attempt == num_retries - 1:
                raise e
            await asyncio.sleep((attempt + 1) * 0.1)


async def read_delta(
    table_path: str,
    storage_options: Optional[Dict[str, str]] = None,
    partitions: Optional[List[Tuple[str, str, Any]]] = None,
    filters: Optional[Any] = None,
    return_lf: Optional[bool] = False,
) -> pl.DataFrame:
    '''
    Read a Delta table into a Polars LazyFrame/DataFrame (can choose).
    
    Supports both local file paths (tested for file:/// prefixed absolute paths and os module defined absolute paths)
    and S3 paths (e.g., "s3://bucket/path/to/table").
    
    Retries with DeltaTable.read() if pl.scan_delta() fails.
    
    Args:
        table_path: The path to the Delta table.
        storage_options: Optional storage options for the Delta table.
        partitions: Optional list of partitions to read.
        filters: Optional filters to apply to the table.
        return_lf: Whether to return a LazyFrame or a DataFrame.
    
    Returns:
        A LazyFrame or DataFrame containing the Delta table data.
    '''
    try:
        pyarrow_opts = None
        if partitions or filters:
            pyarrow_opts = {}
            if partitions:
                pyarrow_opts["partitions"] = partitions
            if filters:
                pyarrow_opts["filter"] = filters

        lf = pl.scan_delta(
            table_path,
            storage_options=storage_options,
            pyarrow_options=pyarrow_opts,
        )
        return lf if return_lf else lf.collect()

    except (TableNotFoundError, DeltaError):
        raise ValueError("Table not found")

    except Exception:
        try:
            dt = DeltaTable(table_path, storage_options=storage_options)
            pa_tbl = dt.to_pyarrow_table(partitions=partitions, filters=filters)
            return pl.from_arrow(pa_tbl)
        except Exception:
            raise ValueError("Failed to read table")