import pyarrow as pa
from pyarrow import ipc
import polars as pl
from typing import List, Dict, Optional, Literal

def write_per_row_stream_ipc(
    data:pl.DataFrame,
    path: str,
    index_column_name: Optional[str]="row",
    index_path: Optional[str] = None,
)-> pl.DataFrame:

    index: List[Dict]=[]
    with pa.OSFile(path,"wb") as sink:
        for i in range(data.height):
            row_df=data.slice(i,1)
            arrow_table=row_df.to_arrow()
            batch=arrow_table.to_batches()[0]

            buf_out=pa.BufferOutputStream()
            write_options=ipc.IpcWriteOptions(compression="zstd")
            with ipc.RecordBatchStreamWriter(buf_out,batch.schema,options=write_options) as stream_writer:
                stream_writer.write_batch(batch)
            buf=buf_out.getvalue()

            start=sink.tell()
            sink.write(buf)
            end=sink.tell()
            length=end-start

            entry={index_column_name:i,"offset":start,"length":length}
            index.append(entry)
    
    if index_path:
        index=pl.DataFrame(index)
        index.write_parquet(index_path)
    return index

def write_batches_stream_ipc(
    batched_data: List[pl.DataFrame],
    path: str,
    index_column_name: Optional[str]="batch",
    index_path: Optional[str] = None,
)-> pl.DataFrame:
    
    index: List[Dict]=[]
    with pa.OSFile(path,"wb") as sink:
        for i in range(len(batched_data)):
            batch_df=batched_data[i]
            arrow_table=batch_df.to_arrow()
            batch=arrow_table.to_batches()[0]

            buf_out=pa.BufferOutputStream()
            write_options=ipc.IpcWriteOptions(compression="zstd")
            with ipc.RecordBatchStreamWriter(buf_out,batch.schema,options=write_options) as stream_writer:
                stream_writer.write_batch(batch)
            buf=buf_out.getvalue()

            start=sink.tell()
            sink.write(buf)
            end=sink.tell()
            length=end-start

            entry={index_column_name:i,"offset":start,"length":length}
            index.append(entry)
    
    if index_path:
        index=pl.DataFrame(index)
        index.write_parquet(index_path)
    return index

def read_row_from_file(
    path:str,
    row_index:int,
    index: pl.DataFrame,
    index_column_name: Optional[str]="row"
)-> pl.DataFrame:
    
    entry=index.row(
        by_predicate=(pl.col(index_column_name)==row_index),
        named=True
    )

    with pa.OSFile(path,"rb") as source:
        source.seek(entry["offset"])
        data=source.read(entry["length"])

    reader=ipc.open_stream(pa.BufferReader(data))
    try:
        batch=next(reader)
    except StopIteration:
        raise RuntimeError(f"Unexpected end of stream for row {row_index}")

    return pl.from_arrow(batch)

def read_batch_from_file(
    path:str,
    batch_index:int,
    index: pl.DataFrame,
    index_column_name: Optional[str]="batch"
)-> pl.DataFrame:
    
    entry=index.row(
        by_predicate=(pl.col(index_column_name)==batch_index),
        named=True
    )

    with pa.OSFile(path,"rb") as source:
        source.seek(entry["offset"])
        data=source.read(entry["length"])

    reader=ipc.open_stream(pa.BufferReader(data))
    try:
        batch=next(reader)
    except StopIteration:
        raise RuntimeError(f"Unexpected end of stream for batch {batch_index}")

    return pl.from_arrow(batch)
