# demo_fixed_per_row_arrow_stream.py
import polars as pl
import pyarrow as pa
from pyarrow import ipc
import zlib
import json
from typing import List, Dict, Optional
from shutil import rmtree
import os
from uuid import uuid4
import time
import numpy as np

NUM_ROWS=1000
NUM_DIMS=1536

def write_per_row_stream_ipc(
    df: pl.DataFrame,
    path: str,
    index_path: Optional[str] = None,
    compute_checksum: bool = True,
) -> List[Dict]:
    index: List[Dict] = []
    with pa.OSFile(path, "wb") as sink:
        for i in range(df.height):
            # one-row slice
            row_df = df.slice(i, 1)
            arrow_table = row_df.to_arrow()  # zero-copy conversion. :contentReference[oaicite:3]{index=3}
            batch = arrow_table.to_batches()[0]  # single RecordBatch

            # serialize as a standalone stream (schema + batch)
            buf_out = pa.BufferOutputStream()
            write_options=ipc.IpcWriteOptions(compression="zstd")
            with ipc.RecordBatchStreamWriter(buf_out, batch.schema, options=write_options) as stream_writer:
                stream_writer.write_batch(batch)  # streaming IPC. :contentReference[oaicite:4]{index=4}
            buf = buf_out.getvalue()
            row_bytes = buf.to_pybytes()

            start = sink.tell()
            sink.write(buf)
            end = sink.tell()
            length = end - start

            entry = {"row": i, "offset": start, "length": length}
            if compute_checksum:
                entry["hash"] = zlib.crc32(row_bytes)
            index.append(entry)

    if index_path:
        with open(index_path, "w") as f:
            json.dump({"version": 1, "rows": index}, f, indent=2)
    return index


def read_row_from_file(
    path: str, index: List[Dict], row_idx: int, verify: bool = True
) -> pl.DataFrame:
    entry = next(e for e in index if e["row"] == row_idx)
    with open(path, "rb") as f:
        f.seek(entry["offset"])
        data = f.read(entry["length"])

    if verify and "hash" in entry:
        if zlib.crc32(data) != entry["hash"]:
            raise ValueError(f"Checksum mismatch for row {row_idx}")

    # open_stream gives a RecordBatchStreamReader over the embedded stream. :contentReference[oaicite:5]{index=5}
    reader = ipc.open_stream(pa.BufferReader(data))
    try:
        batch = next(reader)  # single-row RecordBatch
    except StopIteration:
        raise RuntimeError(f"Unexpected end of stream for row {row_idx}")

    # Convert RecordBatch directly to Polars DataFrame. :contentReference[oaicite:6]{index=6}
    return pl.from_arrow(batch)


def main():
    # toy DataFrame
    try:
        df = pl.DataFrame(
            {
                "id": [str(uuid4()) for _ in range(NUM_ROWS)],
                "name": ["alice"]*NUM_ROWS,
                "scores": np.random.randint(0, 10, (NUM_ROWS, NUM_DIMS)),
            }
        )
        df=df.with_columns(
            pl.col("scores").cast(pl.Array(pl.Int8, NUM_DIMS))
        )
        print("Original DataFrame:")
        print(f"Shape: {df.shape}")
        print(df.head(2))

        os.makedirs("toy", exist_ok=True)

        # write per-row streams & index
        start_time = time.time_ns()
        index = write_per_row_stream_ipc(df, "toy/rows.arrow", index_path="toy/rows.index.json")
        end_time = time.time_ns()
        print(f"Time taken to write per-row streams & index: {(end_time - start_time)/1e6:.2f} ms")

        # read back one row
        row_idx = 1
        start_time = time.time_ns()
        row_df = read_row_from_file("toy/rows.arrow", index, row_idx)
        end_time = time.time_ns()
        print(f"Time taken to read row {row_idx}: {(end_time - start_time)/1e6:.2f} ms")
        print(f"\nRow {row_idx} read back:")
        print(row_df)

        # verify equality
        assert row_df.equals(df.slice(row_idx, 1)), "row mismatch!"
        print(f"\nRow {row_idx} matches original: ✅")

        # reconstruct full frame
        reconstructed = pl.concat([read_row_from_file("toy/rows.arrow", index, i) for i in range(df.height)])
        print("\nReconstructed full DataFrame:")
        print(f"Shape: {reconstructed.shape}")
        print(reconstructed.head(2))
        assert reconstructed.equals(df), "full reconstruction mismatch!"
        print("\nFull reconstruction matches original: ✅")

    finally:
        pass
        #rmtree("toy")

if __name__ == "__main__":
    main()
