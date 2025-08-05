# demo_fixed_per_row_arrow_stream.py
import polars as pl
from shutil import rmtree
import os
from uuid import uuid4
import time
import numpy as np

from depths.io.arrow import (
    write_per_row_stream_ipc,
    write_batches_stream_ipc,
    read_row_from_file,
    read_batch_from_file
)

NUM_ROWS=100
BATCH_SIZE=10
NUM_DIMS=1536

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

        start_time = time.time_ns()
        index = write_per_row_stream_ipc(df, "toy/rows.arrow", index_path="toy/rows_index.parquet")
        end_time = time.time_ns()
        print(f"Time taken to write per-row streams & index: {(end_time - start_time)/1e6:.2f} ms")

        row_idx = 1
        start_time = time.time_ns()
        row_df = read_row_from_file("toy/rows.arrow", row_idx, index)
        end_time = time.time_ns()
        print(f"Time taken to read row {row_idx}: {(end_time - start_time)/1e6:.2f} ms")
        print(f"\nRow {row_idx} read back:")
        print(row_df)

        assert row_df.equals(df.slice(row_idx, 1)), "row mismatch!"
        print(f"\nRow {row_idx} matches original: ✅")
        reconstructed = pl.concat([read_row_from_file("toy/rows.arrow", i, index) for i in range(df.height)])
        print("\nReconstructed full DataFrame:")
        print(f"Shape: {reconstructed.shape}")
        print(reconstructed.head(2))
        assert reconstructed.equals(df), "full reconstruction mismatch!"
        print("\nFull reconstruction matches original: ✅")

        batched_data=[df.slice(i,BATCH_SIZE) for i in range(0,df.height,BATCH_SIZE)]
        start_time = time.time_ns()
        index = write_batches_stream_ipc(batched_data, "toy/batches.arrow", index_path="toy/batches_index.parquet")
        end_time = time.time_ns()
        print(f"Time taken to write batches streams & index: {(end_time - start_time)/1e6:.2f} ms")

        batch_idx = 1
        start_time = time.time_ns()
        batch_df = read_batch_from_file("toy/batches.arrow", batch_idx, index)
        end_time = time.time_ns()
        print(f"Time taken to read batch {batch_idx}: {(end_time - start_time)/1e6:.2f} ms")
        print(f"\nBatch {batch_idx} read back:")
        print(batch_df)

        assert batch_df.equals(df.slice(batch_idx*BATCH_SIZE, BATCH_SIZE)), "batch mismatch!"
        print(f"\nBatch {batch_idx} matches original: ✅")
        print("Test passed ✅")

    finally:
        rmtree("toy")

if __name__ == "__main__":
    main()
