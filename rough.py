from depths.io.delta import create_delta, read_delta
import polars as pl
import asyncio
from uuid import uuid4
import numpy as np
import os
from shutil import rmtree
import time

NUM_ROWS=100
NUM_DIMS=1536

async def main():
    try:
        dummy_df=pl.DataFrame(
            {
                "id": [str(uuid4()) for _ in range(NUM_ROWS)],
                "name": ["alice"]*NUM_ROWS,
                "scores": np.random.randint(0, 10, (NUM_ROWS, NUM_DIMS)),
            }
        )

        os.makedirs("deltalake_test", exist_ok=True)
        test_dir_abs_path=os.path.abspath("deltalake_test")
        start_time=time.time_ns()
        await create_delta(test_dir_abs_path, dummy_df)
        end_time=time.time_ns()
        print(f"Time taken to create delta: {(end_time - start_time)/1e6:.2f} ms")

        start_time=time.time_ns()
        df=await read_delta(test_dir_abs_path)
        end_time=time.time_ns()
        print(f"Time taken to read delta: {(end_time - start_time)/1e6:.2f} ms")
        print(df)
    finally:
        rmtree(test_dir_abs_path)

if __name__ == "__main__":
    asyncio.run(main())

