import apache_beam as beam
from dataclasses import dataclass
import pandas as pd
import xarray as xr
import fsspec
from kerchunk.zarr import ZarrToZarr
import s3fs
import boto3
from boto3.s3.transfer import S3Transfer, TransferConfig
import os
import shutil
import tempfile
from typing import Dict
import uuid

from beam_pyspark_runner.pyspark_runner import PySparkRunner
from pangeo_forge_recipes.storage import FSSpecTarget
from pangeo_forge_recipes.patterns import ConcatDim, FilePattern
from pangeo_forge_recipes.transforms import (
    Indexed,
    OpenURLWithFSSpec,
    OpenWithXarray,
    WriteCombinedReferences
)

import logging
logging.getLogger().setLevel(logging.DEBUG)

SHORT_NAME = 'GPM_3IMERGDF.07'
CONCAT_DIMS = ['time']
IDENTICAL_DIMS = ['lat', 'lon']

# 2023/07/3B-DAY.MS.MRG.3IMERG.20230731
dates = [
    d.to_pydatetime().strftime('%Y/%m/3B-DAY.MS.MRG.3IMERG.%Y%m%d')
    for d in pd.date_range('2000-06-01', '2020-06-01', freq='D')
]


def make_filename(time):
    base_url = f's3://gesdisc-cumulus-prod-protected/GPM_L3/{SHORT_NAME}/'
    return f'{base_url}{time}-S000000-E235959.V07B.nc4'


concat_dim = ConcatDim('time', dates, nitems_per_file=1)
pattern = FilePattern(make_filename, concat_dim)


class DropVarCoord(beam.PTransform):
    """Drops non-viz variables & time_bnds."""

    @staticmethod
    def _dropvarcoord(item: Indexed[xr.Dataset]) -> Indexed[xr.Dataset]:
        index, ds = item
        # Removing time_bnds since it doesn't have spatial dims
        ds = ds.drop_vars('time_bnds')  # b/c it points to nv dimension
        ds = ds[['precipitation']]
        return index, ds

    def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
        return pcoll | beam.Map(self._dropvarcoord)


class TransposeCoords(beam.PTransform):
    """Transform to transpose coordinates for pyramids and drop time_bnds variable"""

    @staticmethod
    def _transpose_coords(item: Indexed[xr.Dataset]) -> Indexed[xr.Dataset]:
        index, ds = item
        ds = ds.transpose('time', 'lat', 'lon')
        return index, ds

    def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
        return pcoll | beam.Map(self._transpose_coords)


def print_and_return(x):
    print(x)
    return x

# NOTE: source uses the EMR serverless execution role (veda-data-reader-dev)
source_fsspec_kwargs = {
  'anon': False,
  'client_kwargs': {'region_name': 'us-west-2'},
}

# NOTE: target uses the EMR serverless execution role (veda-data-reader-dev)
target_fsspec_kwargs = {
	"anon": False,
	"client_kwargs": {"region_name": "us-west-2"}
}
fs_target = s3fs.S3FileSystem(**target_fsspec_kwargs)
#fs_target = fsspec.implementations.local.LocalFileSystem()
target_root = FSSpecTarget(fs_target, 's3://veda-pforge-emr-outputs-v4')
#target_root = FSSpecTarget(fs_target, '/home/jovyan/outputs/')


@dataclass
class RechunkPerFile(beam.PTransform):
    target_chunks: Dict

    def write_intermediate_chunked(self, ds: xr.Dataset) -> xr.Dataset:
        # Create a temporary directory for the Zarr store
        temp_dir = tempfile.mkdtemp()
        zarr_store_path = os.path.join(temp_dir, f'{uuid.uuid1()}.zarr')

        try:
            rechunked_ds = ds.chunk(self.target_chunks)

            # Write the dataset to the local Zarr store
            rechunked_ds.to_zarr(zarr_store_path)

            # Upload the entire Zarr store to S3 using Boto3's S3Transfer
            s3_target_bucket = 'veda-pforge-emr-intermediate-store'
            s3_target_path = f'{uuid.uuid1()}.zarr'

            s3_client = boto3.client('s3')
            transfer = S3Transfer(s3_client, config=TransferConfig(use_threads=True))
            for root, _, files in os.walk(zarr_store_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    s3_key = os.path.relpath(file_path, temp_dir)
                    transfer.upload_file(file_path, s3_target_bucket, os.path.join(s3_target_path, s3_key))

            references = ZarrToZarr(zarr_store_path).translate()
            # Modify the Kerchunk reference to point to the S3 location
            s3_store = f's3://{s3_target_bucket}/{s3_target_path}'
            updated_refs = {}
            for k, v in references['refs'].items():
                if k.startswith(zarr_store_path):
                    new_key = k.replace(zarr_store_path, s3_store)
                    updated_refs[new_key] = v
                else:
                    updated_refs[k] = v
            references['refs'] = updated_refs

            # Clean up the temporary directory
            shutil.rmtree(temp_dir)

            return references
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise
        finally:
            # Clean up the temporary directory
            shutil.rmtree(temp_dir)

    def expand(self, pcoll):
        return pcoll | "Rechunk and open with Xarray" >> beam.MapTuple(
            lambda k, v: (
                k, self.write_intermediate_chunked(v)
            )
        )

target_chunks = {'time': 30, 'lon': 36, 'lat': 18}

with beam.Pipeline(runner=PySparkRunner()) as p:
    (p | beam.Create(pattern.items())
	| OpenURLWithFSSpec(open_kwargs=source_fsspec_kwargs)
	| OpenWithXarray(file_type=pattern.file_type)
    | RechunkPerFile(target_chunks=target_chunks)
	| WriteCombinedReferences(
        target_root=target_root,
		store_name="kerchunk-rechunk-npztest.zarr",
		combine_dims=CONCAT_DIMS,
        identical_dims=IDENTICAL_DIMS,
        target_chunks=target_chunks
	))

