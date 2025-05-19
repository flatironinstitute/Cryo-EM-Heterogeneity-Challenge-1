from typing import Optional, Annotated, Literal, Dict
from typing_extensions import Self
import glob
import os
from pathlib import PurePath
from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveFloat,
    FiniteFloat,
    PositiveInt,
    NonNegativeInt,
    FilePath,
    DirectoryPath,
    AfterValidator,
    Field,
    field_validator,
    model_validator,
)
import pandas
import torch
from torch import Tensor


def _is_submission_file(filename: FilePath) -> FilePath:
    assert filename.suffix == ".pt", f"File {filename} is not a .pt file"
    assert "submission" in filename.name, f"File {filename} is not a submission file"
    return filename


def _contains_mrc_files_minus_mask(path_to_volumes: str) -> bool:
    """
    Check if path_to_volumes contains .mrc files.
    """
    vol_paths = glob.glob(os.path.join(path_to_volumes, "*.mrc"))
    vol_paths = [vol_path for vol_path in vol_paths if "mask" not in vol_path]
    assert len(vol_paths) > 0, "No .mrc files found in the directory"
    return path_to_volumes


def _is_mrc_file(filename: FilePath) -> bool:
    """
    Check if the file is a .mrc file.
    """
    assert filename.suffix == ".mrc", f"File {filename} is not a .mrc file"
    return filename


def _is_pt_file(filename: FilePath) -> FilePath:
    """
    Check if the file is a .pt file.
    """
    assert filename.suffix == ".pt", f"File {filename} is not a .pt file"
    return filename


class MapToMapInputConfigDataSubmission(BaseModel, extra="forbid"):
    path_to_submission_file: Annotated[
        FilePath, AfterValidator(_is_submission_file)
    ] = Field(
        description="Path to the submission file in .pt format",
    )

    volume_key: str = Field(
        default="volumes",
        description="Key to access the volume in the submission file",
    )

    metadata_key: str = Field(
        default="populations",
        description="Key to access the metadata in the submission file",
    )

    label_key: str = Field(
        default="id",
        description="Key to access the labels in the submission file",
    )


class MapToMapInputConfigDataGroundTruth(BaseModel, extra="forbid"):
    path_to_volumes: Annotated[FilePath, AfterValidator(_is_pt_file)] = Field(
        description="Path to the submission file in .pt format",
    )

    path_to_metadata: FilePath = Field(
        description="Path to the metadata file in .csv format",
    )


class MapToMapInputConfigDataMask(BaseModel, extra="forbid"):
    apply_mask: bool = Field(
        default=False,
        description="Apply mask to the volumes",
    )
    path_to_mask: Optional[Annotated[FilePath, AfterValidator(_is_mrc_file)]] = Field(
        default=None,
        description="Path to the mask file in .mrc format",
    )


class MapToMapInputConfigData(BaseModel, extra="forbid"):
    box_size: PositiveInt = Field(
        description="Box size of the submitted volumes",
    )
    voxel_size: PositiveFloat = Field(
        description="Voxel size of the submitted volumes in Angstroms",
    )
    submission_params: dict = Field(
        description="Parameters for the submission file",
    )
    ground_truth_params: dict = Field(
        description="Parameters for the ground truth file",
    )
    mask_params: dict = Field(
        description="Parameters for the mask file",
    )

    @field_validator("submission_params")
    @classmethod
    def validate_submission_params(cls, submission_params):
        return dict(MapToMapInputConfigDataSubmission(**submission_params).model_dump())

    @field_validator("ground_truth_params")
    @classmethod
    def validate_ground_truth_params(cls, ground_truth_params):
        return dict(
            MapToMapInputConfigDataGroundTruth(**ground_truth_params).model_dump()
        )

    @field_validator("mask_params")
    @classmethod
    def validate_mask_params(cls, mask_params):
        return dict(MapToMapInputConfigDataMask(**mask_params).model_dump())


class MapToMapInputConfigNormalize(BaseModel, extra="forbid"):
    do: bool = Field(
        default=False,
        description="Whether to normalize the volumes",
    )
    method: Literal["median_zscore"] = Field(
        default="median_zscore",
        description="Method to use for normalization",
    )


class MapToMapInputConfigMetricsGromovWasserstein(BaseModel, extra="forbid"):
    top_k: PositiveInt = Field(
        default=1,
        description="Number of voxels to use (ranked according to highest mass)",
    )
    downsample_box_size: PositiveInt = Field(
        default=32,
        description="Number of pixels to downsample to (in each dimension)",
    )
    exponent: FiniteFloat = Field(
        description="Exponential weighting of GW cost. Base cost is Euclindean distance"
    )
    cost_scale_factor: FiniteFloat = Field(
        description="multiplicative scaling factor for the cost (before exponentiation)"
    )
    elementwise_not_rowwise: bool = Field(
        default=False,
        description="dask parralelization: whether to call dask.compute on each map-to-map computation, or naively loop through each submitted maps row-wise",
    )
    dask: Optional[dict] = Field(
        default=None,
        description="Dask configuration for parallelization",
    )
    solver: Literal["python_ot", "frank_wolfe"] = Field(
        default="python_ot",
        description="Solver to use for the Gromov-Wasserstein distance",
    )
    python_ot_params: Optional[dict] = Field(
        default=None,
        description="Parameters for the python_ot solver",
    )
    frank_wolfe_params: Optional[dict] = Field(
        default=None,
        description="Parameters for the frank_wolfe solver",
    )
    compute_self_metric: Optional[bool] = Field(
        default=True,
        description="Whether to compute the self metric",
    )

    @field_validator("dask")
    @classmethod
    def validate_dask_params(cls, dask_params):
        if dask_params is not None:
            dask_params = dict(
                MapToMapInputConfigDaskParams(**dask_params).model_dump()
            )
        return dask_params

    @field_validator("python_ot_params")
    @classmethod
    def validate_python_ot_params(cls, python_ot_params):
        # TODO: assert that the solver is python_ot
        if python_ot_params is not None:
            python_ot_params = dict(
                MapToMapInputConfigPythonOTParams(**python_ot_params).model_dump()
            )
        return python_ot_params

    @field_validator("frank_wolfe_params")
    @classmethod
    def validate_frank_wolfe_params(cls, frank_wolfe_params):
        # TODO: assert that the solver is frank_wolfe
        if frank_wolfe_params is not None:
            frank_wolfe_params = dict(
                MapToMapInputConfigFrankWolfeParams(**frank_wolfe_params).model_dump()
            )
        return frank_wolfe_params


class MapToMapInputConfigDaskParams(BaseModel, extra="forbid"):
    scheduler: Optional[str] = Field(
        default=None, description="string argument to dask.compute(scheduler=...)"
    )
    local_directory: Optional[DirectoryPath] = Field(
        default="/tmp", description="directory for dask.distributed.Client"
    )
    scheduler_file_directory: Optional[DirectoryPath] = Field(
        default=None,
        description="directory for dask_jobqueue.slurm.SLURMRunner(scheduler_file=dask_jobqueue.slurm.SLURMRunner + scheduler-job_id.json)",
    )
    slurm: bool = Field(
        default=False, description="Whether to use dask_jobqueue.slurm.SLURMRunner"
    )


class MapToMapInputConfigPythonOTParams(BaseModel, extra="forbid"):
    gw_distance_function_key: Literal["gromov_wasserstein2"] = Field(
        default="gromov_wasserstein2", description="Key for the GW distance function"
    )
    tol_abs: float = Field(default=1e-18, description="Absolute tolerance")
    tol_rel: float = Field(default=1e-18, description="Relative tolerance")
    max_iter: int = Field(default=10000, description="Maximum number of iterations")
    verbose: bool = Field(default=False, description="Verbosity flag")
    loss_fun: Literal["square_loss", "kl_loss"] = Field(
        default="square_loss", description="Loss function"
    )


class MapToMapInputConfigFrankWolfeParams(BaseModel, extra="forbid"):
    max_iter: int = Field(default=100, description="Maximum number of iterations")
    gamma_atol: float = Field(default=1e-6, description="Tolerance for convergence")


class MapToMapInputConfigMetricsProcrustesWasserstein(BaseModel, extra="forbid"):
    downsample_box_size: PositiveInt = Field(
        default=32,
        description="Final box size of downsampled volume.",
    )
    top_k: PositiveInt = Field(
        default=1000,
        description="Number of voxels to use (ranked according to highest mass)",
    )
    max_iter: PositiveInt = Field(
        description="Number of iterations, where each iterations updates the correspondence / transport plan and pose)",
    )
    tol: PositiveFloat = Field(
        description="Stopping tolerance (absolute difference of objective between iterations).",
    )
    compute_self_metric: Optional[bool] = Field(
        default=True,
        description="Whether to compute the self metric",
    )


class MapToMapInputConfigMetricsZernike3d(BaseModel, extra="forbid"):
    gpuID: NonNegativeInt = Field(
        default=0,
        description="Identifier of GPU",
    )
    tmpDir: str = Field(
        default="tmp_zernike",
        description="Name of directory to store intermediate files",
    )
    thr: PositiveInt = Field(
        description="Number of threads to use for the computation",
    )
    numProjections: PositiveInt = Field(
        description="Number of projections. Suggested 20-100.",
    )
    compute_self_metric: Optional[bool] = Field(
        default=True,
        description="Whether to compute the self metric",
    )


class MapToMapInputConfigMetricsL2BioemCorr(BaseModel, extra="forbid"):
    chunk_size_submission: PositiveInt = Field(
        default=20,
        description="Batch size of the submission volumes",
    )
    chunk_size_gt: PositiveInt = Field(
        default=100,
        description="Batch size of the ground truth volumes",
    )
    normalize_params: Optional[Dict] = Field(
        default=None,
        description="Parameters for the normalization of the volumes",
    )
    low_memory: Optional[Dict] = Field(
        default=None,
        description="Parameters for the low memory mode",
    )
    compute_self_metric: Optional[bool] = Field(
        default=True,
        description="Whether to compute the self metric",
    )


class MapToMapInputConfigMetricsFscRes(BaseModel, extra="forbid"):
    compute_self_metric: Optional[bool] = Field(
        default=True,
        description="Whether to compute the self metric",
    )


class MapToMapInputConfigSharedParams(BaseModel, extra="forbid"):
    chunk_size_submission: PositiveInt = Field(
        default=20,
        description="Batch size of the submission volumes",
    )
    chunk_size_gt: PositiveInt = Field(
        default=100,
        description="Batch size of the ground truth volumes",
    )
    normalize_params: Optional[Dict] = Field(
        default=None,
        description="Parameters for the normalization of the volumes",
    )
    low_memory: Optional[Dict] = Field(
        default=None,
        description="Parameters for the low memory mode",
    )

    @field_validator("low_memory")
    @classmethod
    def validate_low_memory_params(cls, low_memory_params):
        if low_memory_params is not None:
            low_memory_params = dict(
                MapToMapInputConfigLowMemory(**low_memory_params).model_dump()
            )
        return low_memory_params

    @field_validator("normalize_params")
    @classmethod
    def validate_normalize_params(cls, normalize_params):
        if normalize_params is not None:
            normalize_params = dict(
                MapToMapInputConfigNormalize(**normalize_params).model_dump()
            )
        return normalize_params


class MapToMapInputConfigLowMemory(BaseModel, extra="forbid"):
    do: bool = Field(
        default=False,
        description="Whether to use low memory mode",
    )
    chunk_size: PositiveInt = Field(
        default=1,
        description="Chunk size for low memory mode",
    )


class MapToMapInputConfigMetrics(BaseModel):
    l2: Optional[dict] = Field(
        default=None,
        description="List of metrics to compute, and there own parameters",
    )
    corr: Optional[dict] = Field(
        default=None,
        description="List of metrics to compute, and there own parameters",
    )
    bioem: Optional[dict] = Field(
        default=None,
        description="List of metrics to compute, and there own parameters",
    )
    fsc: Optional[dict] = Field(
        default=None,
        description="List of metrics to compute, and there own parameters",
    )
    res: Optional[dict] = Field(
        default=None,
        description="List of metrics to compute, and there own parameters",
    )
    procrustes_wasserstein: Optional[dict] = Field(
        default=None,
        description="Extra parameters for the Procrustes Wasserstein distance",
    )
    gromov_wasserstein: Optional[dict] = Field(
        default=None,
        description="Extra parameters for the Gromov-Wasserstein distance",
    )
    zernike3d: Optional[dict] = Field(
        default=None,
        description="Extra parameters for the Zernike3D distance",
    )
    shared_params: Optional[Dict] = Field(
        description="Shared parameters for the metrics",
    )

    @field_validator("shared_params")
    @classmethod
    def validate_shared_params(cls, shared_params):
        if shared_params is not None:
            shared_params = dict(
                MapToMapInputConfigSharedParams(**shared_params).model_dump()
            )
        return shared_params

    @field_validator("l2")
    @classmethod
    def validate_l2_params(cls, l2_params):
        if l2_params is not None:
            l2_params = dict(
                MapToMapInputConfigMetricsL2BioemCorr(**l2_params).model_dump()
            )
        return l2_params

    @field_validator("corr")
    @classmethod
    def validate_corr_params(cls, corr_params):
        if corr_params is not None:
            corr_params = dict(
                MapToMapInputConfigMetricsL2BioemCorr(**corr_params).model_dump()
            )
        return corr_params

    @field_validator("bioem")
    @classmethod
    def validate_bioem_params(cls, bioem_params):
        if bioem_params is not None:
            bioem_params = dict(
                MapToMapInputConfigMetricsL2BioemCorr(**bioem_params).model_dump()
            )
        return bioem_params

    @field_validator("fsc")
    @classmethod
    def validate_fsc_params(cls, params):
        if params is not None:
            params = dict(MapToMapInputConfigMetricsFscRes(**params).model_dump())
        return params

    @field_validator("res")
    @classmethod
    def validate_res_params(cls, params):
        if params is not None:
            params = dict(MapToMapInputConfigMetricsFscRes(**params).model_dump())
        return params

    @field_validator("procrustes_wasserstein")
    @classmethod
    def validate_procrustes_wasserstein_params(cls, params):
        if params is not None:
            params = dict(
                MapToMapInputConfigMetricsProcrustesWasserstein(**params).model_dump()
            )
        return params

    @field_validator("gromov_wasserstein")
    @classmethod
    def validate_gromov_wasserstein_params(cls, params):
        if params is not None:
            params = dict(
                MapToMapInputConfigMetricsGromovWasserstein(**params).model_dump()
            )
        return params

    @field_validator("zernike3d")
    @classmethod
    def validate_zernike3d_params(cls, params):
        if params is not None:
            params = dict(MapToMapInputConfigMetricsZernike3d(**params).model_dump())
        return params


class MapToMapInputConfig(BaseModel, extra="forbid"):
    data_params: dict = Field(
        description="Parameters for loading the submission, ground truth and mask",
    )

    metrics: Dict[str, Dict] = Field(
        description="Dictionary of metrics to compute and their parameters, including shared parameters",
    )
    path_to_output_file: PurePath

    @field_validator("data_params")
    @classmethod
    def validate_data_params(cls, data_params):
        return dict(MapToMapInputConfigData(**data_params).model_dump())

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, metrics):
        return dict(MapToMapInputConfigMetrics(**metrics).model_dump(exclude_none=True))


class MapToMapResultsAllMetrics(BaseModel, extra="forbid"):
    """Validate the output dictionary of each map-to-map distance matrix computation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cost_matrix: pandas.DataFrame = Field(
        default=None,
        description="Cost matrix (ground truth versus submission)",
    )
    cost_matrix_self: pandas.DataFrame = Field(
        default=None,
        description="Cost matrix (submission versus submission)",
    )
    user_submission_label: str = Field(
        default=None,
        description="User submission label",
    )
    computed_assets: dict = Field(
        default=None,
        description="Computed assets",
    )
    computed_assets_self: dict = Field(
        default=None,
        description="Computed assets",
    )


class MapToMapResultsValidator(BaseModel, extra="forbid"):
    """
    Validate the output dictionary of the map-to-map distance matrix computation.

    config: dict, input config dictionary.
    user_submitted_populations: torch.Tensor, user submitted populations, which sum to 1.
    corr: dict, correlation results.
    l2: dict, L2 results.
    bioem: dict, BioEM results.
    fsc: dict, FSC results.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    config: dict = Field(
        description="Input config dictionary",
    )
    user_submitted_populations: Tensor = Field(
        description="User submitted populations, which sum to 1",
    )
    corr: Optional[dict] = Field(
        default=None,
        description="Correlation results",
    )
    l2: Optional[dict] = Field(
        default=None,
        description="L2 results",
    )
    bioem: Optional[dict] = Field(
        default=None,
        description="BioEM results",
    )
    fsc: Optional[dict] = Field(
        default=None,
        description="FSC results",
    )
    res: Optional[dict] = Field(
        default=None,
        description="FSC res results",
    )
    zernike3d: Optional[dict] = Field(
        default=None,
        description="Zernike3D results",
    )
    gromov_wasserstein: Optional[dict] = Field(
        default=None,
        description="Gromov-Wasserstein results",
    )
    procrustes_wasserstein: Optional[dict] = Field(
        default=None,
        description="Procrustes-Wasserstein results",
    )

    @field_validator("config")
    @classmethod
    def validate_config(cls, config):
        return dict(MapToMapInputConfig(**config).model_dump())

    @field_validator("user_submitted_populations")
    @classmethod
    def validate_user_submitted_populations(cls, user_submitted_populations):
        assert torch.isclose(
            torch.ones(1).to(user_submitted_populations.dtype),
            user_submitted_populations.sum(),
        ), "User submitted populations do not sum to 1."
        return user_submitted_populations

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        if self.corr is not None:
            self.corr = dict(MapToMapResultsAllMetrics(**self.corr).model_dump())
        if self.l2 is not None:
            self.l2 = dict(MapToMapResultsAllMetrics(**self.l2).model_dump())
        if self.bioem is not None:
            self.bioem = dict(MapToMapResultsAllMetrics(**self.bioem).model_dump())
        if self.fsc is not None:
            self.fsc = dict(MapToMapResultsAllMetrics(**self.fsc).model_dump())
        if self.res is not None:
            self.res = dict(MapToMapResultsAllMetrics(**self.res).model_dump())
        if self.procrustes_wasserstein is not None:
            self.procrustes_wasserstein = dict(
                MapToMapResultsAllMetrics(**self.procrustes_wasserstein).model_dump()
            )
        if self.gromov_wasserstein is not None:
            self.gromov_wasserstein = dict(
                MapToMapResultsAllMetrics(**self.gromov_wasserstein).model_dump()
            )
        if self.zernike3d is not None:
            self.zernike3d = dict(
                MapToMapResultsAllMetrics(**self.zernike3d).model_dump()
            )
        return self
