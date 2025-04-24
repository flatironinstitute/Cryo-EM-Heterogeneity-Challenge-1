from typing import Optional, Annotated, Literal, List
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
    FilePath,
    DirectoryPath,
    AfterValidator,
    Field,
    field_validator,
    model_validator,
)
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


class MapToMapInputConfigAnalysisNormalize(BaseModel, extra="forbid"):
    do: bool = Field(
        default=False,
        description="Whether to normalize the volumes",
    )
    method: Literal["median_zscore"] = Field(
        default="median_zscore",
        description="Method to use for normalization",
    )


class MapToMapInputConfigAnalysisLowMemory(BaseModel, extra="forbid"):
    do: bool = Field(
        default=False,
        description="Whether to use low memory mode",
    )

    chunk_size: PositiveInt = Field(
        default=1,
        description="Chunk size for low memory mode",
    )


class MapToMapInputConfigAnalysisGromovWasserstein(BaseModel, extra="forbid"):
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
    element_wise: bool = Field(
        description="dask parralelization: whether to call dask.compute on each map-to-map computation, or naively loop through each (80) submitted maps row-wise"
    )
    slurm: bool = Field(
        description="parallelization configuration: whether to use dask_hpc_runner.SlurmRunner as a runner for dask.Client(runner)"
    )

    scheduler: Optional[str] = Field(
        default=None,
        description="string argument to dask.compute",
    )

    local_directory: Optional[DirectoryPath] = Field(
        default=None, description="directory for dask.distributed.Client"
    )


class MapToMapInputConfigAnalysisZernike3d(BaseModel, extra="forbid"):
    gpuID: PositiveInt = Field(
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


class MapToMapInputConfigAnalysis(BaseModel):
    metrics: List[
        Literal[
            "fsc",
            "corr",
            "l2",
            "bioem",
            "res",
            "zernike3d",
            "gromov_wasserstein",
            "procrustes_wasserstein",
        ]
    ] = Field(
        description="List of metrics to compute",
    )

    chunk_size_submission: PositiveInt = Field(
        default=1,
        description="Chunk size for the submission volumes",
    )
    chunk_size_gt: PositiveInt = Field(
        default=1,
        description="Chunk size for the ground truth volumes",
    )

    normalize_params: dict = Field(
        description="Parameters for the normalization of the volumes",
    )

    low_memory: dict = Field(
        description="Parameters for the low memory mode",
    )

    gromov_wasserstein_extra_params: Optional[dict] = Field(
        default=None,
        description="Extra parameters for the Gromov-Wasserstein distance",
    )

    zernike3d_extra_params: Optional[dict] = Field(
        default=None,
        description="Extra parameters for the Zernike3D distance",
    )  # TODO!!!!

    procrustes_wasserstein_extra_params: Optional[dict] = Field(
        default=None,
        description="Extra parameters for the Procrustes Wasserstein distance",
    )  # TODO!!!!

    @field_validator("normalize_params")
    @classmethod
    def validate_normalize_params(cls, normalize_params):
        return dict(
            MapToMapInputConfigAnalysisNormalize(**normalize_params).model_dump()
        )

    @field_validator("low_memory")
    @classmethod
    def validate_low_memory(cls, low_memory):
        return dict(MapToMapInputConfigAnalysisLowMemory(**low_memory).model_dump())

    @field_validator("gromov_wasserstein_extra_params")
    @classmethod
    def validate_gromov_wasserstein_params(cls, gromov_wasserstein_extra_params):
        if gromov_wasserstein_extra_params is not None:
            gromov_wasserstein_extra_params = dict(
                MapToMapInputConfigAnalysisGromovWasserstein(
                    **gromov_wasserstein_extra_params
                ).model_dump()
            )
        return gromov_wasserstein_extra_params

    @field_validator("zernike3d_extra_params")
    @classmethod
    def validate_zernike3d__params(cls, zernike3d_extra_params):
        if zernike3d_extra_params is not None:
            zernike3d_extra_params = dict(
                MapToMapInputConfigAnalysisZernike3d(
                    **zernike3d_extra_params
                ).model_dump()
            )
        return zernike3d_extra_params


class MapToMapInputConfig(BaseModel, extra="forbid"):
    data_params: dict = Field(
        description="Parameters for loading the submission, ground truth and mask",
    )
    analysis: dict = Field(
        description="Parameters for the analysis",
    )
    output: PurePath

    @field_validator("data_params")
    @classmethod
    def validate_data_params(cls, data_params):
        return dict(MapToMapInputConfigData(**data_params).model_dump())

    @field_validator("analysis")
    @classmethod
    def validate_analysis(cls, analysis):
        return dict(MapToMapInputConfigAnalysis(**analysis).model_dump())


def _validate_metric_in_config(metric, metric_name, config):
    if metric is None:
        assert (
            metric_name not in config["analysis"]["metrics"]
        ), f"{metric_name} metric is not computed, but is requested."
    return metric


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

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    # put dataclass here???

    @field_validator("config")
    @classmethod
    def validate_config(cls, config):
        return dict(MapToMapInputConfig(**config).model_dump())

    @field_validator("user_submitted_populations")
    @classmethod
    def validate_user_submitted_populations(cls, user_submitted_populations):
        assert (
            user_submitted_populations.sum() == 1
        ), "User submitted populations do not sum to 1."
        return user_submitted_populations

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        self.corr = _validate_metric_in_config(self.corr, "corr", self.config)
        self.l2 = _validate_metric_in_config(self.l2, "l2", self.config)
        self.bioem = _validate_metric_in_config(self.bioem, "bioem", self.config)
        self.fsc = _validate_metric_in_config(self.fsc, "fsc", self.config)
        self.res = _validate_metric_in_config(self.res, "res", self.config)
        self.zernike3d = _validate_metric_in_config(
            self.zernike3d, "zernike3d", self.config
        )
        self.gromov_wasserstein = _validate_metric_in_config(
            self.gromov_wasserstein, "gromov_wasserstein", self.config
        )
        self.procrustes_wasserstein = _validate_metric_in_config(
            self.procrustes_wasserstein, "procrustes_wasserstein", self.config
        )
        return self
