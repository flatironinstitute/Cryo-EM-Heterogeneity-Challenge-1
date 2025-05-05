import torch
from typing import Optional, Annotated, Literal, List
import glob
import os
from pathlib import PurePath
from pydantic import (
    BaseModel,
    NonNegativeInt,
    PositiveInt,
    FilePath,
    DirectoryPath,
    AfterValidator,
    Field,
    field_validator,
    model_validator,
)


def _contains_submission_files(path_to_submissions: DirectoryPath) -> DirectoryPath:
    submission_files = glob.glob(os.path.join(path_to_submissions, "*.pt"))
    submission_files = [PurePath(file) for file in submission_files]
    submission_files = [file for file in submission_files if "submission" in file.name]
    assert len(submission_files) > 0, "No submission files found in the directory"
    return path_to_submissions


def _is_pt_file(filename: FilePath) -> FilePath:
    assert filename.suffix == ".pt", f"File {filename} is not a .pt file"
    return filename


class SVDInputConfigNormalize(BaseModel, extra="forbid"):
    path_to_mask: Optional[FilePath] = Field(
        default=None,
        description="Path to the mask file. If None, no mask is applied.",
    )
    downsample_box_size: Optional[PositiveInt] = Field(
        default=None,
        description="Downsampled box size. If None, no downsampling is applied.",
    )
    threshold_percentile: Optional[Annotated[float, Field(ge=0.0, le=100.0)]] = Field(
        default=None,
        description="Threshold percentile. If None, no thresholding is applied.",
    )
    normalize_power_spectrum: bool = Field(
        default=True,
        description="Normalize power spectrum. If True, the power spectrum is normalized.",
    )


class SVDInputConfigGT(BaseModel, extra="forbid"):
    path_to_gt_volumes: Annotated[FilePath, AfterValidator(_is_pt_file)] = Field(
        description="Path to the ground truth volumes file in .pt format",
    )
    skip_vols: NonNegativeInt = Field(
        default=1,
        description="Number of volumes to skip in the ground truth file",
    )

    @field_validator("path_to_gt_volumes")
    def check_volumes_shape(cls, value):
        vols_gt = torch.load(value, mmap=True, weights_only=False)

        if len(vols_gt.shape) not in [2, 4]:
            raise ValueError(
                "Ground truth volumes must have shape: (N, D, D, D) or (N, D*D*D)"
            )
        return value


class SVDInputConfigOutput(BaseModel, extra="forbid"):
    path_to_output_dir: PurePath = Field(
        description="Path to the output directory",
    )
    keep_prep_submissions_for_svd: bool = Field(
        default=False,
        description="Save data for continue. If True, this will save the prepared data for continue.",
    )
    generate_plots: bool = Field(
        default=False,
        description="Generate plots. If True, this will generate plots for the analysis.",
    )
    overwrite: bool = Field(
        default=False,
        description="Overwrite existing files. If True, this will overwrite existing files.",
    )


class SVDInputConfig(BaseModel, extra="forbid"):
    # Main configuration fields
    path_to_submissions: Annotated[
        DirectoryPath, AfterValidator(_contains_submission_files)
    ] = Field(
        description="Path to the submission files in .pt format",
    )
    excluded_submissions: List[str] = Field(
        default=[],
        description="List of submission files in `path_to_submissions` to exclude from the analysis",
    )
    dtype: Literal["float32", "float64"] = Field(
        default="float32",
        description="Data to be used for analysis",
    )
    svd_max_rank: Optional[PositiveInt] = Field(
        default=None,
        description="Maximum rank for SVD. If None, no limit is applied.",
    )
    continue_from_previous: bool = Field(
        default=False,
        description="Continue from previous analysis. If True, this will load the previously prepared data.",
    )

    # Subdictionaries
    normalize_params: dict = Field(
        default={},
        description="Normalization parameters",
    )
    gt_params: Optional[dict] = Field(
        default=None,
        description="Ground truth parameters",
    )
    output_params: dict = Field(
        description="Parameters for the output of the method",
    )

    @field_validator("normalize_params")
    def check_normalize_params(cls, value):
        return dict(SVDInputConfigNormalize(**value).model_dump())

    @field_validator("gt_params")
    def check_gt_params(cls, value):
        if value is not None:
            value = dict(SVDInputConfigGT(**value).model_dump())
        return value

    @field_validator("output_params")
    def check_output_params(cls, value):
        return dict(SVDInputConfigOutput(**value).model_dump())

    @model_validator(mode="after")
    def check_submission_files_not_empty(self):
        path_to_submissions = self.path_to_submissions
        excluded_submissions = self.excluded_submissions

        submission_files = glob.glob(os.path.join(path_to_submissions, "*.pt"))
        submission_files = [PurePath(file) for file in submission_files]
        filtered_submission_files = []
        for file in submission_files:
            if "submission" in file.name:
                if file.name not in excluded_submissions:
                    filtered_submission_files.append(file)

        if len(submission_files) == 0:
            raise ValueError(
                f"No submission files found after excluding {excluded_submissions}."
            )

        return self
