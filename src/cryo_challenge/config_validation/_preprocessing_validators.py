from typing import Optional, Annotated, Literal
import glob
import os
from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    PositiveInt,
    FilePath,
    DirectoryPath,
    AfterValidator,
)

from pathlib import PurePath


def _is_mrc_file(filename: FilePath) -> bool:
    """
    Check if the file is a .mrc file.
    """
    assert filename.suffix == ".mrc", f"File {filename} is not a .mrc file"
    return filename


def _contains_mrc_files_minus_mask(path_to_volumes: str) -> bool:
    """
    Check if path_to_volumes contains .mrc files.
    """
    vol_paths = glob.glob(os.path.join(path_to_volumes, "*.mrc"))
    vol_paths = [vol_path for vol_path in vol_paths if "mask" not in vol_path]
    assert len(vol_paths) > 0, "No .mrc files found in the directory"
    return path_to_volumes


class PreprocessingDatasetSubmissionConfig(BaseModel, extra="forbid"):
    path_to_volumes: Annotated[
        DirectoryPath, AfterValidator(_contains_mrc_files_minus_mask)
    ] = Field(
        description="Path to the submitted volumes in .mrc format",
    )
    path_to_populations_file: FilePath = Field(
        description="Path to the populations file in .txt format",
    )
    submission_name: str = Field(
        description="Name of the submitted",
    )
    submission_version: str = Field(
        description="Version of the submission",
    )
    ice_cream_flavor: str = Field(
        description="Ice cream flavor assigned to the submitted",
    )
    box_size: PositiveInt = Field(
        description="Box size of the submitted volumes",
    )
    voxel_size: PositiveFloat = Field(
        description="Voxel size of the submitted volumes",
    )

    threshold_percentile: PositiveFloat = Field(
        description="Percentile to use for thresholding the submitted volumes",
    )

    flip_handedness: Optional[bool] = Field(
        default=False,
        description="Whether to flip the handedness of the submitted volumes",
    )
    align: Optional[bool] = Field(
        default=False,
        description="Whether to align the submitted volumes to a reference volume",
    )


class PreprocessingDatasetReferenceConfig(BaseModel, extra="forbid"):
    path_to_reference_volume: Annotated[FilePath, AfterValidator(_is_mrc_file)] = Field(
        description="Path to the reference volume in .mrc format",
    )

    box_size: PositiveInt = Field(
        description="Box size of the submitted volumes",
    )
    voxel_size: PositiveFloat = Field(
        description="Voxel size of the submitted volumes",
    )


class PreprocessingRunConfig(BaseModel, extra="forbid"):
    """
    Configuration for the preprocessing run.
    """

    box_size_for_BOTAlign: PositiveInt = Field(
        default=32,
        description="Box size used for volume alignment",
    )

    loss_for_BOTAlign: Literal["wemd", "l2"] = Field(
        default="wemd",
        description="Loss function used for volume alignment",
    )

    num_iterations_for_BOTAlign: PositiveInt = Field(
        default=200,
        description="Number of iterations for volume alignment",
    )

    run_refinement_BOTAlign: bool = Field(
        default=True,
        description="Whether to run refinement for volume alignment",
    )

    output_path: PurePath = Field(
        description="Path to save the preprocessed volumes",
    )

    path_to_submissions_config: FilePath = Field(
        description="Path to the file with configs for the submissions",
    )

    path_to_reference_config: FilePath = Field(
        description="Path to the file with configs for the reference",
    )
