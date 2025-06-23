from typing import Optional, Annotated, Literal, Dict, Any, List
import glob
import os
from pydantic import (
    BaseModel,
    Field,
    PositiveFloat,
    NonNegativeFloat,
    PositiveInt,
    FilePath,
    DirectoryPath,
    AfterValidator,
    field_validator,
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


class PreprocessingSubmissionConfigInitRotation(BaseModel, extra="forbid"):
    seq: str = Field(
        default="x",
        description="Sequence of rotations to apply to the volume.",
    )
    angles: float | List[float] = Field(
        default=0.0,
        description="Angle of rotation in degrees.",
    )
    degrees: bool = Field(
        default=True,
        description="Whether the angle is in degrees or radians.",
    )


class PreprocessingSubmissionConfig(BaseModel, extra="forbid"):
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

    threshold_percentile: NonNegativeFloat = Field(
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
    initial_rotation_guess: Optional[Dict] = Field(
        default=PreprocessingSubmissionConfigInitRotation().model_dump(),
        description=(
            "Parameters for initializating Scipy Rotation object from euler for an initial guess of the rotation matrix."
        ),
    )

    @field_validator("initial_rotation_guess", mode="after")
    @classmethod
    def validate_initial_rotation_guess(cls, values):
        return dict(PreprocessingSubmissionConfigInitRotation(**values).model_dump())


class PreprocessingReferenceConfig(BaseModel, extra="forbid"):
    path_to_reference_volume: Annotated[FilePath, AfterValidator(_is_mrc_file)] = Field(
        description="Path to the reference volume in .mrc format",
    )

    box_size: PositiveInt = Field(
        description="Box size of the submitted volumes",
    )
    voxel_size: PositiveFloat = Field(
        description="Voxel size of the submitted volumes",
    )


class PreprocessingRunConfigBOTAlign(BaseModel, extra="forbid"):
    loss_type: Literal["wemd", "euclidean"] = Field(
        default="wemd",
        description="If the heterogeneity between vol_ref and vol_given is high, 'euclidean' is recommended.",
    )
    loss_params: Optional[Any] = Field(
        default=None,
        description="dictionary for overriding parameters in aspire.utils.bot_align.loss_types. Defaults to empty dictionary.",
    )
    downsampled_size: PositiveInt = Field(
        default=32,
        description="Downsampling (pixels). Integer, defaults to 32. If alignment fails try larger values.",
    )
    refinement_downsampled_size: PositiveInt = Field(
        default=32,
        description="Downsampling (pixels) used with refinement. Integer, defaults to 32.",
    )
    max_iters: PositiveInt = Field(
        default=200,
        description="Maximum iterations. Integer, defaults 200. If alignment fails try larger values.",
    )
    refine: bool = Field(
        default=True,
        description="Whether to perform refinement. Boolean, defaults True.",
    )
    tau: PositiveFloat = Field(
        default=1e-3,
        description="Regularization parameter for surrogate problems. Numeric, defaults 1e-3.",
    )
    surrogate_max_iter: PositiveInt = Field(
        default=500,
        description="Stopping criterion for surrogate problems--maximum iterations. Integer, defaults 500.",
    )
    surrogate_min_grad: PositiveFloat = Field(
        default=0.1,
        description="Stopping criterion for surrogate problems--minimum gradient norm. Numeric, defaults 0.1.",
    )
    surrogate_min_step: PositiveFloat = Field(
        default=0.1,
        description="Stopping criterion for surrogate problems--minimum step size. Numeric, defaults 0.1.",
    )
    verbosity: Literal[0, 1, 2] = Field(
        default=0,
        description="Surrogate problem optimization detail level. Integer, defaults 0 (silent). 2 is most verbose.",
    )
    dtype: Optional[Any] = Field(
        default=None,
        description="Numeric dtype to perform computations with. Default None infers dtype from vol_ref.",
    )


class PreprocessingRunConfig(BaseModel, extra="forbid"):
    """
    Configuration for the preprocessing run.
    """

    BOTAlign_params: Dict = Field(
        default={},
        description=(
            "Parameters for the BOTAlign algorithm."
            + "See aspire.utils.bot_align.align_BO"
        ),
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

    @field_validator("BOTAlign_params", mode="after")
    @classmethod
    def validate_BOTAlign_params(cls, values):
        return dict(PreprocessingRunConfigBOTAlign(**values).model_dump())
