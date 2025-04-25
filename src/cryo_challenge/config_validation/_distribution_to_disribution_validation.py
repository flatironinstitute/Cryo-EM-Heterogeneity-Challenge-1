from typing import Literal, List, Any, Optional
from typing_extensions import Self
from pydantic import (
    BaseModel,
    ConfigDict,
    PositiveFloat,
    PositiveInt,
    FilePath,
    Field,
    field_validator,
    model_validator,
)
from pathlib import PurePath
from torch import Tensor


class DistToDistInputConfigOptimalQKL(BaseModel, extra="forbid"):
    n_iter: PositiveInt = Field(
        description="Number of iterations for the optimization."
    )
    break_atol: PositiveFloat = Field(
        description="Absolute tolerance for the optimization.",
    )


class DistToDistInputConfig(BaseModel, extra="forbid"):
    path_to_map_to_map_results: FilePath = Field(
        description="Path to the map-to-map results file",
    )
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

    path_to_ground_truth_metadata: FilePath = Field(
        description="Path to the ground truth metadata file",
    )

    n_replicates: PositiveInt = Field(
        default=1,
        description="Number of replicates to compute",
    )

    n_pool_microstate: PositiveInt = Field(
        default=1,
        description="Number of microstates to pool (low values less than 3-5 can cause problems for optimization convergence in CVXPY numerical solvers).",
    )
    replicate_fraction: PositiveFloat = Field(
        description="Fraction of the data to use for replicates."
    )
    cvxpy_solver: Literal["ECOS", "CVXOPT", "CLARABEL", "GUROBI", "SCS", "MOSEK"] = (
        Field(
            default="ECOS",
            description="CVXPY solver to use for optimization. See https://www.cvxpy.org/tutorial/solvers/index.html.",
        )
    )
    optimal_q_kl_params: dict = Field(
        description="Parameters for the optimal q KL divergence.",
    )

    path_to_output_file: PurePath = Field(
        description="Path to the output file",
    )

    @field_validator("optimal_q_kl_params")
    @classmethod
    def validate_optimal_q_kl_params(cls, params):
        return dict(DistToDistInputConfigOptimalQKL(**params).model_dump())


class DistToDistResultsValidatorReplicateEMD(BaseModel, extra="forbid"):
    """
    Validate the output dictionary of one EMD in the the distribution-to-distribution pipeline.

    q_opt: List[float], optimal user submitted distribution, which sums to 1.
    EMD_opt: float, EMD between the ground truth distribution (p) and the (optimized) user submitted distribution (q_opt).
        The transport plan is a joint distribution, such that:
        summing over the rows gives the (optimized) user submitted distribution, and summing over the columns gives the ground truth distribution.
    transport_plan_opt: List[List[float]], transport plan between the ground truth distribution (p, rows) and the (optimized) user submitted distribution (q_opt, columns).
    flow_opt: cvx output, optimization output (problem is formulated as a network flow problem).
    prob_opt: cvx output, optimization 'problem' output.
    runtime_opt: float, runtime of the optimization.
    EMD_submitted: float, EMD between the ground truth distribution (p) and the user submitted distribution (q).
    transport_plan_submitted: List[List[float]], transport plan between the ground truth distribution (p, rows) and the user submitted distribution (q, columns).
        The transport plan is a joint distribution, such that:
        summing over the rows gives the user submitted distribution, and summing over the columns gives the ground truth distribution.
    """

    q_opt: List[PositiveFloat]
    EMD_opt: PositiveFloat
    transport_plan_opt: List[List[float]]
    flow_opt: Any
    prob_opt: Any
    runtime_opt: float
    EMD_submitted: float
    transport_plan_submitted: List[List[float]]


class DistToDistResultsValidatorReplicateKL(
    BaseModel, extra="forbid", arbitrary_types_allowed=True
):
    """
    Validate the output dictionary of one KL divergence in the the distribution-to-distribution pipeline.

    q_opt: List[float], optimal user submitted distribution, which sums to 1.
    klpq_opt: float, KL divergence between the ground truth distribution (p) and the (optimized) user submitted distribution (q_opt).
    klqp_opt: float, KL divergence between the (optimized) user submitted distribution and the ground truth distribution (p).
    A: List[List[float]], assignment matrix.
    iter_stop: int, number of iterations until convergence.
    eps_stop: float, stopping tolerance.
    klpq_submitted: float, KL divergence between the ground truth distribution (p) and the user submitted distribution (q).
    klqp_submitted: float, KL divergence between the user submitted distribution (q) and the ground truth distribution (p).
    """

    q_opt: List[PositiveFloat]
    klpq_opt: float
    klqp_opt: float
    A: List[List[float]]
    iter_stop: int
    eps_stop: float
    klpq_submitted: float
    klqp_submitted: float
    objective: Tensor


class DistToDistResultsValidatorMetrics(BaseModel, extra="forbid"):
    replicates: dict

    @field_validator("replicates")
    @classmethod
    def validate_replicates(cls, replicates):
        for key, replicate in replicates.items():
            replicates[key]["EMD"] = dict(
                DistToDistResultsValidatorReplicateEMD(**replicate["EMD"]).model_dump()
            )
            replicates[key]["KL"] = dict(
                DistToDistResultsValidatorReplicateKL(**replicate["KL"]).model_dump()
            )

        return replicates


def _validate_metric(metric, metric_name, config):
    if metric is None:
        assert (
            metric_name not in config["metrics"]
        ), f"{metric_name} metric is not computed, but is requested."
    else:
        metric = dict(DistToDistResultsValidatorMetrics(**metric).model_dump())
        assert (
            len(metric["replicates"]) == config["n_replicates"]
        ), f"Replicates in {metric_name} metric do not match the number of replicates in the config."
    return metric


class DistToDistResultsValidator(BaseModel, extra="forbid"):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: dict
    user_submitted_populations: Tensor
    id: str
    fsc: Optional[dict] = None
    bioem: Optional[dict] = None
    res: Optional[dict] = None
    l2: Optional[dict] = None
    corr: Optional[dict] = None
    zernike3d: Optional[dict] = None
    gromov_wasserstein: Optional[dict] = None

    @field_validator("config")
    @classmethod
    def validate_config(cls, config):
        return dict(DistToDistInputConfig(**config).model_dump())

    @model_validator(mode="after")
    def validate_metrics_and_replicates(self) -> Self:
        self.fsc = _validate_metric(self.fsc, "fsc", self.config)
        self.bioem = _validate_metric(self.bioem, "bioem", self.config)
        self.res = _validate_metric(self.res, "res", self.config)
        self.l2 = _validate_metric(self.l2, "l2", self.config)
        self.corr = _validate_metric(self.corr, "corr", self.config)
        self.zernike3d = _validate_metric(self.zernike3d, "zernike3d", self.config)
        self.gromov_wasserstein = _validate_metric(
            self.gromov_wasserstein, "gromov_wasserstein", self.config
        )

        return self
