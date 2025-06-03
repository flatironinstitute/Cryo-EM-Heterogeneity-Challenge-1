from typing import List, Any, Optional, Union
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
    confloat,
)
from pathlib import PurePath
from torch import Tensor


TolerantPositiveFloat = confloat(
    ge=-1e-6
)  # numerical issues in CVXPY can cause negative values


class DistToDistInputConfigEmdRegularization(BaseModel, extra="forbid"):
    cvxpy_solve_kwargs: dict = Field(
        description="Keyword arguments for the CVXPY solver.",
        default={},
    )
    scalar_hyperparam_self_emd: PositiveFloat = Field(
        default=0.0,
        description="Scalar hyperparameter for the self EMD regularization.",
    )
    scalar_hyperparam_self_entropy_q: PositiveFloat = Field(
        default=0.0,
        description="Scalar hyperparameter for the self entropy regularization.",
    )
    scalar_hyperparam_weighted_l2_in_cost: PositiveFloat = Field(
        default=0.0,
        description="Scalar hyperparameter for the weighted L2 regularization by the self cost matrix.",
    )
    q_greater_than_constraint: PositiveFloat = Field(
        description="Epsilon for numerical stability of the entropy regularization.",
        default=1e-6,
    )


class DistToDistInputConfigOptimalQKL(BaseModel, extra="forbid"):
    n_iter: PositiveInt = Field(
        description="Number of iterations for the optimization."
    )
    break_atol: PositiveFloat = Field(
        description="Absolute tolerance for the optimization.",
    )


class DistToDistInputConfigReplicateParams(BaseModel, extra="forbid"):
    n_replicates: PositiveInt = Field(
        default=1,
        description="Number of replicates to compute",
    )

    n_pool_ground_truth_microstates: PositiveInt = Field(
        default=1,
        description="Number of microstates to pool (low values less than 3-5 can cause problems for optimization convergence in CVXPY numerical solvers).",
    )
    replicate_fraction: PositiveFloat = Field(
        description="Fraction of the data to use for replicates."
    )


class DistToDistInputConfigMetrics(BaseModel, extra="forbid"):
    l2: Optional[dict] = Field(
        default=None,
        description="L2 metric parameters.",
    )
    bioem: Optional[dict] = Field(
        default=None,
        description="BioEM metric parameters.",
    )
    res: Optional[dict] = Field(
        default=None,
        description="RES metric parameters.",
    )
    fsc: Optional[dict] = Field(
        default=None,
        description="FSC metric parameters.",
    )
    corr: Optional[dict] = Field(
        default=None,
        description="Correlation metric parameters.",
    )
    zernike3d: Optional[dict] = Field(
        default=None,
        description="Zernike3D metric parameters.",
    )
    gromov_wasserstein: Optional[dict] = Field(
        default=None,
        description="Gromov-Wasserstein metric parameters.",
    )
    procrustes_wasserstein: Optional[dict] = Field(
        default=None,
        description="Procrustes-Wasserstein metric parameters.",
    )

    @field_validator("l2")
    @classmethod
    def validate_l2(cls, params):
        if params is not None:
            return dict(DistToDistInputConfigSingleMetricParams(**params).model_dump())
        else:
            return None

    @field_validator("bioem")
    @classmethod
    def validate_bioem(cls, params):
        if params is not None:
            return dict(DistToDistInputConfigSingleMetricParams(**params).model_dump())
        else:
            return None

    @field_validator("res")
    @classmethod
    def validate_res(cls, params):
        if params is not None:
            return dict(DistToDistInputConfigSingleMetricParams(**params).model_dump())
        else:
            return None

    @field_validator("fsc")
    @classmethod
    def validate_fsc(cls, params):
        if params is not None:
            return dict(DistToDistInputConfigSingleMetricParams(**params).model_dump())
        else:
            return None

    @field_validator("corr")
    @classmethod
    def validate_corr(cls, params):
        if params is not None:
            return dict(DistToDistInputConfigSingleMetricParams(**params).model_dump())
        else:
            return None

    @field_validator("zernike3d")
    @classmethod
    def validate_zernike3d(cls, params):
        if params is not None:
            return dict(DistToDistInputConfigSingleMetricParams(**params).model_dump())
        else:
            return None

    @field_validator("gromov_wasserstein")
    @classmethod
    def validate_gromov_wasserstein(cls, params):
        if params is not None:
            return dict(DistToDistInputConfigSingleMetricParams(**params).model_dump())
        else:
            return None

    @field_validator("procrustes_wasserstein")
    @classmethod
    def validate_procrustes_wasserstein(cls, params):
        if params is not None:
            return dict(DistToDistInputConfigSingleMetricParams(**params).model_dump())
        else:
            return None


class DistToDistInputConfigSingleMetricParams(BaseModel, extra="forbid"):
    apply_rank_normalization: Optional[bool] = Field(
        default=True,
        description="Apply rank normalization to the cost matrix.",
    )
    metric_specific_params: Optional[dict] = Field(
        default={},
        description="Metric specific parameters.",
    )


class DistToDistInputConfig(BaseModel, extra="forbid"):
    path_to_map_to_map_results: FilePath = Field(
        description="Path to the map-to-map results file",
    )
    metrics: dict[str, dict] = Field(
        default={},
        description="Dictionary of metrics to compute. If None, the metric is not computed.",
    )
    path_to_ground_truth_metadata: FilePath = Field(
        description="Path to the ground truth metadata file",
    )

    replicate_params: dict = Field(
        description="Dictionary of replicates to compute",
        default=None,
    )

    cvxpy_solve_kwargs: dict = Field(
        description="Keyword arguments for the CVXPY solver.",
        default={},
    )

    emd_regularization: dict = Field(
        description="Parameters for the optimal q KL divergence.",
    )
    optimal_q_kl_params: dict = Field(
        description="Parameters for the optimal q KL divergence.",
    )

    path_to_output_file: PurePath = Field(
        description="Path to the output file",
    )

    @field_validator("emd_regularization")
    @classmethod
    def validate_regularization(cls, params):
        return dict(DistToDistInputConfigEmdRegularization(**params).model_dump())

    @field_validator("optimal_q_kl_params")
    @classmethod
    def validate_optimal_q_kl_params(cls, params):
        return dict(DistToDistInputConfigOptimalQKL(**params).model_dump())

    @field_validator("cvxpy_solve_kwargs")
    @classmethod
    def validate_cvxpy_solve_kwargs(cls, params):
        assert isinstance(params, dict), "cvxpy_solve_kwargs must be a dictionary."
        assert "solver" in params, "cvxpy_solve_kwargs must contain a 'solver' key."
        supported_solvers = [
            "ECOS",
            "CVXOPT",
            "CLARABEL",
            "GUROBI",
            "SCS",
            "MOSEK",
        ]
        if params["solver"] not in supported_solvers:
            raise ValueError(
                f"Solver {params['solver']} is not supported. Supported solvers are: {supported_solvers}."
            )
        return params

    @field_validator("replicate_params")
    @classmethod
    def validate_replicates(cls, params):
        return dict(DistToDistInputConfigReplicateParams(**params).model_dump())

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, params):
        if params is not None:
            return dict(
                DistToDistInputConfigMetrics(**params).model_dump(exclude_none=True)
            )
        else:
            return None


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

    q_opt: List[TolerantPositiveFloat]
    EMD_opt: PositiveFloat
    transport_plan_opt: List[List[float]]
    flow_opt: Any
    prob_opt: Any
    runtime_opt: float
    q_opt_reg: List[PositiveFloat]
    EMD_opt_reg: float
    transport_plan_opt_reg: List[List[float]]
    transport_plan_opt_self: Optional[Union[List[List[float]], None]]
    flow_opt_reg: Any
    prob_opt_reg: Any
    runtime_opt_reg: float
    EMD_submitted: float
    transport_plan_submitted: List[List[float]]


class DistToDistResultsValidatorReplicateKL(BaseModel, extra="forbid"):
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
    objective: List[float], objective function values at each iteration.
    """

    q_opt: List[TolerantPositiveFloat]
    klpq_opt: float
    klqp_opt: float
    A: List[List[float]]
    iter_stop: int
    eps_stop: float
    klpq_submitted: float
    klqp_submitted: float
    objective: List[float]


class DistToDistResultsValidatorMetrics(BaseModel, extra="forbid"):
    replicates: dict
    cost_self: Optional[List[List[float]]] = Field(
        description="Cost matrix for the self transport plan.",
        default=None,
    )

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
            len(metric["replicates"]) == config["replicate_params"]["n_replicates"]
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
