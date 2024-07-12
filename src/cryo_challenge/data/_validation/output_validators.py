import torch
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List, Any, Optional

from .config_validators import (
    validate_input_config_disttodist,
    validate_input_config_mtm,
    validate_maptomap_result,
)


@dataclass_json
@dataclass
class MapToMapResultsValidator:
    '''
    Validate the output dictionary of the map-to-map distance matrix computation.

    config: dict, input config dictionary.
    user_submitted_populations: torch.Tensor, user submitted populations, which sum to 1.
    corr: dict, correlation results.
    l2: dict, L2 results.
    bioem: dict, BioEM results.
    fsc: dict, FSC results.
    '''
    config: dict
    user_submitted_populations: torch.Tensor
    corr: Optional[dict] = None
    l2: Optional[dict] = None
    bioem: Optional[dict] = None
    fsc: Optional[dict] = None
    res: Optional[dict] = None

    def __post_init__(self):
        validate_input_config_mtm(self.config)

        for metric in self.config["analysis"]["metrics"]:
            assert self.__dict__[metric] is not None

        for metric in self.config["analysis"]["metrics"]:
            validate_maptomap_result(self.__dict__[metric])

        return


@dataclass_json
@dataclass
class ReplicateValidatorEMD:
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
    q_opt: List[float]
    EMD_opt: float
    transport_plan_opt: List[List[float]]
    flow_opt: Any
    prob_opt: Any
    runtime_opt: float
    EMD_submitted: float
    transport_plan_submitted: List[List[float]]

    def __post_init__(self):
        pass


@dataclass_json
@dataclass
class ReplicateValidatorKL:
    """
    Validate the output dictionary of one KL divergence in the the distribution-to-distribution pipeline.

    q_opt: List[float], optimal user submitted distribution, which sums to 1.
    klpq_opt: float, KL divergence between the ground truth distribution (p) and the (optimized) user submitted distribution (q_opt).
    klqp_opt: float, KL divergence between the (optimized) user submitted distribution and the ground truth distribution (p).
    A: List[List[float]], assignment matrix.
    iter_stop: int, number of iterations until convergence.
    eps_stop: float, stopping criterion.
    klpq_submitted: float, KL divergence between the ground truth distribution (p) and the user submitted distribution (q).
    klqp_submitted: float, KL divergence between the user submitted distribution (q) and the ground truth distribution (p).    
    """
    q_opt: List[float]
    klpq_opt: float
    klqp_opt: float
    A: List[List[float]]
    iter_stop: int
    eps_stop: float
    klpq_submitted: float
    klqp_submitted: float

    def __post_init__(self):
        pass
        # raise NotImplementedError("ReplicateValidatorKL not implemented yet")


@dataclass_json
@dataclass
class MetricDistToDistValidator:
    '''
    Validate the output dictionary of one map to map metric in the the distribution-to-distribution pipeline.

    replicates: dict, dictionary of replicates.
    '''
    replicates: dict

    def validate_replicates(self, n_replicates):
        assert self.replicates.keys() == set(range(n_replicates))

        for replicate_idx, replicate in self.replicates.items():
            ReplicateValidatorEMD.from_dict(replicate["EMD"])
            ReplicateValidatorKL.from_dict(replicate["KL"])

        return


@dataclass_json
@dataclass
class DistributionToDistributionResultsValidator:
    '''
    Validate the output dictionary of the distribution-to-distribution pipeline.

    config: dict, input config dictionary.
    user_submitted_populations: torch.Tensor, user submitted populations, which sum to 1.
    id: str, submission id.
    fsc: dict, FSC distance results.
    bioem: dict, BioEM distance results.
    l2: dict, L2 distance results.
    corr: dict, correlation distance results.
    '''
    config: dict
    user_submitted_populations: torch.Tensor
    id: str
    fsc: Optional[dict] = None
    bioem: Optional[dict] = None
    res: Optional[dict] = None
    l2: Optional[dict] = None
    corr: Optional[dict] = None

    def __post_init__(self):
        validate_input_config_disttodist(self.config)

        for metric in self.config["metrics"]:
            # assert metric in self.__dict__.keys()
            assert self.__dict__[metric] is not None

        for metric in self.config["metrics"]:
            MetricDistToDistValidator.from_dict(
                self.__dict__[metric]
            ).validate_replicates(self.config["n_replicates"])
        return
