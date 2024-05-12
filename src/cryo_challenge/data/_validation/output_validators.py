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
    config: dict
    user_submitted_populations: torch.Tensor
    corr: Optional[dict] = None
    l2: Optional[dict] = None
    bioem: Optional[dict] = None
    fsc: Optional[dict] = None

    def __post_init__(self):
        validate_input_config_mtm(self.config)

        for metric in self.config["metrics"]:
            assert self.__dict__[metric] is not None

        for metric in self.config["metrics"]:
            validate_maptomap_result(self.__dict__[metric])

        return


@dataclass_json
@dataclass
class ReplicateValidatorEMD:
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
    config: dict
    user_submitted_populations: torch.Tensor
    id: str
    fsc: Optional[dict] = None
    bioem: Optional[dict] = None
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
