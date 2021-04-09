from dataclasses import dataclass
from copy import deepcopy
from typing import Dict, List
from high_cost_data_engine import config


@dataclass(init=False)
class ConvWithBatchNormConfig:
    conv_params: Dict
    batch_norm_params: Dict
    relu_params: Dict

    def __init__(self, conv: Dict, batch_norm: Dict, relu: Dict, **_):
        self.conv_params = conv
        self.batch_norm_params = batch_norm
        self.relu_params = relu


@dataclass(init=False)
class ResidualBlockConfig:
    conv_with_bn_config: ConvWithBatchNormConfig

    def __init__(self, conv: Dict, batch_norm: Dict, relu: Dict, **_):
        conv_copy = deepcopy(conv)
        conv_copy["in_channels"] = conv_copy["out_channels"]
        self.conv_with_bn_config = ConvWithBatchNormConfig(conv=conv_copy, batch_norm=batch_norm, relu=relu)


@dataclass(init=False)
class ResNet9Config:
    l0_conv_with_bn_config: ConvWithBatchNormConfig

    l1_conv_with_bn_config: ConvWithBatchNormConfig
    l1_max_pool_params: Dict
    l1_residual_config: ResidualBlockConfig

    l2_conv_with_bn_config: ConvWithBatchNormConfig
    l2_max_pool_params: Dict

    l3_conv_with_bn_config: ConvWithBatchNormConfig
    l3_max_pool_params: Dict
    l3_residual_config: ResidualBlockConfig

    final_max_pool_params: Dict
    linear_params: Dict
    scalar: Dict

    def __init__(self, config: Dict) -> None:
        model_config = config["model"]
        self.l0_conv_with_bn_config = ConvWithBatchNormConfig(**model_config["layers"]["layer_0"])

        self.l1_conv_with_bn_config = ConvWithBatchNormConfig(**model_config["layers"]["layer_1"])
        self.l1_max_pool_params = model_config["layers"]["layer_1"]["max_pool"]
        self.l1_residual_config = ResidualBlockConfig(**model_config["layers"]["layer_1"])

        self.l2_conv_with_bn_config = ConvWithBatchNormConfig(**model_config["layers"]["layer_2"])
        self.l2_max_pool_params = model_config["layers"]["layer_2"]["max_pool"]

        self.l3_conv_with_bn_config = ConvWithBatchNormConfig(**model_config["layers"]["layer_3"])
        self.l3_max_pool_params = model_config["layers"]["layer_3"]["max_pool"]
        self.l3_residual_config = ResidualBlockConfig(**model_config["layers"]["layer_3"])

        self.final_max_pool_params = model_config["layers"]["final"]["max_pool"]
        self.linear_params = model_config["layers"]["final"]["linear"]
        self.scalar = model_config["layers"]["final"]["scalar"]


@dataclass(init=False)
class OptimizerConfig:
    sgd_params: Dict
    _lr_scheduler_params: Dict

    def __init__(self, config: Dict) -> None:
        config = config["optimizer"]
        self.sgd_params = config["SGD"]
        self._lr_scheduler_params = config["lr_scheduler"]

    @property
    def lr_scheduler_epochs(self) -> List[int]:
        return list(self._lr_scheduler_params["epochs"].values())

    @property
    def lr_scheduler_values(self) -> List[float]:
        return list(self._lr_scheduler_params["values"].values())


@dataclass(init=False)
class DatasetConfig(config.Dataset):
    pad_params: Dict
    crop_params: Dict
    cutout_params: Dict

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        transforms = config["data_transformations"]
        self.pad_params = transforms["pad"]
        self.crop_params = transforms["crop"]
        self.cutout_params = transforms["cutout"]
