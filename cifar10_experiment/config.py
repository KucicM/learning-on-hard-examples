from dataclasses import dataclass, asdict
from typing import Dict, List
from high_cost_data_engine import utils


@dataclass
class ConvParams:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: int
    bias: bool


@dataclass
class BatchNormParams:
    num_features: int
    eps: float
    momentum: float


@dataclass
class ReluParams:
    inplace: bool


@dataclass(init=False)
class ConvWithBatchNormConfig:
    _conv_params: ConvParams
    _batch_norm_params: BatchNormParams
    _relu_params: ReluParams

    def __init__(self, conv: Dict, batch_norm: Dict, relu: Dict, **other):
        self._conv_params = ConvParams(**conv)
        self._batch_norm_params = BatchNormParams(**batch_norm)
        self._relu_params = ReluParams(**relu)

    @property
    def conv_params(self) -> Dict:
        return asdict(self._conv_params)

    @property
    def batch_norm_params(self) -> Dict:
        return asdict(self._batch_norm_params)

    @property
    def relu_params(self) -> Dict:
        return asdict(self._relu_params)


@dataclass(init=False)
class ResidualBlockConfig:
    conv_with_bn_config: ConvWithBatchNormConfig

    def __init__(self, conv: Dict, batch_norm: Dict, relu: Dict, **_):
        conv["in_channels"] = conv["out_channels"]
        self.conv_with_bn_config = ConvWithBatchNormConfig(conv=conv, batch_norm=batch_norm, relu=relu)


@dataclass
class MaxPoolParams:
    kernel_size: int
    stride: int


@dataclass
class LinearParams:
    in_features: int
    out_features: int
    bias: bool


@dataclass
class ScalarParams:
    weight: int


@dataclass(init=False)
class ResNet9Config:
    l0_conv_with_bn_config: ConvWithBatchNormConfig

    l1_conv_with_bn_config: ConvWithBatchNormConfig
    _l1_max_pool_params: MaxPoolParams
    l1_residual_config: ResidualBlockConfig

    l2_conv_with_bn_config: ConvWithBatchNormConfig
    _l2_max_pool_params: MaxPoolParams

    l3_conv_with_bn_config: ConvWithBatchNormConfig
    _l3_max_pool_params: MaxPoolParams
    l3_residual_config: ResidualBlockConfig

    _final_max_pool_params: MaxPoolParams
    _linear_params: LinearParams
    _scalar: ScalarParams

    def __init__(self, config: Dict) -> None:
        model_config = config["model"]
        self.l0_conv_with_bn_config = ConvWithBatchNormConfig(**model_config["layers"]["layer_0"])

        self.l1_conv_with_bn_config = ConvWithBatchNormConfig(**model_config["layers"]["layer_1"])
        self._l1_max_pool_params = MaxPoolParams(**model_config["layers"]["layer_1"]["max_pool"])
        self.l1_residual_config = ResidualBlockConfig(**model_config["layers"]["layer_1"])

        self.l2_conv_with_bn_config = ConvWithBatchNormConfig(**model_config["layers"]["layer_2"])
        self._l2_max_pool_params = MaxPoolParams(**model_config["layers"]["layer_2"]["max_pool"])

        self.l3_conv_with_bn_config = ConvWithBatchNormConfig(**model_config["layers"]["layer_3"])
        self._l3_max_pool_params = MaxPoolParams(**model_config["layers"]["layer_3"]["max_pool"])
        self.l3_residual_config = ResidualBlockConfig(**model_config["layers"]["layer_3"])

        self._final_max_pool_params = MaxPoolParams(**model_config["layers"]["final"]["max_pool"])
        self._linear_params = LinearParams(**model_config["layers"]["final"]["linear"])
        self._scalar = ScalarParams(**model_config["layers"]["final"]["scalar"])

    @property
    def l1_max_pool_params(self) -> Dict:
        return asdict(self._l1_max_pool_params)

    @property
    def l2_max_pool_params(self) -> Dict:
        return asdict(self._l2_max_pool_params)

    @property
    def l3_max_pool_params(self) -> Dict:
        return asdict(self._l3_max_pool_params)

    @property
    def final_max_pool_params(self) -> Dict:
        return asdict(self._final_max_pool_params)

    @property
    def linear_params(self) -> Dict:
        return asdict(self._linear_params)

    @property
    def scalar_params(self) -> Dict:
        return asdict(self._scalar)


@dataclass(init=False)
class OptimizerConfig:
    base_weight_decay: float
    momentum: float
    nesterov: bool
    lr_peek_epoch: int
    lr_peek_value: float

    def __init__(self, config: Dict) -> None:
        config = config["optimizer"]
        self.base_weight_decay = config["base_weight_decay"]
        self.momentum = config["momentum"]
        self.nesterov = config["nesterov"]
        self.lr_peek_epoch = config["lr_peek_epoch"]
        self.lr_peek_value = config["lr_peek_value"]


@dataclass
class PadParams:
    padding: int
    fill: int
    padding_mode: str


@dataclass
class CropParams:
    size: int


@dataclass
class CutoutParams:
    size: int


@dataclass(init=False)
class NormalizeParams:
    mean: List[float]
    std: List[float]

    def __init__(self, mean, std, **_) -> None:
        self.mean = mean
        self.std = std


@dataclass(init=False)
class DatasetConfig:
    data_path: str
    _norm_params: NormalizeParams
    _pad_params: PadParams
    _crop_params: CropParams
    _cutout_params: CutoutParams

    def __init__(self, config: Dict) -> None:
        dataset_config = config["dataset"]
        self.data_path = dataset_config["data_path"]
        self._norm_params = NormalizeParams(**dataset_config)
        transforms = config["data_transformations"]
        self._pad_params = PadParams(**transforms["pad"])
        self._crop_params = CropParams(**transforms["crop"])
        self._cutout_params = CutoutParams(**transforms["cutout"])

    @property
    def norm_params(self) -> Dict:
        return asdict(self._norm_params)

    @property
    def pad_params(self) -> Dict:
        return asdict(self._pad_params)

    @property
    def crop_params(self) -> Dict:
        return asdict(self._crop_params)

    @property
    def cutout_params(self) -> Dict:
        return asdict(self._cutout_params)


@dataclass
class DataLoaderParams:
    batch_size: int
    num_workers: int
    # high_cost_selection_setting: int = None


@dataclass(init=False)
class DataLoaderConfig:
    _train_params: DataLoaderParams

    def __init__(self, config: Dict) -> None:
        self._train_params = DataLoaderParams(**config["dataloader"])

    @property
    def batch_size(self) -> int:
        return self._train_params.batch_size

    @property
    def train_params(self) -> Dict:
        return asdict(self._train_params)

    @property
    def test_params(self) -> Dict:
        ret = asdict(self._train_params)
        # ret.pop("high_cost_selection_setting")
        return ret
