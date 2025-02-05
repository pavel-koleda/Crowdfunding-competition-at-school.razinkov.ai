from enum import IntEnum

SetType = IntEnum('SetType', ('train', 'validation', 'test'))
WeightsInitType = IntEnum('WeightsInitType', ('normal', 'uniform'))
PreprocessingType = IntEnum('PreprocessingType', ('normalization', 'standardization'))
LoggingParamType = IntEnum('LoggingParamType', ('loss', 'metric'))
LossType = IntEnum('LossType', ('sigmoid', 'softmax'))
RegularizationType = IntEnum('RegularizationType', ('lasso', 'ridge', 'lasso_ridge', 'none'))

# Clusterization
LinkageMethod = IntEnum('LinkageMethod', ('single', 'complete', 'average'))
StoppingCriteria = IntEnum('StoppingCriteria', ('distance', 'clusters_num'))
