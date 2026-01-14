from dataclasses import dataclass

@dataclass(frozen=True)
class NUCConfig:
    NUC_TAB = {'A': 6, 'C': 8, 'G': 9, 'T': 7}
    ACGT_IDX = [6, 8, 9, 7]

# 全局唯一实例
NUC_CONFIG = NUCConfig()