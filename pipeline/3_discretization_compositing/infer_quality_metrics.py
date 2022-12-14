from inferQualityMetrics import inferQualityMetrics
from getIDsFromFolder import getIDsFromFolder
from inferQualityMetrics import metricPairwiseRowMean

ids = getIDsFromFolder("E:/dragonfly-patterner/data/patterns")
ids = ids[0:9]

inferQualityMetrics(ids, group_records_col=None,metrics=["mse_pat","emd_pat","struct_diff_pat","symmetry_pat","light_glared_seg"])
print(ids)
print(len(ids))
#inferQualityMetrics(ids, group_records_col=None,metrics=["mse_pat","emd_pat","struct_diff_pat","symmetry_pat","light_glared_seg"])
