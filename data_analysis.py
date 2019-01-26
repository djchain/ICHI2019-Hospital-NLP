from data_preprocessing import data

analysis = data(path=r'E:/Yue/Entire Data/CNMC/hospital_data')
analysis.unclear_lbl.append('Monitor Vital Signs')
analysis.auto_process(merge_unclear=False)
