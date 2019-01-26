import os
import xlrd
from xlrd import xldate_as_tuple
import math
from datetime import datetime
path="E:\\Yue\\Entire Data\\CNMC\\hospital_data"

KEY="Blood Pressure"

files= os.listdir(path)
os.chdir(path)
statis=[0 for i in range(35)]
for file in files:
    if "xlsx" in file and not "~$" in file:
        print(file)
        dict = {}

        wb = xlrd.open_workbook(file)
        for s in wb.sheets():
            ncols = s.ncols
            for col in range(ncols):
                if "Activity" in str(s.cell(0, col).value):
                    Activity_col=col

            nrows = s.nrows
            for row in range(1, nrows):
                # if "pupils" in str(s.cell(row, Activity_col).value) or "Pupils" in str(s.cell(row, Activity_col).value):
                #     year, month, day, hour, minute, second = xldate_as_tuple(s.cell(row, 1).value, wb.datemode)
                #     start = hour*3600 + minute*60 + second
                #     dict.update({start: KEY})
                #
                # elif "GCS" in str(s.cell(row, Activity_col).value):
                #     year, month, day, hour, minute, second = xldate_as_tuple(s.cell(row, 1).value, wb.datemode)
                #     start = hour*3600 + minute*60 + second
                #     dict.update({start: str("GCS")})
                #
                # elif "spine" in str(s.cell(row, Activity_col).value) or "Spine" in str(s.cell(row, Activity_col).value):
                #     year, month, day, hour, minute, second = xldate_as_tuple(s.cell(row, 1).value, wb.datemode)
                #     start = hour*3600 + minute*60 + second
                #     dict.update({start: str("CS")})
                #

                if KEY in str(s.cell(row, Activity_col).value):
                    year, month, day, hour, minute, second = xldate_as_tuple(s.cell(row, 1).value, wb.datemode)
                    start = hour * 3600 + minute * 60 + second
                    dict.update({start: KEY})

                elif "Pulse Check" in str(s.cell(row, Activity_col).value):
                    year, month, day, hour, minute, second = xldate_as_tuple(s.cell(row, 1).value, wb.datemode)
                    start = hour * 3600 + minute * 60 + second
                    dict.update({start: str("Pulse Check")})

                elif "Capillary" in str(s.cell(row, Activity_col).value):
                    year, month, day, hour, minute, second = xldate_as_tuple(s.cell(row, 1).value, wb.datemode)
                    start = hour * 3600 + minute * 60 + second
                    dict.update({start: str("CR")})

                elif "ABP" in str(s.cell(row, Activity_col).value):
                    year, month, day, hour, minute, second = xldate_as_tuple(s.cell(row, 1).value, wb.datemode)
                    start = hour * 3600 + minute * 60 + second
                    dict.update({start: str("ABP")})

                elif "Cardiac" in str(s.cell(row, Activity_col).value):
                    year, month, day, hour, minute, second = xldate_as_tuple(s.cell(row, 1).value, wb.datemode)
                    start = hour * 3600 + minute * 60 + second
                    dict.update({start: str("Cardiac")})

                elif "IV Placement" in str(s.cell(row, Activity_col).value):
                    year, month, day, hour, minute, second = xldate_as_tuple(s.cell(row, 1).value, wb.datemode)
                    start = hour * 3600 + minute * 60 + second
                    dict.update({start: str("IV Placement")})

                elif "osseous" in str(s.cell(row, Activity_col).value):
                    year, month, day, hour, minute, second = xldate_as_tuple(s.cell(row, 1).value, wb.datemode)
                    start = hour * 3600 + minute * 60 + second
                    dict.update({start: str("IO")})

                elif "Venous" in str(s.cell(row, Activity_col).value):
                    year, month, day, hour, minute, second = xldate_as_tuple(s.cell(row, 1).value, wb.datemode)
                    start = hour * 3600 + minute * 60 + second
                    dict.update({start: str("VC")})

                elif "Central Line" in str(s.cell(row, Activity_col).value):
                    year, month, day, hour, minute, second = xldate_as_tuple(s.cell(row, 1).value, wb.datemode)
                    start = hour * 3600 + minute * 60 + second
                    dict.update({start: str("CVL")})

                elif "Arterial" in str(s.cell(row, Activity_col).value):
                    year, month, day, hour, minute, second = xldate_as_tuple(s.cell(row, 1).value, wb.datemode)
                    start = hour * 3600 + minute * 60 + second
                    dict.update({start: str("AL")})

                elif "Bolus" in str(s.cell(row, Activity_col).value) or "bolus" in str(s.cell(row, Activity_col).value):
                    year, month, day, hour, minute, second = xldate_as_tuple(s.cell(row, 1).value, wb.datemode)
                    start = hour * 3600 + minute * 60 + second
                    dict.update({start: str("B")})

                elif "Hemorrhage" in str(s.cell(row, Activity_col).value):
                    year, month, day, hour, minute, second = xldate_as_tuple(s.cell(row, 1).value, wb.datemode)
                    start = hour * 3600 + minute * 60 + second
                    dict.update({start: str("EH")})

                elif "Pericardiocentesis" in str(s.cell(row, Activity_col).value):
                    year, month, day, hour, minute, second = xldate_as_tuple(s.cell(row, 1).value, wb.datemode)
                    start = hour * 3600 + minute * 60 + second
                    dict.update({start: str("PC")})

                elif "CPR" in str(s.cell(row, Activity_col).value):
                    year, month, day, hour, minute, second = xldate_as_tuple(s.cell(row, 1).value, wb.datemode)
                    start = hour * 3600 + minute * 60 + second
                    dict.update({start: str("CPR")})

                elif "Transfusion" in str(s.cell(row, Activity_col).value):
                    year, month, day, hour, minute, second = xldate_as_tuple(s.cell(row, 1).value, wb.datemode)
                    start = hour * 3600 + minute * 60 + second
                    dict.update({start: str("T")})

        dict=sorted(dict.items(), key=lambda item: item[0])
        print(dict)

        for index in range(1, len(dict)):
            if dict[index][1] == KEY and dict:
                for i in range(index):
                    statis[int(math.ceil((int(dict[index][0])-int(dict[i][0]))/30.0))] = statis[int(math.ceil((int(dict[index][0])-int(dict[i][0]))/30.0))]+1

print(statis)
with open("Blood Pressure_time.txt","w") as f:
    for i in range(len(statis)):
        f.write(str(statis[i])+" ")







