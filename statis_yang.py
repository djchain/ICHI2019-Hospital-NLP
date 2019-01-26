import os
import xlrd
path="E:\\Yue\\Entire Data\\CNMC\\hospital_data"

KEY="Pulse Check"

files= os.listdir(path)
os.chdir(path)
statis=[0 for i in range(235)]
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
                if KEY in str(s.cell(row, Activity_col).value):
                    dict.update({int(row+1): str("Pulse Check")})

                elif "Blood Pressure" in str(s.cell(row, Activity_col).value):
                    dict.update({int(row + 1): str("BP")})

                elif "Capillary" in str(s.cell(row, Activity_col).value):
                    dict.update({int(row + 1): str("CR")})

                elif "ABP" in str(s.cell(row, Activity_col).value):
                    dict.update({int(row + 1): str("ABP")})

                elif "Cardiac" in str(s.cell(row, Activity_col).value):
                    dict.update({int(row + 1): str("Cardiac")})

                elif "IV Placement" in str(s.cell(row, Activity_col).value):
                    dict.update({int(row + 1): str("IV Placement")})

                elif "osseous" in str(s.cell(row, Activity_col).value):
                    dict.update({int(row + 1): str("IO")})

                elif "Venous" in str(s.cell(row, Activity_col).value):
                    dict.update({int(row + 1): str("VC")})

                elif "Central Line" in str(s.cell(row, Activity_col).value):
                    dict.update({int(row + 1): str("CVL")})

                elif "Arterial" in str(s.cell(row, Activity_col).value):
                    dict.update({int(row + 1): str("AL")})

                elif "Bolus" in str(s.cell(row, Activity_col).value) or "bolus" in str(s.cell(row, Activity_col).value):
                    dict.update({int(row + 1): str("B")})

                elif "Hemorrhage" in str(s.cell(row, Activity_col).value):
                    dict.update({int(row + 1): str("EH")})

                elif "Pericardiocentesis" in str(s.cell(row, Activity_col).value):
                    dict.update({int(row + 1): str("PC")})

                elif "CPR" in str(s.cell(row, Activity_col).value):
                    dict.update({int(row + 1): str("CPR")})

                elif "Transfusion" in str(s.cell(row, Activity_col).value):
                    dict.update({int(row + 1): str("T")})

        dict=sorted(dict.items(), key=lambda item: item[0])
        print(dict)

        for index in range(1, len(dict)):
            if dict[index][1]==KEY and dict:
                for i in range(index):
                    statis[int(dict[index][0])-int(dict[i][0])] = statis[int(dict[index][0])-int(dict[i][0])]+1


print(statis)
with open("Pulse Check_SHUNXU.txt","w") as f:
    for i in range(len(statis)):
        f.write(str(statis[i])+" ")







