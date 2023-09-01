import os 

dir_name = 'pangyo_dataset'

file_list = os.listdir(dir_name)
print(file_list)

for idx, file_name in enumerate(sorted(file_list)):
    # file_name = file_name.rsplit('.PNG')[0]
    if 0 <= idx < 10:
        change_name = '000' + str(idx + 1)
    elif 10 <= idx < 100:
        change_name = '00' + str(idx+ 1)
    elif 100 <= idx < 999:
        change_name = '0' + str(idx + 1)
    elif idx == 999:
        change_name = str(idx + 1)

    os.rename(os.path.join(dir_name, file_name), os.path.join(dir_name, change_name + '.png')) 