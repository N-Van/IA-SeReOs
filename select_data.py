import os
import shutil

def select_data(src_folder, dest_folder, number, start, end):
    interval = (end-start)/(number+1)
    number_to_select = [*range(start, end, int(interval))]
    print('number to select:', len(number_to_select))
    files_to_select = []
    files_list = os.listdir(src_folder)
    for num in number_to_select:
        files_to_select.append([x for x in files_list if ('petreux'+str(num)+'.tif' == x or 'petreux'+'0'+str(num)+'.tif' == x)])
    files_to_select = [item for row in files_to_select for item in row]
    for file in files_to_select:
        file = src_folder+'/'+file
        shutil.copy(file, dest_folder)
    return files_to_select

    
# src_folder = '/home/mhelias004/IA-SeReOs-main/projet_sem/prediction3-5'
# dest_folder = '/home/mhelias004/IA-SeReOs-main/projet_sem/trainingsemi50/segmented/'
src_folder = '/home/mhelias004/IA-SeReOs-main/projet_sem/initial/original'
dest_folder = '/home/mhelias004/IA-SeReOs-main/projet_sem/trainingsemi50/original'
print((select_data(src_folder, dest_folder, 50, 150, 1500)))
