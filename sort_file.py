import os
import glob

kim = glob.glob(r'C:\Users\Timaz\Documents\ECAM\ETS\MTI830-Forage de texte\projet\*.mp4')
j = 0
for index, i in enumerate(kim):
    j = index / 7
    print(index%7, j)
    new_index = "TIM_%d_0%d.mp4" %(j,index%7)
    print(new_index)
    os.rename(kim[index],new_index)


