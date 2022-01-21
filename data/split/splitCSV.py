csvfile = open('../inat_odonata_usa.csv', 'r',encoding='utf8').readlines()
filename = 1
for i in range(len(csvfile)):
    if i % 175000 == 0:
        open(str(filename) + '.csv', 'w+',encoding='utf8').writelines(csvfile[i:i+1000])
        filename += 1
