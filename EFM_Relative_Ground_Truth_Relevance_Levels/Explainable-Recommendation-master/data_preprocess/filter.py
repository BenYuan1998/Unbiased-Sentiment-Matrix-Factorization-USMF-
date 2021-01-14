file = "../data/feature-oponion.txt"
with open(file) as data_file:
    list =[]
    for line in data_file:
        feature = line.split(':')[0]
        oponions = line.split(':')[1]
        if(len(feature) <= 1):
            continue
        if 40 >= len(oponions.split(',')) > 10:
            list.append(line)
    with open('lexicon.txt', 'w') as file:
        for line in list:
            file.write(line)