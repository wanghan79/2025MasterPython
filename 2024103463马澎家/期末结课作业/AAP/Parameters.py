import os, pickle, json
import numpy as np

lines = open('/storage/htc/joshilab/jghhd/SC/stability_change/wild_feature_seq/parameter/AA13para.txt', 'r').readlines()
# sites253 = pickle.load(open('/home/gongjt057/stability_change/Reimplement/253mutation_sites.pickle','rb'))

# examples = pickle.load(open('/u1/home/jghhd/stability_change/Get_DSSPfile_Code/All_parse/Allmusite_dataset.out','rb'))
# examples = open('/u1/home/jghhd/stability_change/Get_DSSPfile_Code/Sep_parse/motify_dataset.out').readlines()
examples = open('/storage/htc/joshilab/jghhd/SC/stability_change/Data_pro/S1676').readlines()
# path = '/u1/home/jghhd/stability_change1/datasets_s1676/S1676.txt'
# pathTrain = '/u1/home/jghhd/stability_change1/datasets_s1676/subALLS1676.txt'
# examples = open('/u1/home/jghhd/stability_change1/datasets_s1676/S1676.txt').readlines()

para11 = {}
for line in lines:
    aaline = line.strip('\n').split()
    arr = aaline[1:]
    # a = list(map(float,a))
    a = []
    for item in arr:
        a.append(float(item))
    para11.update({aaline[0]: a})

print(para11)
pic = open('./parameters_list.json','w')
json.dump(para11,pic)
SEQpara11 = []
# for each in sites253:

for each in examples[1:]:
    print(each)
    # wild = each.split()[3][0]
    # muta = each.split()[3][-1]

    wild = each.split()[3][0]
    muta = each.split()[3][-1]

    DDpara = []
    for i in range(0, len(para11[muta])):
        DDpara.append(round(para11[muta][i] - para11[wild][i],3))

    # DDpara = list(np.array(para11[each[2]]) - np.array(para11[each[3]]))
    print(DDpara)
    SEQpara11.append(DDpara)
# print(SEQpara11[0])
print(len(SEQpara11))

pic = open('/storage/htc/joshilab/jghhd/SC/stability_change/wild_feature_seq/parameter/para_fea1676.json','w')
json.dump(SEQpara11,pic)
