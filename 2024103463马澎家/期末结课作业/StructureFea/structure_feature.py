#seq features windows
import pickle,os
import numpy as np


def get_global_features(mutasitePath,alldsspparsed,rd = 10):
    mutafea = []
    examples = Open_pickle_file(mutasitePath)
    for example in examples:
        dis = global_dist(example,alldsspparsed)
        #print(dis)
        if dis == []:
            pass
        else:
            muta = get_feamatrix(dis, r=rd)
            mutafea.append(muta)
        #print(muta)
        #mutafea.append(muta)
    meanfea = np.nanmean(np.array(mutafea),axis=0)
    return mutafea,meanfea

def global_dist(example,alldsspparsed,dssppath=None,precise_match=True):
    pdb = example[0].split('_')[0]
    pdbchain = example[0].split('_')[1]
    mutasite = example[-1]
    mutasitekey = str(example[1]) + '_' + mutasite
    if  dssppath == None:
        pdbid =pdb
    else:
        pdbid= os.path.join(dssppath,pdb)
    cal_dist = True

    muxca,muyca,muzca = None,None,None
    if pdbid in alldsspparsed.keys():
        if pdbchain in alldsspparsed[pdbid].keys():
            if precise_match==True:
                if mutasitekey in alldsspparsed[pdbid][pdbchain].keys():
                    sitedssp = alldsspparsed[pdbid][pdbchain][mutasitekey]
                    muxca = sitedssp[-3]
                    muyca = sitedssp[-2]
                    muzca = sitedssp[-1]
                else:
                    cal_dist = False

            else:
                for eachsitekey in alldsspparsed[pdbid][pdbchain].keys():
                    if eachsitekey.split('_')[0] == str(example[1]):
                        sitedssp = alldsspparsed[pdbid][pdbchain][eachsitekey]
                        muxca = sitedssp[-3]
                        muyca = sitedssp[-2]
                        muzca = sitedssp[-1]
                        break
        else:
            cal_dist = False

        #check unformat residues in PDB files, such as missing residues
        if (muxca == None and muyca == None and muzca==None):
            resi_exit = False
        else:
            resi_exit = True


        mutadict1 = {}
        if cal_dist == False or resi_exit == False:
            mutad2 = []
            pass
        else:
            for eachsitekey in alldsspparsed[pdbid][pdbchain].keys():
                Xfea = alldsspparsed[pdbid][pdbchain][eachsitekey]
                a = abs(float(Xfea[-3]) - float(muxca))
                b = abs(float(Xfea[-2]) - float(muyca))
                c = abs(float(Xfea[-1]) - float(muzca))
                distance = (a * a + b * b + c * c) ** 0.5
                distance1 = round(distance, 4)
                # list1.append(distance1)
                mutadict1.update({eachsitekey: distance1})
                mutad2 = sorted(mutadict1.items(), key=lambda item: item[1])
    else:
        mutad2 = []

    return mutad2


def get_feamatrix(distencelist, r=10):
    pe = {'A': 0, 'V': 0, 'L': 0, 'I': 0, 'M': 0, 'C': 0, 'F': 0, 'W': 0, 'Y': 0, 'H': 0, 'S': 0, 'T': 0, 'N': 0,
          'Q': 0, 'K': 0, 'R': 0, 'D': 0, 'E': 0, 'G': 0, 'P': 0, 'X': 0}
    aa_count = 0
    aa_distence = []
    for z in range(1, len(distencelist)):
        eachitem = distencelist[z]
        if eachitem[1] <= r:
            aa_count = aa_count + 1
            # print(eachitem[0])
            try:
                pe[eachitem[0].split('_')[1]] = pe[eachitem[0].split('_')[1]] + 1
            except:
                pe['X'] = pe['X'] + 1

    for key in ['A', 'V', 'L', 'I', 'M', 'C', 'F', 'W', 'Y', 'H', 'S', 'T', 'N',
          'Q', 'K', 'R', 'D', 'E', 'G', 'P']:
        aa_distence.append(round(pe[key] / aa_count, 4))

    return aa_distence

def file_name(file_dir,fileformat):
    # fileformat : like .pdb or .dssp
    L=[]
    for dirpath, dirnames, filenames in os.walk(file_dir):
        #print(dirpath, dirnames, filenames)
        for file in filenames :
            if os.path.splitext(file)[1] == fileformat:
                L.append(os.path.join(dirpath, file))
    return L

def Open_pickle_file(picklefile):
    with open(picklefile,'rb') as f:
        content = pickle.load(f)
    return content


if __name__ == "__main__":

    path = '/storage/htc/joshilab/jghhd/SC/stability_change/1663_g2s/muta_all_site/'
    alldsspparsed = Open_pickle_file('/storage/htc/joshilab/jghhd/SC/stability_change/1663_g2s/DSSPfile/stability_dsspfea_muta_change.pic')
    print(len(alldsspparsed))

    muta_global_fea = {}
    mutasites = file_name(path,'.pickle')
    for onesite in mutasites:
        muta = onesite.split('/')[-1].split('_')[0]
        _,meanfea = get_global_features(onesite,alldsspparsed,rd = 20)

        print(len(meanfea))
        muta_global_fea.update({muta:meanfea})

    print(len(muta_global_fea))
    pickle.dump(muta_global_fea, open('/u1/home/jghhd/stability_change/1663_g2s/StructureFea/mutasite380_str_disfea20.pic','wb'), protocol=2)
    #pickle.dump(nonwrongdata_site, open('C:\\Myself\\GNN\\fromwang\\ease-mm\\datasets\\picklefile\\1662_datasite_update.pic','wb'), protocol=2)
    
