import glob, os, random
import torch, pickle
from pymol import cmd
from pymol import stored


from datasets.pdbbind import read_mol
from datasets.process_mols import read_molecule, get_rec_graph, generate_conformer, \
    get_lig_graph_with_matching, extract_receptor_structure, parse_receptor, parse_pdb_from_path

def main():

    propath = 'data/PDBBind_processed/'
    name  ='10gs' 
    nearresi = near_residues(propath, name)


    pdbobject = '10gs.pdb'
    cmd.delete('all')
    cmd.load(pdbobject)
    chains = cmd.get_chains()
    n_chains = len(chains)
    print(chains)
    stored.r = []
    stored.angle = 0
    # name: atomname, resn: name
    cmd.iterate('All', 'stored.r.append((chain,resi,resn))')
    id2name = {'{}_{}'.format(chain, resi):resn for (chain,resi,resn) in stored.r} # dict
    print(stored.r[:30])

    cmd.select('chain {} and resid {} and not h.'.format('A', 3))
    xyz = cmd.get_coords('sele', 1)
    print(xyz)
    sidec = 'N-CA-CB-CG'
    a1,a2,a3,a4 = sidec.split('-')
    chainid = 'A'
    resid='3'
    oriangle = cmd.get_dihedral('chain {} and resi {} and name {}'.format(chainid, resid, a1),'chain {} and resi {} and name {}'.format(chainid, resid, a2), 'chain {} and resi {} and  name {}'.format(chainid, resid, a3), 'chain {} and resi {} and name {}'.format(chainid, resid, a4))
    print(oriangle)
    angle = -83
    cmd.do('set_dihedral chain {} and resi {} and name {}, chain {} and resi {} and name {}, chain {} and resi {} and  name {}, chain {} and resi {} and name {}, {}'.format(chainid, resid, a1,chainid, resid, a2, chainid, resid, a3, chainid, resid, a4, angle))

    cmd.select('chain {} and resid {} and not h.'.format('A', 3))
    xyz = cmd.get_coords('sele', 1)
    print(xyz)
    cmd.save('test.pdb', selection='10gs', state=-1, format='', ref='',ref_state=-1, quiet=1, partial=0)



def near_residues(path, name):
    #ligpath = 'data/PDBBind_processed/'
    #name  ='10gs' 
    ligpath = path

    try:
        lig = read_mol(ligpath, name, remove_hs=False)
    except Exception as e:
        print('error:')
        print(e)
        return []
 
    assert lig is not None
    lig_coords = torch.from_numpy(lig.GetConformer().GetPositions()).float() # n,3

    pdbobject = f'{ligpath}/{name}/{name}_protein_processed.pdb'
    cmd.delete('all')
    cmd.load(pdbobject)
    xyz = cmd.get_coords(name, 1)
    procoord = torch.FloatTensor(xyz) # n1,3
    #distance

    dis = (lig_coords.unsqueeze(0)-procoord.unsqueeze(1))**2
    dis = torch.sqrt(torch.sum(dis,2))
    #print(dis.size())
    thres = 8
    nearflag = torch.sum(dis<thres,1)>0.5
    nearflag = nearflag.tolist()

    stored.r = []
    # name: atomname, resn: name
    cmd.iterate('All', 'stored.r.append((chain,resi,resn,name))')
    names = [f'{chain}_{resi}_{resn}_{name}' for (chain,resi,resn,name) in stored.r] # dict
    #print(len(names),names[:10])

    nearatoms = [name for name,flag in zip(names,nearflag) if flag]
    nearresi = set(['@'.join(a.split('_')[:3]) for a in nearatoms])

    return list(nearresi)

def gen(proname):

    sidefile = open('data/sidechains.txt')
    lines = sidefile.read().splitlines()
    name2sidechains = {}
    for line in lines:
        name,sidec = line.split()
        if name in name2sidechains:
            name2sidechains[name].append(sidec)
        else:
            name2sidechains[name] = [sidec]

    propath = 'data/PDBBind_processed/'
    pdbobject = f'{propath}/{proname}/{proname}_protein_processed.pdb'
    saveobject = f'{propath}/{proname}/{proname}_protein_fur_processed.pdb'
    savepurbed_residues = f'{propath}/{proname}/{proname}_protein_residues.pkl'
    objectname = f'{proname}_protein_processed'

    nearresi = near_residues(propath, proname)
    if len(nearresi)==0:return None


    prob_thres = 0.5

    cmd.delete('all')
    cmd.load(pdbobject)
    print(pdbobject)

    sidechain2change = {}
    print(nearresi)
    for resname1 in nearresi:
        chainid,resid, resname = resname1.split('@')
        if resname not in name2sidechains: continue
        for sidec in name2sidechains[resname]:
            #if random.random()>prob_thres:continue
            #print(resname1,sidec)
            a1,a2,a3,a4 = sidec.split('-')
            try:
                oriangle = cmd.get_dihedral('chain {} and resi {} and name {}'.format(chainid, resid, a1),'chain {} and resi {} and name {}'.format(chainid, resid, a2), 'chain {} and resi {} and  name {}'.format(chainid, resid, a3), 'chain {} and resi {} and name {}'.format(chainid, resid, a4))
            except Exception as e:
                print(f'error:')
                print(resname1)
                print(e)
                continue
            inc = random.uniform(-30,30)
            angle = oriangle+inc
            cmd.do('set_dihedral chain {} and resi {} and name {}, chain {} and resi {} and name {}, chain {} and resi {} and  name {}, chain {} and resi {} and name {}, {}'.format(chainid, resid, a1,chainid, resid, a2, chainid, resid, a3, chainid, resid, a4, angle))
            sidechain2change[resname1+f'_{sidec}'] = oriangle



    cmd.save(saveobject, selection=objectname, state=-1, format='', ref='',ref_state=-1, quiet=1, partial=0)

    #cmd.delete('all')
    #cmd.load('test1.pdb')
    #for sidechain,change in sidechain2change.items():

    #    info = sidechain.split('_')
    #    chainid,resid, resname = info[0].split('-')
    #    sidec = info[1]
    #    a1,a2,a3,a4 = sidec.split('-')
    #    cmd.do('set_dihedral chain {} and resi {} and name {}, chain {} and resi {} and name {}, chain {} and resi {} and  name {}, chain {} and resi {} and name {}, {}'.format(chainid, resid, 
    #        a1,chainid, resid, a2, chainid, resid, a3, chainid, resid, a4,change))


    #cmd.save('test.pdb', selection='test1',state=-1, format='', ref='',ref_state=-1, quiet=1, partial=0)

    filehandler = open(savepurbed_residues,"wb")
    pickle.dump(sidechain2change,filehandler)


def generate():
    propath = 'data/PDBBind_processed/'
    dirs = glob.glob(propath+'/*')

    for i, dirr in enumerate(dirs):
        name = dirr.split('/')[-1]
        print(f'--------{i}------------processing {name}')
        gen(name)





#main()
#near_residues()
#gen(name)
generate()
