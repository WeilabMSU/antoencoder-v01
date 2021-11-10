# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 19:22:56 2020

@author: gaoka
"""

import os
import multiprocessing as mp
from rdkit import Chem

def canonical_smile(sml):
    """Helper Function that returns the RDKit canonical SMILES for a input SMILES sequnce.
    Args:
        sml: SMILES sequence.
    Returns:
        canonical SMILES sequnce."""
    try: 
        cs=Chem.MolToSmiles(Chem.MolFromSmiles(sml), canonical=True)
    except:
        pass 
    else:
        return(cs)
    
fo=open("data-bin/smiles/250k_trainset-canonical.smi","w")

inputfilename = "data-bin/smiles/250k_trainset.smi"
cores = 8

def can_smiles_wrapper(chunkStart, chunkSize,filename):
    num = 0
    can_smiles=[]
    with open(filename) as f:
        f.seek(chunkStart)
        lines = f.read(chunkSize).splitlines()
        for l in lines:
            l=l.strip()
            l=l.split()
            if(l[0]=="InChI"):
                continue
            cs=canonical_smile(l[0])
            if(cs!=None):
                can_smiles.append(cs)
    return can_smiles

def chunkify(fname,size=1024*1024):
    fileEnd = os.path.getsize(fname)
    with open(fname,'r') as f:
        chunkEnd = f.tell()
        while True:
            chunkStart = chunkEnd
            f.seek(chunkStart+size,0)
            f.readline()
            chunkEnd = f.tell()
            yield chunkStart, chunkEnd - chunkStart
            if chunkEnd > fileEnd:
                break
            
pool = mp.Pool(cores)
jobs = []

for chunkStart, chunkSize in chunkify(inputfilename):
    jobs.append(pool.apply_async(can_smiles_wrapper, (chunkStart,chunkSize,inputfilename)))

res = []
for job in jobs:
    for m in job.get():
        res.append(m)

pool.close()

for m in res:
    print(m,file=fo)
