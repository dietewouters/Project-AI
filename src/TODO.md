- [ ] Research SOTA for MLP architecture
- Finish `optimize.py`
- [ x ] Test on small dataset 
- [ ] Test on dataset using indices 
- [ ] Test in log-space 
- [ ] Research methods to deal with sparse vectors 
  - [ x ]Embedding 
  - [ ]PCA
- [ ] Test NN methods 
- [ ] SMILES to embedding instead of fingerprints using ChemBERTa (768 bits instead of 2048)
  - Can be interesting as it is more memory-efficient and brings contextual awareness, while fingerprints only 
    tell us whether a substructure is present
- [ ] Thinking about data leakage

Variations of our MLP
- [ x ] Base, nothing extra (already implemented, must figure out how many nodes)
- [ ] Use ChemBERTa as alternative for Morgan fingerprints 
- [ ] Add NN as extra guideline 
- [ ] Extra 