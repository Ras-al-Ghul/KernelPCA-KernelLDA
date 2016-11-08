# KernelPCA-KernelLDA-SVM

To run: `python filename.py`  

The datasets used for this are:  
- The Arcene dataset <https://archive.ics.uci.edu/ml/datasets/Arcene>  
- The Madelon dataset <https://archive.ics.uci.edu/ml/datasets/Madelon>  

**Kernel Function used : RBF kernel**  

1. `kernelpca.py` - This implements the kernel PCA technique. The kernel used here is the `RBF` kernel. `numoffeatures` indicates the number of features in the train data file. For the Arcene dataset it is 10000. `newnumoffeatures` indicates the `k` which is the value of the new number of dimensions (Please take care to keep this value lesser than the number of train samples as otherwise it leads to eigen vectors not being found). `numofdata` indicates the number of samples to be used for this dimensionality reduction.  
2. `kernellda.py` - Implemented from <https://en.wikipedia.org/wiki/Kernel_Fisher_discriminant_analysis> `numoffeatures`, `newnumoffeatures` (should be `1` for 1D LDA space) and `numofdata` carry the same meaning as above. This implements the kernel LDA technique.  

For the above two files, the Madelon dataset can also be used. Modify the above files based on the info given above.  

3. `svmkernelpca.py` - `numoffeatures`, `newnumoffeatures`, `numofdata` carry the same meaning as above. `numofvaliddata` indicates the number of samples to be considered for testing - from the valid files. There are two sets of the above values, one to be commented out, one for the Arcene dataset and the other for the Madelon dataset. Please comment out/uncomment appropriate sections of the code (`4 lines`) in the `__main__` method. `svm` is used to classify the data. The kernel used for the SVM is again the `rbf` kernel.  

4. `svmkernellda.py` - `numoffeatures`, `newnumoffeatures` (should be `1` for the 1D LDA space), `numofdata`, `numofvaliddata` carry the same meaning as above. Again, SVM with the rbf kernel is used.
