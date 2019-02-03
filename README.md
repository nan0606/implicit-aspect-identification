# Code and data for Non-negative Matrix Factorization for Implicit Aspect Identification

### Requirements
  * NumPy = 1.15.3
  * stop_words
  * nltk
  
### Read data
read datasets in /data/ by 'read_raw_text.py' and read datasets in /souce/ by 'read_semeval.py'

### extract aspect and opinion words
extract dependency relations by 'dependency_parser.py'
apply Double Propagation by 'double_propagation.py'

### predition and identification
run 'nmf.py' to cluster aspects and predict the implicit aspect
