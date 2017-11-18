# WD pipeline K2
K2 photometry optimized for WDs which reads, reduces and detrends pixel target files and search for transiting/eclipsing planets and other substellar bodies.

How to use this github:
(1) download all files as a zip
(2) unpack the zip file
(3) run the long cadence pipeline by going to the lc_scripts directory and using the command 'python main.py' in a terminal.
(4) run the short cadence pipeline (it is important to run this pipeline afterwards, since output of the long cadence pipeline is used) by going to the sc_scripts directory and using the command 'python main.py' in a terminal.

More information on the content of the folders:
- lc_data: contains two long cadence pixel target files. These are used as examples to illustrate the performance of the pipeline. The input is pixel files which can be downloaded from the MAST database (https://archive.stsci.edu/k2/data_search/search.php). 
- sc_data: contains two short cadence pixel target files. These are used as examples to illustrate the performance of the pipeline.  The input is pixel files which can be downloaded from the MAST database (https://archive.stsci.edu/k2/data_search/search.php).
- lc_output: folder where the long cadence output of the pipeline is stored.
- sc_output: folder where the short cadence output of the pipeline is stored.
- lc_scripts: python scripts which make up the long cadence pipeline. To run the pipeline use the command 'python main.py' while in this folder.
- sc-scripts: python scripts which make up the short cadence pipeline. To run the pipeline use the command 'python main.py' while in this folder.

Feel free to use and modify the code here!

For questions or commends please send an email vansluijs@strw.leidenuniv.nl.

Disclaimer: The BLS algorithm used to detect periodic events is taken from a Python implementation by Ruth Angus and Dan Foreman-Mackey (see https://github.com/dfm/python-bls), and repeated here. A transit model using Mandel & Agol models is included for completeness but not a part of the pipeline, it was implented in Python by Ian Crossfield (http://www.lpl.arizona.edu/~ianc/python/transit.html)
