

Elementy do zaadoptowania z DPVO `https://github.com/princeton-vl/DPVO`.


1) Feature extraction:
2 network (sumilar or identical, CNN + residual con):
features exctrator, contex ectractor

2) Patchifire:
Eextrac patches from places:
a) random
b) with strong echo intensity <- prefered

Also we should add current frame to graph, where n last frames (whole images) are kept.

for each patch represent it as:
- features for this patch, 
- contex for this patch
- pos vect [x, y, 1, d], where x, y -> pos of center of patch, d - inverse depth
(d can be init with:
- random value,
- based on mean d vals from other patches (my idea) <- prefered in first place
- based on other sensor (my idea) <- prefer for later development, development that dont need antoher training)


3) For extracted points:
2D sonar image point -> 3D space point -> move with expected shift and rotation of vehicle -> 2D sonar image point

expected shif can be: 
- estimated shift from previous iteration

4) Calculate error os estimation:
calc correlation between tranformed patch and its neighbour on whole frame





Feature / Context Extractor	Wyciąga głębokie cechy (deskryptory) z obrazu sonaru oraz kontekst dla GRU.	models.py -> BasicEncoder	PyTorch (CNN / ResNet)

Patch Sampler	Wybiera rzadki zbiór punktów (np. silne odbicia od dna), które będziemy śledzić.	dpvo.py -> patches / __init__	PyTorch

Sonar Projector (Warping)Przelicza, gdzie punkt $(r, \theta)$ z klatki A powinien znaleźć się w klatce B przy danym ruchu.projective_ops.py -> ProjectiveOpsPyTorch (Matematyka sonaru)

Correlation Sampler	Buduje piramidę podobieństwa (kosztu) między patchem a nowym obrazem.	altcorr/ (CUDA) lub CorrSampler	PyTorch (lub CUDA dla wydajności)

Update Block (GRU)Iteracyjnie wylicza poprawki ruchu ($\Delta$ pozycji) oraz pewność pomiaru.update.py -> UpdateBlockPyTorch (ConvGRU)

Weighting / Masking	Generuje wagi ufności (weights) dla każdego patcha, aby wykluczyć szumy i odbicia.	update.py -> mask	PyTorch (Sigmoid/Softmax)

Factor Graph (BA)	Optymalizuje globalnie całą trajektorię (Bundle Adjustment), łącząc wszystkie pomiary.	ba.py / FactorGraph	PyGTSAM (C++ pod maską)

Sliding Window Manager	Zarządza pamięcią: dodaje nowe klatki i usuwa te, które wyszły poza zasięg.	dpvo.py -> DPVO (pętla główna)	Python
