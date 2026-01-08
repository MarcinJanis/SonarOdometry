

Elementy do zaadoptowania z DPVO `https://github.com/princeton-vl/DPVO`.

Moduł,Funkcja i cel,Odpowiednik w DPVO (Plik / Klasa),Najlepsze narzędzie
Feature / Context Extractor,Wyciąga głębokie cechy (deskryptory) z obrazu sonaru oraz kontekst dla GRU.,models.py -> BasicEncoder,PyTorch (CNN / ResNet)
Patch Sampler,"Wybiera rzadki zbiór punktów (np. silne odbicia od dna), które będziemy śledzić.",dpvo.py -> patches / __init__,PyTorch
Sonar Projector (Warping),"Przelicza, gdzie punkt (r,θ) z klatki A powinien znaleźć się w klatce B przy danym ruchu.",projective_ops.py -> ProjectiveOps,PyTorch (Matematyka sonaru)
Correlation Sampler,Buduje piramidę podobieństwa (kosztu) między patchem a nowym obrazem.,altcorr/ (CUDA) lub CorrSampler,PyTorch (lub CUDA dla wydajności)
Update Block (GRU),Iteracyjnie wylicza poprawki ruchu (Δ pozycji) oraz pewność pomiaru.,update.py -> UpdateBlock,PyTorch (ConvGRU)
Weighting / Masking,"Generuje wagi ufności (weights) dla każdego patcha, aby wykluczyć szumy i odbicia.",update.py -> mask,PyTorch (Sigmoid/Softmax)
Factor Graph (BA),"Optymalizuje globalnie całą trajektorię (Bundle Adjustment), łącząc wszystkie pomiary.",ba.py / FactorGraph,PyGTSAM (C++ pod maską)
Sliding Window Manager,"Zarządza pamięcią: dodaje nowe klatki i usuwa te, które wyszły poza zasięg.",dpvo.py -> DPVO (pętla główna),Python
